from datetime import datetime
import logging
import os
import shutil
import time
from tqdm import tqdm
import yaml

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader

from src.aux import EarlyStopper, longtiming
from src.data import LazyDataset, Reader, Scaler
from src.convresnet import ConvResNet

logger = logging.getLogger(__name__)


def fit(
    model: ConvResNet,
    *,
    train_dataloader,
    valid_dataloader,
    optimizer,
    num_epochs,
    patience,
    min_delta,
    update_every=1,
    batch_split=1,
    checkpoint=None,
    shuffle=False,
    device=None,
    dtype=None,
):
    """
    Fit function for `ConvResNet` model.

    Inputs:
        * model: model to train, instance of `ConvResNet`.
        * train_dataloader: dataloader for training dataset (instance of PyTorch's `DataLoader`).
        * valid_dataloader: dataloader for validating dataset (instance of PyTorch's `DataLoader`).
        * optimizer: optimizer for updating model parameters (instance of `torch.optim`).
        * num_epochs: number of epochs to train for.
        * patience: number of updates (epochs) to wait before early stopping (see `EarlyStopper`).
        * min_delta: minimum improvement in validation loss before early stopping (see `EarlyStopper`).
        * update_every: number of epochs between parameter updates.
        * batch_split: batch splitting factor. Effectively reduces batch size at runtime to fit on GPU.
            Alternatively, batch size can be reduced at data creation (see `src/data.py`).
        * device
        * dtype
    """
    factory_kwargs = {"device": device, "dtype": dtype}

    start = time.time()

    now = datetime.now()
    dt = now.strftime("%d-%m-%Y %H:%M:%S")
    print(f"\nSTARTING @ {dt}\n")

    stopper = EarlyStopper(patience=patience, min_delta=min_delta)

    for epoch in range(1, num_epochs + 1):
        epoch_start = time.time()
        train_loss = 0.0
        valid_loss = 0.0

        model.train()
        optimizer.zero_grad()
        for i, (signals, labels, _) in enumerate(tqdm(train_dataloader), start=1):
            # squeeze(0) to remove 'fake' batch_size=1 when working with pre-batched files
            # transpose(-1,-2) to change from (...(N), length, channels) to (...(N), channels, length),
            #     required by `torch.nn.Conv1d` inside `ConvResNet` (see https://pytorch.org/docs/stable/generated/torch.nn.Conv1d.html)
            signals = signals.squeeze(0).transpose(-1, -2)
            labels = labels.squeeze(0)

            if shuffle:
                p = torch.randperm(signals.shape[0])
                signals = signals[p]
                labels = labels[p]

            if batch_split > 1:
                signals = signals.reshape(batch_split, -1, *signals.shape[1:])
                labels = labels.reshape(batch_split, -1, *labels.shape[1:])

                for s, l in zip(signals, labels):
                    s = s.to(**factory_kwargs)
                    l = l.to(
                        device=factory_kwargs["device"], dtype=torch.int64
                    )  # int64 is required for backward

                    output = model(s, raw=True)
                    loss = F.cross_entropy(output, l)
                    loss.backward()
                    train_loss += loss.item()

            else:
                signals = signals.to(**factory_kwargs)
                labels = labels.to(device=factory_kwargs["device"], dtype=torch.int64)

                output = model(signals, raw=True)
                loss = F.cross_entropy(output, labels)
                loss.backward()
                train_loss += loss.item()

            if i % update_every == 0 or i == len(train_dataloader):
                optimizer.step()
                optimizer.zero_grad()

        train_loss /= batch_split * len(train_dataloader)

        model.eval()
        with torch.no_grad():
            for signals, labels, _ in tqdm(valid_dataloader):
                signals = signals.squeeze(0).transpose(-1, -2)
                labels = labels.squeeze(0)

                if batch_split > 1:
                    signals = signals.reshape(batch_split, -1, *signals.shape[1:])
                    labels = labels.reshape(batch_split, -1, *labels.shape[1:])

                    for s, l in zip(signals, labels):
                        s = s.to(**factory_kwargs)
                        l = l.to(device=factory_kwargs["device"], dtype=torch.int64)

                        output = model(s, raw=True)
                        loss = F.cross_entropy(output, l)
                        valid_loss += loss.item()

                else:
                    signals = signals.to(**factory_kwargs)
                    labels = labels.to(
                        device=factory_kwargs["device"], dtype=torch.int64
                    )

                    output = model(signals, raw=True)
                    loss = F.cross_entropy(output, labels)
                    valid_loss += loss.item()

        valid_loss /= batch_split * len(valid_dataloader)

        now = time.time()
        print(
            f"\nEPOCH #{epoch} out of {num_epochs} | ET (EPOCH ET):  {longtiming(now - start)} ({longtiming(now-epoch_start)})"
        )
        print(f"TRAIN LOSS: {train_loss:.6f} | VALID LOSS: {valid_loss:.6f}\n\n")

        stop = stopper(valid_loss, model.state_dict())

        if checkpoint is not None:
            stopper.save(checkpoint)

        if stop:
            print("\nEARLY STOPPING\n")
            break

    now = time.time()
    print(f"\nTRAINING COMPLETE\nTOTAL ET:   {longtiming(now - start)}\n\n")


if __name__ == "__main__":

    rootdir = os.path.dirname(os.path.realpath(__file__))
    baserundir = os.path.join(rootdir, "runs")
    if not os.path.exists(baserundir):
        os.makedirs(baserundir)

    with open(os.path.join(rootdir, "config.yaml"), "r") as f:
        config = yaml.safe_load(f)

    now = datetime.now()
    rundir = os.path.join(baserundir, f"{now.strftime('%Y%m%d-%H%M')}")

    if os.path.exists(rundir):
        remove_existing = True
        shutil.rmtree(rundir)
    else:
        remove_existing = False
    os.makedirs(rundir)

    logging.basicConfig(
        filename=os.path.join(rundir, "runinfo.log"), level=logging.INFO
    )

    if remove_existing:
        logger.warning(f"Removing existing directory: {os.path.abspath(rundir)}")

    logger.info(f"Run directory: {os.path.abspath(rundir)}")

    shutil.copy(
        os.path.join(rootdir, "config.yaml"), os.path.join(rundir, "config.yaml")
    )

    DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    DTYPE = torch.bfloat16
    factory_kwargs = {"device": DEVICE, "dtype": DTYPE}

    datadir = config["paths"]["datadir"]
    scaler_path = os.path.join(rundir, config["paths"]["scaler"])
    model_path = os.path.join(rundir, config["paths"]["model"])
    ckpt_path = (
        os.path.splitext(model_path)[0] + "__CKPT" + os.path.splitext(model_path)[1]
    )
    model_kwargs = config["model"]

    model = ConvResNet(
        in_channels=1,
        out_features=4,
        activation=nn.ReLU,
        **model_kwargs,
        **factory_kwargs,
    )
    optimizer = optim.AdamW(model.parameters(), **config["optimizer"])

    print(
        f"\nConvResNet blocks-{(model_kwargs['num_blocks'])}  -  MODEL DESCRIPTION:\n\n",
        model,
        "\n\n",
    )

    reader = Reader(datadir)
    scaler = Scaler(norm=config["scaler"], dim=-1)
    scaler.lazy_fit(reader(key="train"), **config["loaders"])
    scaler.save(scaler_path)
    #print(f"{os.path.abspath(datadir)=}")
    #print(f"{os.path.join(datadir, 'p*')=}")
    #print(f"{os.getcwd()=}")
    train_dataset = LazyDataset(reader(key="train"), scaler=scaler)
    valid_dataset = LazyDataset(reader(key="valid"), scaler=scaler)

    train_dataloader = DataLoader(
        train_dataset, batch_size=1, shuffle=True, **config["loaders"]
    )
    valid_dataloader = DataLoader(
        valid_dataset, batch_size=1, shuffle=True, **config["loaders"]
    )

    fit(
        model=model,
        train_dataloader=train_dataloader,
        valid_dataloader=valid_dataloader,
        optimizer=optimizer,
        checkpoint=ckpt_path,
        shuffle=True,
        **config["fit"],
        **factory_kwargs,
    )

    model.save(model_path)
    print(f"\n\nMODEL SAVED SUCCESSFULLY TO:\n{os.path.abspath(model_path)}\n")
    logger.info(f"\n\nMODEL SAVED SUCCESSFULLY TO:\n{os.path.abspath(model_path)}\n")
    os.remove(ckpt_path)
