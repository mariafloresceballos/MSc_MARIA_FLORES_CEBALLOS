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
from torch.utils.data import DataLoader, RandomSampler

from src.aux import EarlyStopper, longtiming
from src.data import LazyDataset, Reader
from src.resnext import ResNeXt

logger = logging.getLogger(__name__)


def get_dataloaders(
    datasets,
    *,
    num_batches_per_epoch=None,
    batch_size=1,
    num_workers=0,
    prefetch_factor=None,
):
    dataloaders = {
        k: DataLoader(
            ds,
            shuffle=True if num_batches_per_epoch is None else None,
            sampler=(
                RandomSampler(ds, replacement=False, num_samples=num_batches_per_epoch)
                if num_batches_per_epoch is not None
                else None
            ),
            batch_size=batch_size,
            num_workers=num_workers,
            prefetch_factor=prefetch_factor,
        )
        for k, ds in datasets.items()
    }

    return dataloaders


def fit(
    model: ResNeXt,
    *,
    train_dataloader,
    valid_dataloader,
    optimizer,
    num_epochs,
    patience,
    min_delta,
    update_every=1,
    checkpoint=None,
    shuffle=False,
    device=None,
    dtype=None,
):
    """
    Fit function for `ResNeXt` model.

    Inputs:
        * model: model to train, instance of `ResNeXt`.
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
            signals = (
                signals.reshape(-1, *signals.shape[-2:])
                .transpose(-1, -2)
                .to(**factory_kwargs)
            )
            labels = labels.reshape(-1, *labels.shape[-2:]).to(
                device=factory_kwargs["device"], dtype=torch.int64
            )

            if shuffle:
                p = torch.randperm(signals.shape[0])
                signals = signals[p]
                labels = labels[p]

            output = model(signals, raw=True)
            loss = F.cross_entropy(output, labels, reduction="mean")
            loss.backward()
            train_loss += loss.item()

            if i % update_every == 0 or i == len(train_dataloader):
                optimizer.step()
                optimizer.zero_grad()

        train_loss /= len(train_dataloader)

        model.eval()
        with torch.no_grad():
            for signals, labels, _ in tqdm(valid_dataloader):
                signals = (
                    signals.reshape(-1, *signals.shape[-2:])
                    .transpose(-1, -2)
                    .to(**factory_kwargs)
                )
                labels = labels.reshape(-1, *labels.shape[-2:]).to(
                    device=factory_kwargs["device"], dtype=torch.int64
                )

                output = model(signals, raw=True)
                loss = F.cross_entropy(output, labels, reduction="mean")
                valid_loss += loss.item()

        valid_loss /= len(valid_dataloader)

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

    with open(os.path.join(rootdir, "configs", "resnext.yaml"), "r") as f:
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
        os.path.join(rootdir, "configs", "resnext.yaml"),
        os.path.join(rundir, "resnext.yaml"),
    )

    DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    DTYPE = torch.bfloat16
    factory_kwargs = {"device": DEVICE, "dtype": DTYPE}

    datadir = config["datadir"]
    model_path = os.path.join(rundir, config["model"])
    ckpt_path = (
        os.path.splitext(model_path)[0] + "__CKPT" + os.path.splitext(model_path)[1]
    )

    model = ResNeXt(
        in_channels=1,
        out_features=4,
        activation=nn.SiLU,
        **config["resnext"],
        **factory_kwargs,
    )
    optimizer = optim.AdamW(model.parameters(), **config["optimizer"])

    print(
        f"\nResNeXt-{sum(config['resnext']['num_blocks']) * 3 + 2}  -  MODEL DESCRIPTION:\n\n",
        model,
        "\n\n",
    )

    reader = Reader(datadir)
    datasets = {
        k: LazyDataset(files=reader(key=k), label=True) for k in ["train", "valid"]
    }

    dataloaders = get_dataloaders(
        datasets,
        num_batches_per_epoch=config["num_batches_per_epoch"],
        **config["loader"],
    )

    fit(
        model=model,
        train_dataloader=dataloaders["train"],
        valid_dataloader=dataloaders["valid"],
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
