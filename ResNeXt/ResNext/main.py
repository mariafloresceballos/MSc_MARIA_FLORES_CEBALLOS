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

import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, balanced_accuracy_score
from sklearn.utils.multiclass import unique_labels

from src.metrics import get_metrics, print_metrics, plot_losses
from src.aux import EarlyStopper, longtiming
from src.data import Reader, Scaler, FullDataset, fit_scaler_manual
from src.resnext import ResNeXt

import warnings
from sklearn.exceptions import UndefinedMetricWarning, ConvergenceWarning

warnings.filterwarnings("ignore", category=UserWarning, module="sklearn.metrics._classification")

logger = logging.getLogger(__name__)

def fit(
    model: ResNeXt,
    *,
    train_dataloader,
    valid_dataloader,
    test_dataloader,
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
    factory_kwargs = {"device": device, "dtype": dtype}

    start = time.time()
    now = datetime.now()
    dt = now.strftime("%d-%m-%Y %H:%M:%S")
    print(f"\nSTARTING @ {dt}\n")

    stopper = EarlyStopper(patience=patience, min_delta=min_delta)
    num_classes = 4

    train_losses = []
    valid_losses = []

    train_metrics_micro = get_metrics("micro", num_classes, device)
    valid_metrics_micro = get_metrics("micro", num_classes, device)
    train_metrics_per_class = get_metrics("none", num_classes, device)
    valid_metrics_per_class = get_metrics("none", num_classes, device)

    for epoch in range(1, num_epochs + 1):
        train_loss = 0.0
        valid_loss = 0.0

        for m in [train_metrics_micro, valid_metrics_micro, train_metrics_per_class, valid_metrics_per_class]:
            for metric in m.values():
                metric.reset()

        model.train()
        optimizer.zero_grad()

        train_all_preds = []
        train_all_labels = []

        for i, (signals, labels, _) in enumerate(tqdm(train_dataloader, desc=f"Training Epoch {epoch}"), start=1):
            signals = signals.transpose(-1, -2)
            labels = labels.to(device=device, dtype=torch.int64)

            if shuffle:
                p = torch.randperm(signals.shape[0])
                signals = signals[p]
                labels = labels[p]

            if batch_split > 1:
                signals = signals.reshape(batch_split, -1, *signals.shape[1:])
                labels = labels.reshape(batch_split, -1, *labels.shape[1:])
                for s, l in zip(signals, labels):
                    s = s.to(**factory_kwargs)

                    output = model(s, raw=True)
                    loss = F.cross_entropy(output, l)
                    loss.backward()
                    train_loss += loss.item()

                    preds = output.argmax(dim=1).cpu()
                    train_all_preds.append(preds)
                    train_all_labels.append(l.cpu())
            else:
                signals = signals.to(**factory_kwargs)

                output = model(signals, raw=True)
                loss = F.cross_entropy(output, labels)
                loss.backward()
                train_loss += loss.item()

                preds = output.argmax(dim=1).cpu()
                train_all_preds.append(preds)
                train_all_labels.append(labels.cpu())

            if i % update_every == 0 or i == len(train_dataloader):
                optimizer.step()
                optimizer.zero_grad()

        train_loss /= batch_split * len(train_dataloader)

        model.eval()

        valid_all_preds = []
        valid_all_labels = []

        with torch.no_grad():
            for signals, labels, _ in tqdm(valid_dataloader, desc=f"Validation Epoch {epoch}"):
                signals = signals.transpose(-1, -2)
                labels = labels.to(device=device, dtype=torch.int64)

                if batch_split > 1:
                    signals = signals.reshape(batch_split, -1, *signals.shape[1:])
                    labels = labels.reshape(batch_split, -1, *labels.shape[1:])
                    for s, l in zip(signals, labels):
                        s = s.to(**factory_kwargs)
                        output = model(s, raw=True)
                        loss = F.cross_entropy(output, l)
                        valid_loss += loss.item()

                        preds = output.argmax(dim=1).cpu()
                        valid_all_preds.append(preds)
                        valid_all_labels.append(l.cpu())
                else:
                    signals = signals.to(**factory_kwargs)
                    output = model(signals, raw=True)
                    loss = F.cross_entropy(output, labels)
                    valid_loss += loss.item()

                    preds = output.argmax(dim=1).cpu()
                    valid_all_preds.append(preds)
                    valid_all_labels.append(labels.cpu())

        valid_loss /= batch_split * len(valid_dataloader)


        train_losses.append(train_loss)
        valid_losses.append(valid_loss)

        print(f"\nEPOCH #{epoch} of {num_epochs} | Time: {longtiming(time.time() - start)}")
        print(f"TRAIN LOSS: {train_loss:.6f} | VALID LOSS: {valid_loss:.6f}")

        stop = stopper(valid_loss, model.state_dict())
        if checkpoint is not None:
            stopper.save(checkpoint)
        if stop:
            print("\nEARLY STOPPING\n")
            break

    plot_losses(checkpoint, train_losses, valid_losses)

    print(f"\nTRAINING COMPLETE\nTOTAL TIME: {longtiming(time.time() - start)}")

    print("\n" + "=" * 60)
    print("EVALUATING FINAL MODEL ON TEST SET")
    print("=" * 60)

    test_metrics_micro = get_metrics("micro", num_classes, device)
    test_metrics_per_class = get_metrics("none", num_classes, device)

    model.eval()

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for signals, labels, _ in tqdm(test_dataloader, desc="Test Evaluation"):
            signals = signals.transpose(-1, -2)
            labels = labels.to(device=device, dtype=torch.int64)

            if batch_split > 1:
                signals = signals.reshape(batch_split, -1, *signals.shape[1:])
                labels = labels.reshape(batch_split, -1, *labels.shape[1:])
                for s, l in zip(signals, labels):
                    s = s.to(**factory_kwargs)
                    output = model(s, raw=True)
                    for metric_set in [test_metrics_micro, test_metrics_per_class]:
                        for metric in metric_set.values():
                            metric.update(output, l)
                    preds = output.argmax(dim=1).cpu()
                    all_preds.append(preds)
                    all_labels.append(l.cpu())
            else:
                signals = signals.to(**factory_kwargs)
                output = model(signals, raw=True)
                for metric_set in [test_metrics_micro, test_metrics_per_class]:
                    for metric in metric_set.values():
                        metric.update(output, labels)
                preds = output.argmax(dim=1).cpu()
                all_preds.append(preds)
                all_labels.append(labels.cpu())

    for metric_set in [test_metrics_micro, test_metrics_per_class]:
        for metric in metric_set.values():
            metric.compute()  


    all_preds = torch.cat(all_preds).numpy()
    all_labels = torch.cat(all_labels).numpy()
    test_labels_union = unique_labels(np.concatenate([all_labels, all_preds]))
    acc = accuracy_score(all_labels, all_preds)
    balanced_acc = balanced_accuracy_score(all_labels, all_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels, all_preds, labels=test_labels_union, average='macro', zero_division=0
    )

    print("\nTEST METRICS:")
    print(f"Accuracy: {acc:.6f}")
    print(f"Balanced Accuracy: {balanced_acc:.6f}")
    print(f"Precision: {precision:.6f}")
    print(f"Recall: {recall:.6f}")
    print(f"F1 score: {f1:.6f}")

    print_metrics(test_metrics_micro, test_metrics_per_class, "TEST")


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

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    DTYPE = torch.bfloat16
    factory_kwargs = {"device": DEVICE, "dtype": DTYPE}

    datadir = config["paths"]["datadir"]
    scaler_path = os.path.join(rundir, config["paths"]["scaler"])
    model_path = os.path.join(rundir, config["paths"]["model"])
    ckpt_path = (
        os.path.splitext(model_path)[0] + "__CKPT" + os.path.splitext(model_path)[1]
    )
    model_kwargs = config["resnext"]

    model = ResNeXt(
        in_channels=1,
        out_features=4, 
        activation=nn.SiLU,
        **model_kwargs,
        **factory_kwargs,
    )
    optimizer = optim.AdamW(model.parameters(), **config["optimizer"])

    print(
        f"\nResNext blocks-{(model_kwargs['num_blocks'])}  -  MODEL DESCRIPTION:\n\n",
        model,
        "\n\n",
    )

    model = model.to(DEVICE).to(dtype=DTYPE)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"\nTotal number of model parameters: {total_params}\n")

    reader = Reader(datadir)
    scaler = Scaler(norm=config["scaler"], dim=-1)

    fit_scaler_manual(reader(key="train"), scaler, batch_size=32, max_workers=8)
    scaler.save(scaler_path)

    train_dataset = FullDataset(reader(key="train"), scaler=scaler)
    valid_dataset = FullDataset(reader(key="valid"), scaler=scaler)
    test_dataset = FullDataset(reader(key="test"), scaler=scaler)

    train_dataloader = DataLoader(
        train_dataset, batch_size=config['loaders']['batch_size'], shuffle=True, num_workers=config['loaders']['num_workers'], pin_memory=True
    )

    valid_dataloader = DataLoader(
        valid_dataset, batch_size=config['loaders']['batch_size'], shuffle=True, num_workers=config['loaders']['num_workers'], pin_memory=True
    )

    test_dataloader = DataLoader(
        test_dataset, batch_size=config['loaders']['batch_size'], shuffle=True, num_workers=config['loaders']['num_workers'], pin_memory=True
    )

    fit(
        model=model,
        train_dataloader=train_dataloader,
        valid_dataloader=valid_dataloader,
        test_dataloader=test_dataloader,
        optimizer=optimizer,
        checkpoint=ckpt_path,
        shuffle=True,
        **config["fit"],
        **factory_kwargs,
    )

    torch.save(model, model_path)
    print(f"\n\nMODEL SAVED SUCCESSFULLY TO:\n{os.path.abspath(model_path)}\n")
    logger.info(f"\n\nMODEL SAVED SUCCESSFULLY TO:\n{os.path.abspath(model_path)}\n")
