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
from sklearn.utils.multiclass import unique_labels

import warnings
from sklearn.exceptions import UndefinedMetricWarning, ConvergenceWarning

import GPyOpt

from src.metrics import get_metrics, print_metrics, plot_losses
from src.aux import EarlyStopper, longtiming
from src.data import Reader, Scaler, FullDataset, fit_scaler_manual
from src.resnext import ResNeXt

warnings.filterwarnings("ignore", category=UserWarning, module="sklearn.metrics._classification")

logger = logging.getLogger(__name__)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DTYPE = torch.bfloat16

def fit(
    model: ResNeXt,
    *,
    train_dataloader,
    valid_dataloader,
    test_dataloader=None,
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

    return valid_losses[-1]


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
        shutil.rmtree(rundir)
    os.makedirs(rundir)

    logging.basicConfig(
        filename=os.path.join(rundir, "runinfo.log"), level=logging.INFO
    )

    shutil.copy(
        os.path.join(rootdir, "config.yaml"), os.path.join(rundir, "config.yaml")
    )

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    DTYPE = torch.bfloat16

    datadir = config["paths"]["datadir"]
    scaler_path = os.path.join(rundir, config["paths"]["scaler"])

    reader = Reader(datadir)
    scaler = Scaler(norm=config["scaler"], dim=-1)

    fit_scaler_manual(reader(key="train"), scaler, batch_size=config['loaders']['batch_size'], max_workers=config['loaders']['num_workers'])
    scaler.save(scaler_path)

    train_dataset = FullDataset(reader(key="train"), scaler=scaler)
    valid_dataset = FullDataset(reader(key="valid"), scaler=scaler)
    test_dataset = FullDataset(reader(key="test"), scaler=scaler)

    valid_cardinalities = [1, 2, 4, 8, 16]

    space = [
        {"name": "cardinality", "type": "discrete", "domain": tuple(valid_cardinalities)},
        {"name": "num_blocks", "type": "discrete", "domain": tuple(range(1, 7))},
    ]

    def objective(x):
        params = x[0]
        cardinality = int(params[0])
        num_blocks_param = int(params[1])

        if cardinality > 8:
            print(f"Reduciendo cardinality de {cardinality} a 8 para evitar OOM")
            cardinality = 8
        if num_blocks_param > 2:
            print(f"Reduciendo num_blocks de {num_blocks_param} a 2 para evitar OOM")
            num_blocks_param = 2

        num_blocks = [num_blocks_param] * cardinality

        print(f"Evaluando: cardinality={cardinality}, num_blocks={num_blocks}")

        model_kwargs = dict(config["resnext"])
        model_kwargs.update({
            "cardinality": cardinality,
            "num_blocks": num_blocks,
            "in_channels": 1,
            "out_features": 4,
            "activation": nn.SiLU,
        })

        try:
            torch.cuda.empty_cache()

            model = ResNeXt(**model_kwargs, device=DEVICE, dtype=DTYPE)
            model = model.to(DEVICE).to(dtype=DTYPE)
        except RuntimeError as e:
            if "out of memory" in str(e):
                print("OOM detectado durante la creación del modelo, saltando esta configuración")
                return np.array([[np.inf]])
            else:
                raise e

        optimizer = optim.AdamW(model.parameters(), **config["optimizer"])

        base_batch_size = config['loaders']['batch_size']
        if cardinality * num_blocks_param > 8:
            adjusted_batch_size = max(1, base_batch_size // 4)
            print(f"Reduciendo batch size de {base_batch_size} a {adjusted_batch_size} para evitar OOM")
            batch_size = adjusted_batch_size
        else:
            batch_size = base_batch_size

        train_dataloader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True,
            num_workers=config['loaders']['num_workers'], pin_memory=True
        )
        valid_dataloader = DataLoader(
            valid_dataset, batch_size=batch_size, shuffle=False,
            num_workers=config['loaders']['num_workers'], pin_memory=True
        )

        num_epochs = config['fit']['num_epochs']
        patience = config['fit']['patience']
        min_delta = config['fit']['min_delta']
        update_every = config['fit']['update_every']
        batch_split = config['fit']['batch_split']

        valid_loss = fit(
            model=model,
            train_dataloader=train_dataloader,
            valid_dataloader=valid_dataloader,
            test_dataloader=None,
            optimizer=optimizer,
            num_epochs=num_epochs,
            patience=patience,
            min_delta=min_delta,
            update_every=update_every,
            batch_split=batch_split,
            device=DEVICE,
            dtype=DTYPE,
            shuffle=True,
        )

        del model, optimizer, train_dataloader, valid_dataloader
        torch.cuda.empty_cache()

        return np.array([[valid_loss]])


    optimizer = GPyOpt.methods.BayesianOptimization(
        f=objective,
        domain=space,
        acquisition_type='EI',
        exact_feval=True,
        maximize=False,
        verbosity=True,
    )

    max_iter = 30
    optimizer.run_optimization(max_iter=max_iter)

    best_x = optimizer.X[np.argmin(optimizer.Y)]
    best_cardinality = int(best_x[0])
    best_num_blocks = int(best_x[1])

    print(f"\nMejores parámetros encontrados:")
    print(f"cardinality: {best_cardinality}")
    print(f"num_blocks: {best_num_blocks}")
    print(f"Mejor valid_loss: {optimizer.Y.min()}")
