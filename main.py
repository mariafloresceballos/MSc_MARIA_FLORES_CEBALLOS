import csv
from datetime import datetime
import logging
import os
import shutil
import time
from tqdm import tqdm
import yaml
from collections import Counter

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torchmetrics.classification import Accuracy

from src.aux import EarlyStopper, longtiming
from src.data import LazyDataset, Reader, Scaler
from src.convresnet import ConvResNet
from gradcam import GradCAM1D

logger = logging.getLogger(__name__)

from torcheval.metrics.functional import multiclass_precision_recall_curve
import numpy as np
from torchmetrics.classification import Accuracy, Precision, Recall, F1Score
import matplotlib.pyplot as plt  # Importar matplotlib

def fit(
    model: ConvResNet,
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

    def get_metrics(average):
        return {
            "accuracy": Accuracy(task="multiclass", num_classes=num_classes, average=average).to(device),
            "precision": Precision(task="multiclass", num_classes=num_classes, average=average).to(device),
            "recall": Recall(task="multiclass", num_classes=num_classes, average=average).to(device),
            "f1": F1Score(task="multiclass", num_classes=num_classes, average=average).to(device),
        }

    train_metrics_micro = get_metrics("micro")
    valid_metrics_micro = get_metrics("micro")
    train_metrics_per_class = get_metrics("none")
    valid_metrics_per_class = get_metrics("none")

    for epoch in range(1, num_epochs + 1):
        epoch_start = time.time()
        train_loss = 0.0
        valid_loss = 0.0

        for m in [train_metrics_micro, valid_metrics_micro, train_metrics_per_class, valid_metrics_per_class]:
            for metric in m.values():
                metric.reset()

        model.train()
        optimizer.zero_grad()
        for i, (signals, labels, _) in enumerate(tqdm(train_dataloader, desc=f"Training Epoch {epoch}"), start=1):
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
                    l = l.to(device=device, dtype=torch.int64)

                    output = model(s, raw=True)
                    loss = F.cross_entropy(output, l)
                    loss.backward()
                    train_loss += loss.item()

                    for metric_set in [train_metrics_micro, train_metrics_per_class]:
                        for metric in metric_set.values():
                            metric.update(output, l)
            else:
                signals = signals.to(**factory_kwargs)
                labels = labels.to(device=device, dtype=torch.int64)

                output = model(signals, raw=True)
                loss = F.cross_entropy(output, labels)
                loss.backward()
                train_loss += loss.item()

                for metric_set in [train_metrics_micro, train_metrics_per_class]:
                    for metric in metric_set.values():
                        metric.update(output, labels)

            if i % update_every == 0 or i == len(train_dataloader):
                optimizer.step()
                optimizer.zero_grad()

        train_loss /= batch_split * len(train_dataloader)

        model.eval()
        all_outputs = []
        all_labels = []

        with torch.no_grad():
            for signals, labels, _ in tqdm(valid_dataloader, desc=f"Validation Epoch {epoch}"):
                signals = signals.squeeze(0).transpose(-1, -2)
                labels = labels.squeeze(0)

                if batch_split > 1:
                    signals = signals.reshape(batch_split, -1, *signals.shape[1:])
                    labels = labels.reshape(batch_split, -1, *labels.shape[1:])
                    for s, l in zip(signals, labels):
                        s = s.to(**factory_kwargs)
                        l = l.to(device=device, dtype=torch.int64)
                        output = model(s, raw=True)
                        loss = F.cross_entropy(output, l)
                        valid_loss += loss.item()

                        for metric_set in [valid_metrics_micro, valid_metrics_per_class]:
                            for metric in metric_set.values():
                                metric.update(output, l)

                        all_outputs.append(output)
                        all_labels.append(l)
                else:
                    signals = signals.to(**factory_kwargs)
                    labels = labels.to(device=device, dtype=torch.int64)

                    output = model(signals, raw=True)
                    loss = F.cross_entropy(output, labels)
                    valid_loss += loss.item()

                    for metric_set in [valid_metrics_micro, valid_metrics_per_class]:
                        for metric in metric_set.values():
                            metric.update(output, labels)

                    all_outputs.append(output)
                    all_labels.append(labels)

        valid_loss /= batch_split * len(valid_dataloader)

        train_losses.append(train_loss)
        valid_losses.append(valid_loss)

        all_outputs_tensor = torch.cat(all_outputs, dim=0)
        all_labels_tensor = torch.cat(all_labels, dim=0)

        precision, recall, _ = multiclass_precision_recall_curve(
            input=all_outputs_tensor,
            target=all_labels_tensor,
            num_classes=num_classes
        )

        pr_auc_per_class = []
        for p, r in zip(precision, recall):
            p = p.cpu().numpy()
            r = r.cpu().numpy()
            sorted_indices = np.argsort(r)
            r_sorted = r[sorted_indices]
            p_sorted = p[sorted_indices]
            auc = np.trapz(p_sorted, r_sorted)
            pr_auc_per_class.append(auc)

        pr_auc_mean = np.mean(pr_auc_per_class)

        print(f"\nEPOCH #{epoch} of {num_epochs} | Time: {longtiming(time.time() - start)}")
        print(f"TRAIN LOSS: {train_loss:.6f} | VALID LOSS: {valid_loss:.6f}")

        def print_metrics(metrics_micro, metrics_per_class, tag):
            for name, metric in metrics_micro.items():
                print(f"{tag} {name.upper()} (GLOBAL): {metric.compute().item():.6f}")
            for name, metric in metrics_per_class.items():
                values = metric.compute().tolist()
                for i, v in enumerate(values):
                    print(f"{tag} {name.upper()} [Class {i}]: {v:.6f}")

        print_metrics(train_metrics_micro, train_metrics_per_class, "TRAIN")
        print_metrics(valid_metrics_micro, valid_metrics_per_class, "VALID")

        print("VALID PR AUC (per class):")
        for i, auc in enumerate(pr_auc_per_class):
            print(f"PR AUC [Class {i}]: {auc:.6f}")
        print(f"PR AUC (Mean): {pr_auc_mean:.6f}\n")

        if checkpoint is not None:
            plt.figure(figsize=(10, 8))
            for i, (p, r, auc) in enumerate(zip(precision, recall, pr_auc_per_class)):
                p = p.cpu().numpy()
                r = r.cpu().numpy()
                plt.plot(r, p, label=f'Class {i} (PR AUC = {auc:.4f})')
            plt.xlabel('Recall')
            plt.ylabel('Precision')
            plt.title(f'Precision-Recall Curves (Validation Epoch {epoch})')
            plt.legend()
            plt.grid(True)
            plt.tight_layout()
            pr_path = os.path.join(os.path.dirname(checkpoint), f'pr_curve_epoch_{epoch}.png')
            plt.savefig(pr_path)
            plt.close()

        stop = stopper(valid_loss, model.state_dict())
        if checkpoint is not None:
            stopper.save(checkpoint)
        if stop:
            print("\nEARLY STOPPING\n")
            break

    # Plot loss evolution
    if checkpoint is not None:
        plt.figure(figsize=(10, 6))
        plt.plot(range(1, len(train_losses)+1), train_losses, label='Train', marker='o')
        plt.plot(range(1, len(valid_losses)+1), valid_losses, label='Validation', marker='s')
        plt.xlabel('Epochs')
        plt.ylabel('Cross-Entropy Loss')
        plt.title('Loss Evolution')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        loss_path = os.path.join(os.path.dirname(checkpoint), 'loss_curve.png')
        plt.savefig(loss_path)
        plt.close()

    print(f"\nTRAINING COMPLETE\nTOTAL TIME: {longtiming(time.time() - start)}")

    # Final evaluation on TEST set
    print("\n" + "=" * 60)
    print("EVALUATING FINAL MODEL ON TEST SET")
    print("=" * 60)

    test_metrics_micro = get_metrics("micro")
    test_metrics_per_class = get_metrics("none")
    all_outputs = []
    all_labels = []

    model.eval()
    with torch.no_grad():
        for signals, labels, _ in tqdm(test_dataloader, desc="Test Evaluation"):
            signals = signals.squeeze(0).transpose(-1, -2)
            labels = labels.squeeze(0)

            if batch_split > 1:
                signals = signals.reshape(batch_split, -1, *signals.shape[1:])
                labels = labels.reshape(batch_split, -1, *labels.shape[1:])
                for s, l in zip(signals, labels):
                    s = s.to(**factory_kwargs)
                    l = l.to(device=device, dtype=torch.int64)
                    output = model(s, raw=True)
                    for metric_set in [test_metrics_micro, test_metrics_per_class]:
                        for metric in metric_set.values():
                            metric.update(output, l)
                    all_outputs.append(output)
                    all_labels.append(l)
            else:
                signals = signals.to(**factory_kwargs)
                labels = labels.to(device=device, dtype=torch.int64)
                output = model(signals, raw=True)
                for metric_set in [test_metrics_micro, test_metrics_per_class]:
                    for metric in metric_set.values():
                        metric.update(output, labels)
                all_outputs.append(output)
                all_labels.append(labels)

    print_metrics(test_metrics_micro, test_metrics_per_class, "TEST")

    all_outputs_tensor = torch.cat(all_outputs, dim=0)
    all_labels_tensor = torch.cat(all_labels, dim=0)

    precision, recall, _ = multiclass_precision_recall_curve(
        input=all_outputs_tensor,
        target=all_labels_tensor,
        num_classes=num_classes
    )

    pr_auc_per_class = []
    for p, r in zip(precision, recall):
        p = p.cpu().numpy()
        r = r.cpu().numpy()
        sorted_indices = np.argsort(r)
        r_sorted = r[sorted_indices]
        p_sorted = p[sorted_indices]
        auc = np.trapz(p_sorted, r_sorted)
        pr_auc_per_class.append(auc)

    pr_auc_mean = np.mean(pr_auc_per_class)

    print("\nTEST PR AUC (per class):")
    for i, auc in enumerate(pr_auc_per_class):
        print(f"PR AUC [Class {i}]: {auc:.6f}")
    print(f"PR AUC (Mean): {pr_auc_mean:.6f}")

    if checkpoint is not None:
        plt.figure(figsize=(10, 8))
        for i, (p, r, auc) in enumerate(zip(precision, recall, pr_auc_per_class)):
            p = p.cpu().numpy()
            r = r.cpu().numpy()
            plt.plot(r, p, label=f'Class {i} (PR AUC = {auc:.4f})')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curve (Test)')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        pr_path = os.path.join(os.path.dirname(checkpoint), 'pr_curve_test.png')
        plt.savefig(pr_path)
        plt.close()

        # print("\nApplying Grad-CAM 1D to a test sample...")

        # # Obtener una muestra del test set (un bloque con múltiples señales)
        # sample_signal, sample_label, _ = next(iter(test_dataloader))

        # # Extraer solo una señal y etiqueta del batch
        # sample_signal = sample_signal.squeeze(0).transpose(-1, -2)  # (batch_size, length, channels)
        # sample_signal = sample_signal[0].unsqueeze(0).to(device=device, dtype=torch.float32)  # (1, C, L)
        # sample_label = sample_label[0].item()

        # # Inicializar Grad-CAM
        # grad_cam = GradCAM1D(model, target_layer=model.conv_stack[-1])
        # cam_output, probs = grad_cam.generate(sample_signal, target_class=sample_label)
        # grad_cam.remove_hooks()

        # # Visualización
        # signal_np = sample_signal.squeeze().cpu().numpy()
        # probs_str = ", ".join([f"Class {i}: {p:.2f}" for i, p in enumerate(probs[0])])

        # plt.figure(figsize=(12, 4))
        # plt.plot(signal_np, label="ECG Signal")
        # plt.plot(cam_output * np.max(signal_np), label="Grad-CAM", color="red", alpha=0.6)
        # plt.title(f"Grad-CAM on Test Sample | True class: {sample_label}\n{probs_str}")
        # plt.xlabel("Time (samples)")
        # plt.ylabel("Amplitude")
        # plt.legend()
        # plt.grid(True)
        # plt.tight_layout()

        # if checkpoint is not None:
        #     gradcam_path = os.path.join(os.path.dirname(checkpoint), "gradcam_test_sample.png")
        #     plt.savefig(gradcam_path)
        #     print(f"Grad-CAM image saved to: {gradcam_path}")
        # else:
        #     plt.show()

        # plt.close()



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

    # Imprimir el número total de parámetros del modelo
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\nNúmero total de parámetros del modelo: {total_params}\n")

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

    test_dataset = LazyDataset(reader(key="test"), scaler=scaler)
    test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False, **config["loaders"])

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


    # Save full model in .pt format (for Netron)
    torch.save(model, model_path)
    print(f"\n\nMODEL SAVED SUCCESSFULLY TO:\n{os.path.abspath(model_path)}\n")
    logger.info(f"\n\nMODEL SAVED SUCCESSFULLY TO:\n{os.path.abspath(model_path)}\n")

    # Optional: also export to ONNX for full Netron compatibility
    try:
        # Define dummy input with appropriate shape
        input_dummy = torch.randn(1, 1, config["data"]["length"]).to(DEVICE).to(dtype=torch.float32)

        onnx_path = os.path.splitext(model_path)[0] + ".onnx"

        torch.onnx.export(
            model,
            input_dummy,
            onnx_path,
            input_names=["input"],
            output_names=["output"],
            dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
            export_params=True,
            opset_version=17
        )
        print(f"ONNX MODEL SAVED TO:\n{onnx_path}")
        logger.info(f"ONNX MODEL SAVED TO:\n{onnx_path}")
    except Exception as e:
        print(f"ERROR DURING ONNX EXPORT: {e}")
        logger.error(f"ERROR DURING ONNX EXPORT: {e}")

    # Cleanup checkpoint if saved
    if os.path.exists(ckpt_path):
        os.remove(ckpt_path)
