import os
from torchmetrics.classification import Accuracy, Precision, Recall, F1Score
import matplotlib.pyplot as plt 
import numpy as np

def get_metrics(average, num_classes, device):
    return {
            "accuracy": Accuracy(task="multiclass", num_classes=num_classes, average=average).to(device),
            "precision": Precision(task="multiclass", num_classes=num_classes, average=average).to(device),
            "recall": Recall(task="multiclass", num_classes=num_classes, average=average).to(device),
            "f1": F1Score(task="multiclass", num_classes=num_classes, average=average).to(device),
        }

def print_metrics(metrics_micro, metrics_per_class, tag):
    for name, metric in metrics_micro.items():
        print(f"{tag} {name.upper()} (GLOBAL): {metric.compute().item():.6f}")

    recall_values_global = metrics_micro["recall"].compute()
    balanced_acc = recall_values_global.mean().item()
    print(f"{tag} BALANCED_ACCURACY (GLOBAL): {balanced_acc:.6f}")
    
    for name, metric in metrics_per_class.items():
        values = metric.compute().tolist()
        for i, v in enumerate(values):
            print(f"{tag} {name.upper()} [Class {i}]: {v:.6f}")


def print_metrics_global(metrics_micro, tag):
    for name, metric in metrics_micro.items():
        print(f"{tag} {name.upper()} (GLOBAL): {metric.compute().item():.6f}")

    recall_values_global = metrics_micro["recall"].compute()
    balanced_acc = recall_values_global.mean().item()
    print(f"{tag} BALANCED_ACCURACY (GLOBAL): {balanced_acc:.6f}")



def plot_metrics(checkpoint, precision, recall, pr_auc_per_class, epoch):
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

def plot_metrics_validation(checkpoint, precision, recall, pr_auc_per_class):
    if checkpoint is not None:
        plt.figure(figsize=(10, 8))
        for i, (p, r, auc) in enumerate(zip(precision, recall, pr_auc_per_class)):
            p = p.cpu().numpy()
            r = r.cpu().numpy()
            plt.plot(r, p, label=f'Class {i} (PR AUC = {auc:.4f})')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title(f'Precision-Recall Curves (Test Epoch)')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        pr_path = os.path.join(os.path.dirname(checkpoint), f'pr_curve_test.png')
        plt.savefig(pr_path)
        plt.close()


def plot_losses(checkpoint, train_losses, valid_losses):
    if checkpoint is not None:
        plt.figure(figsize=(10, 6))
        epochs = range(1, len(train_losses) + 1)
        
        plt.plot(epochs, train_losses, label='Train', color='blue', linestyle='-', marker='o', markersize=5)
        plt.plot(epochs, valid_losses, label='Validation', color='purple', linestyle='-', marker='s', markersize=5)
        
        plt.xlabel('Epochs')
        plt.ylabel('Cross-entropy loss')
        plt.title('Loss evolution over epochs')
        
        plt.legend()
        plt.grid(False)
        plt.xticks(ticks=np.arange(1, len(train_losses) + 1, step=1))
        plt.xlim(1, len(train_losses))
        
        plt.tight_layout()
        loss_path = os.path.join(os.path.dirname(checkpoint), 'loss_curve.png')
        plt.savefig(loss_path)
        plt.close()
