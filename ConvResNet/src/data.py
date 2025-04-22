from copy import deepcopy
from glob import glob
import logging
import os
import shutil
import yaml

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import wfdb

logger = logging.getLogger(__name__)


def label(rhythm):
    """
    Designed such that, for our problem, the largest labels are the most interesting ones
        -> reduction with `max` function along signal length.

    Inputs:
        * rhythm: list of annotations.

    Outputs:
        * Annotations for each point in numerical form.
    """
    labeled_rhythm = []
    active_class = 1  # by default, start with "Unknown"

    for r in rhythm:
        if r == ")":
            active_class = 1  # Unknown
        elif r == "(N":
            active_class = 0  # Normal sinus rhythm
        elif r == "(AFIB":
            active_class = 3  # Atrial fibrillation !!!
        elif r == "(AFL":
            active_class = 2  # Atrial flutter
        # else:  # r is None
        #     pass
        labeled_rhythm.append(active_class)

    return np.array(labeled_rhythm, dtype=np.uint8)


def construct_arrays(file):
    """
    Obtains arrays from file (split in header, attributes and data files).

    Inputs:
        * file: path to file to be read.

    Outputs:
        * signal: raw data points.
        * rhythm: rhythm annotations for each point.
        * bpm: locally estimated beats per minute (BPM) at each point.
    """
    record = wfdb.rdrecord(file)
    signal = record.p_signal
    fs = record.fs

    annotations = wfdb.rdann(file, "atr")
    notes = annotations.aux_note
    samples = annotations.sample

    rhythm = np.array([None] * len(signal))
    for n, s in zip(notes, samples):
        rhythm[s] = n

    rhythm = label(rhythm)

    samples = np.sort(np.unique(samples))
    _xbpm = samples[1:] - samples[:-1]
    _bpm = 60 * fs / _xbpm

    xbpm = np.linspace(start=0, stop=len(signal) - 1, num=len(signal))
    bpm = np.interp(xbpm, _xbpm, _bpm).astype(np.float32)

    return signal, rhythm, bpm


def sample(signal, rhythm, bpm, length=1000, N=None, oversampling=10.0, avg_bpm=False):
    """
    Extract sequences of fixed length from arrays.

    Inputs:
        * signal
        * rhythm
        * bpm
        * N: number of sequences to extract. If `None`, calculated using signal length.
        * length: length of output sequences.
        * oversampling: oversampling factor for the sequence. The number of sequences,
            N, is calculated as: (signal length / length) * oversampling.
        * avg_bpm: whether to average BPM over sequence length or not.

    Outputs:
        * Stack of signals with shape (N, length, number of channels).
        * Stack of rhythm annotations with shape (N,).
        * Stack of BPM annotations with shape (N,) or (N, length) (see `avg_bpm`).
    """
    if N is None:
        N = int(len(signal) * oversampling / length)

    startlast = len(signal) - length
    starts = np.random.randint(low=0, high=startlast + 1, size=N)

    axis = 0

    signals = np.array(
        [signal[(slice(None),) * axis + (slice(s, s + length),)] for s in starts],
        dtype=np.float16,
    )
    rhythms = np.array(
        [rhythm[(slice(None),) * axis + (slice(s, s + length),)] for s in starts],
        dtype=np.uint8,
    ).max(axis=-1)
    bpms = np.array(
        [bpm[(slice(None),) * axis + (slice(s, s + length),)] for s in starts],
        dtype=np.float16,
    )
    if avg_bpm:
        bpms = bpms.mean(axis=-1)

    return signals, rhythms, bpms


class Reader:
    """
    Class for reading data directories.
    """

    def __init__(self, folder):
        """
        Initializer for Reader class.

        Inputs:
            * folder: base directory to be read. The expected directory structure is:

                .
                └── folder/
                    ├── test/
                    │   ├── x01.npz
                    │   ├── x02.npz
                    │   └── ...
                    ├── train/
                    │   ├── x03.npz
                    │   ├── x04.npz
                    │   └── ...
                    └── valid/
                        └── ...

                    Each .npz must contain three arrays:
                    'signals', 'labels' (rhythm annotations) and 'bpms' (BPM annotations).
        """
        self.keys = ["train", "valid", "test"]
        self.folder = folder

        self.files = (
            {k: sorted(glob(os.path.join(self.folder, k, "*.npz"))) for k in self.keys}
            if isinstance(self.keys, list)
            else sorted(glob(os.path.join(self.folder, self.keys, "*.npz")))
        )

    @staticmethod
    def read(f):
        """
        Read file to dictionary of arrays with same labels as npz file.

        Inputs:
            * f: path of file to be read.

        Outputs:
            * Dictionary containing data.
        """
        data = dict(np.load(f))
        signals = torch.tensor(data["signals"])
        labels = torch.tensor(data["labels"])
        bpms = torch.tensor(data["bpms"])
        return signals, labels, bpms

    def __call__(self, key=None):
        """
        Get all files reader's folder for given key (test/train/valid).

        Inputs:
            * key: key to retrieve files.
        """
        if key is not None:
            data = self.files[key]
        else:
            data = {k: self.files[k] for k in self.keys}

        return data


class LazyDataset(Dataset):
    """
    Subclass of PyTorch's `Dataset` that loads data lazily to memory.
    """

    def __init__(self, files, scaler=None):
        """
        Initializer for `LazyDataset`.

        Inputs:
            * files: list of file (paths) to load in the dataset.
            * scaler: whether to use a `Scaler` or not.
        """
        self.files = files
        self.scaler = scaler if scaler is not None else nn.Identity()
        self.num_files = len(files)

    def __len__(self):
        """
        Length of dataset.
        """
        return self.num_files

    def __getitem__(self, idx):
        """
        Read file located at a given index of the dataset.

        Inputs:
            * idx: index of file to be read.

        Outputs:
            * Scaled (if appropriate) signal.
            * Labels, i.e. rhythm annotations.
            * BPM annotations.
        """
        f = self.files[idx]
        signals, labels, bpms = Reader.read(f=f)
        return self.scaler(signals), labels, bpms


class Scaler(nn.Module):
    """
    Scaler class (instance of `torch.nn.Module`) to implement standardization or normalization.
    """

    def __init__(self, norm="std", *, dim=-1):
        """
        Initializer for `Scaler`.

        Inputs:
            * norm: whether to use standardization ('std') or min-max normalization ('minmax').
            * dim: feature dimension. If data has shape (...(N), length, channels), then `dim=-1`.
        """
        super().__init__()

        self.norm = norm
        assert self.norm in ["std", "minmax"]

        self.register_buffer("dim", torch.tensor(dim))

        # minmax
        self.register_buffer("min", None)
        self.register_buffer("max", None)
        # std
        self.register_buffer("mean", None)
        self.register_buffer("std", None)
        self.cumlen = torch.tensor(0, dtype=torch.int64)

    def reset(self):
        """
        Reset scaler to defaults (untrained).
        """
        # minmax
        self.register_buffer("min", None)
        self.register_buffer("max", None)
        # std
        self.register_buffer("mean", None)
        self.register_buffer("std", None)
        self.cumlen = torch.tensor(0, dtype=torch.int64)

    @torch.no_grad()
    def _aggregate(self, xmin=None, xmax=None, xmean=None, xstd=None, xlen=None):
        """
        Aggregate batch statistics for `lazy_fit`.

        Inputs:
            * xmin: minimum of signal (minmax only).
            * xmax: maximum of signal (minmax only).
            * xmean: mean of signal (std only).
            * xstd: standard deviation of signal (std only).
            * xlen: length of signal, necessary for accumulation of mean and std (std only).
        """
        if self.norm == "minmax":
            if self.min is not None and self.max is not None:
                self.min = torch.cat((self.min, xmin), dim=0).min(dim=0, keepdims=True)[
                    0
                ]
                self.max = torch.cat((self.max, xmax), dim=0).min(dim=0, keepdims=True)[
                    0
                ]
            else:
                self.min = xmin
                self.max = xmax

        elif self.norm == "std":
            if self.mean is not None and self.std is not None and self.cumlen > 0:
                self.std = (
                    (
                        (
                            (self.cumlen - 1) * self.std.to(torch.float32) ** 2
                            + (xlen - 1) * xstd.to(torch.float32) ** 2
                        )
                        / (self.cumlen + xlen - 1)
                    )
                    + (
                        (
                            self.cumlen
                            * xlen
                            * (self.mean - xmean).to(torch.float32) ** 2
                        )
                        / ((self.cumlen + xlen) * (self.cumlen + xlen - 1))
                    )
                ).sqrt()
                self.mean = (self.mean * self.cumlen + xmean * xlen) / (
                    self.cumlen + xlen
                )
                self.cumlen += xlen
            else:
                self.mean = xmean
                self.std = xstd
                self.cumlen += xlen

    @torch.no_grad()
    def _extract_tensor_stats(self, x):
        """
        Extract batch statistics for `lazy_fit`.

        Inputs:
            * x: tensor for which to extract statistics.

        Outputs:
            * Dictionary containing tensor statistics to be aggregated using `_aggregate`.
        """
        stats = {"xmin": None, "xmax": None, "xmean": None, "xstd": None, "xlen": None}
        x_ = x.transpose(-1, self.dim).reshape(-1, x.shape[self.dim])

        if self.norm == "minmax":
            stats["xmin"] = x_.min(dim=0, keepdim=True)[0]
            stats["xmax"] = x_.max(dim=0, keepdim=True)[0]

        elif self.norm == "std":
            stats["xmean"] = x_.mean(dim=0, keepdim=True)[0]
            stats["xstd"] = x_.std(dim=0, keepdim=True)[0]
            stats["xlen"] = x_.shape[0]

        return stats

    @torch.no_grad()
    def _batch_fit(self, x):
        """
        Single batch fit, to be used inside `lazy_fit`.

        Inputs:
            * x: tensor to fit.
        """
        stats = self._extract_tensor_stats(x)
        self._aggregate(**stats)

    @torch.no_grad()
    def lazy_fit(self, data, *, num_workers=0, prefetch_factor=None):
        """
        Fit scaler by batches. Useful when it is impossible to load all the dataset at once.

        Inputs:
            * data: list of path of files to fit to (used as input for `LazyDataset`).
            * num_workers: number of workers for dataloader (see PyTorch's `DataLoader`).
            * prefetch_factor: prefetch factor for dataloader (see PyTorch's `DataLoader`).
        """
        dl = DataLoader(
            LazyDataset(files=data),
            batch_size=1,
            num_workers=num_workers,
            prefetch_factor=prefetch_factor,
        )
        for x, _, _ in tqdm(dl, desc="Lazy fit progress"):
            x = (
                x.squeeze(0).reshape(-1, x.shape[-1]).to(dtype=torch.float64)
            )  # reshape for pre-batched files + float64 for overflow error
            self._batch_fit(x)

    @torch.no_grad()
    def fit(self, x):
        """
        Fit scaler in single step.

        Inputs:
            * x: tensor to fit to.
        """

        x_ = x.transpose(-1, self.dim.item()).reshape(-1, x.shape[self.dim.item()])

        if self.norm == "minmax":
            self.min = x_.min(dim=0, keepdim=True)[0]
            self.max = x_.max(dim=0, keepdim=True)[0]

        elif self.norm == "std":
            self.mean = x_.mean(dim=0, keepdim=True)[0]
            self.std = x_.std(dim=0, keepdim=True)[0]

        else:
            raise NotImplementedError

    @torch.no_grad()
    def transform(self, x):
        """
        Transform tensor with trained scaler.

        Inputs:
            * x: tensor to transform.

        Outputs:
            * Transformed tensor.
        """

        x_ = x.transpose(-1, self.dim.item()).reshape(-1, x.shape[self.dim.item()])

        if self.norm == "minmax":
            assert (
                self.min is not None and self.max is not None
            ), "Scaler is not fitted!"
            x_ = (x_ - self.min) / (self.max - self.min)

        elif self.norm == "std":
            assert (
                self.mean is not None and self.std is not None
            ), "Scaler is not fitted!"
            x_ = (x_ - self.mean) / self.std

        else:
            raise NotImplementedError

        return x_.reshape(x.transpose(-1, self.dim.item()).shape).transpose(
            -1, self.dim.item()
        )

    @torch.no_grad()
    def fit_transform(self, x):
        """
        Single-step fit scaler and then transform input data.

        Inputs:
            * x: tensor to fit to and transform.

        Outputs:
            * Transformed tensor.
        """
        self.fit(x)
        return self.transform(x)

    @torch.no_grad()
    def inverse_transform(self, x):
        """
        Revert transformation of tensor to return to original coordinates.

        Inputs:
            * x: tensor to de-transform.

        Outputs:
            * De-transformed tensor.
        """

        x_ = x.transpose(-1, self.dim.item()).reshape(-1, x.shape[self.dim.item()])

        if self.norm == "minmax":
            assert (
                self.min is not None and self.max is not None
            ), "Scaler is not fitted!"
            x_ = x_ * (self.max - self.min) + self.min

        elif self.norm == "std":
            assert (
                self.mean is not None and self.std is not None
            ), "Scaler is not fitted!"
            x_ = x_ * self.std + self.mean

        else:
            raise NotImplementedError

        return x_.reshape(x.transpose(-1, self.dim.item()).shape).transpose(
            -1, self.dim.item()
        )

    @torch.no_grad()
    def forward(self, x):
        """
        Forward method for scaler (implicitly called via `__call__`).
            Calls `transform` internally.

        Inputs:
            * x: tensor to transform.

        Outputs:
            * Transformed tensor.

        >>> scaler = Scaler(norm='std', dim=-1)
        >>> x = scaler(x0)  # transform tensor `x0`
        """
        return self.transform(x)

    def save(self, filename):
        """
        Save scaler's state (`state_dict`).

        Inputs:
            * filename: path to state file.
        """
        torch.save(deepcopy(self.state_dict()), filename)

    def load(self, filename):
        """
        Load state from file (`state_dict`). Automatically handles initialization parameters (see example).

        Inputs:
            * filename: path to state file.

        >>> f = 'path/to/state/file.pt'
        >>> sc0 = Scaler(norm='minmax', dim=-2)
        >>> sc0.save(f)
        >>> sc1 = Scaler(norm='std', dim=-1)
        >>> sc1.load(f)  # Correctly loads scaler with min-max normalization along dimension -1
        """
        sd = torch.load(filename, weights_only=True)

        is_minmax = {"min", "max"}.issubset(sd.keys())
        is_std = {"mean", "std"}.issubset(sd.keys())

        self.dim = sd["dim"]

        assert not (is_minmax and is_std)

        if is_minmax:
            self.min, self.max = sd["min"], sd["max"]
            self.norm = "minmax"
        elif is_std:
            self.mean, self.std = sd["mean"], sd["std"]
            self.norm = "std"


def generate_target(splits):
    r = np.random.rand()
    if r <= splits["train"]:
        return "train"
    elif r <= (splits["train"] + splits["valid"]):
        return "valid"
    elif r <= (splits["train"] + splits["valid"] + splits["test"]):
        return "test"
    else:
        return None


def filter_data(signal, rhythm, bpm, amp_thr=None, selected_rhythms=None):
    mask = np.ones(signal.shape[0])

    if amp_thr is not None:
        amp = np.array(
            [signal[i].max() - signal[i].min() for i in range(signal.shape[0])]
        )
        mask = mask * (amp > amp_thr)

    if selected_rhythms is not None:
        rmask = np.zeros(signal.shape[0])
        for r in selected_rhythms:
            rmask = rmask + (rhythm == r)
        mask = mask * rmask

    mask = mask.astype(bool)

    signal = signal[mask]
    rhythm = rhythm[mask]
    bpm = bpm[mask]

    return signal, rhythm, bpm
from collections import Counter
from glob import glob

if __name__ == "__main__":
    rootdir = os.path.split(os.path.dirname(os.path.realpath(__file__)))[0]

    with open(os.path.join(rootdir, "config.yaml"), "r") as f:
        config = yaml.safe_load(f)

    srcdir = (
        os.path.abspath(os.path.join(rootdir, config["paths"]["srcdir"], "p*"))
        if config["data"]["pidx"] is None
        else os.path.abspath(
            os.path.join(
                rootdir, config["paths"]["srcdir"], f"p{config['data']['pidx']:02d}"
            )
        )
    )
    datadir = os.path.abspath(os.path.join(rootdir, config["paths"]["datadir"]))

    splits = config["data"]["splits"]
    assert sum(splits.values()) <= 1.0, "Invalid split ratios: sum exceeds 1.0"

    if os.path.exists(datadir):
        shutil.rmtree(datadir)
        remove_existing = True
    else:
        remove_existing = False
    os.makedirs(datadir)
    for split_name in ["train", "valid", "test"]:
        os.makedirs(os.path.join(datadir, split_name))

    logging.basicConfig(
        filename=os.path.join(datadir, "datainfo.log"), level=logging.INFO
    )

    if remove_existing:
        logger.warning(f"Removing existing directory: {datadir}")

    segment_length = config["data"]["length"]
    num_segments = config["data"]["N"]
    oversampling = config["data"]["oversampling"]
    average_bpm = config["data"]["avg_bpm"]
    amplitude_threshold = config["data"]["amp_thr"]
    allowed_rhythms = config["data"]["rhythms"]
    batch_size = config["data"]["batch_size"]

    print(f"{os.path.join(srcdir, 'p*')=}")
    print(f"{os.getcwd()=}")
    patient_dirs = sorted(glob(os.path.join(srcdir, "p*")))

    np.random.seed(42)  # ensure reproducibility

    # Initialize label counters for each split
    label_counters = {
        "train": Counter(),
        "valid": Counter(),
        "test": Counter(),
    }

    for patient_path in tqdm(patient_dirs[:30]):
        split_target = generate_target(splits)
        if split_target not in label_counters:
            continue

        patient_id = os.path.split(patient_path)[-1]
        session_ids = [
            os.path.splitext(dat_file.split("_")[-1])[0]
            for dat_file in sorted(glob(os.path.join(patient_path, "*.dat")))
        ]

        signals_list = []
        rhythms_list = []
        bpms_list = []
        for session_id in session_ids:
            record_path = os.path.join(patient_path, f"{patient_id}_{session_id}")
            try:
                signal, rhythm, bpm = construct_arrays(record_path)
            except:
                logger.warning(f"Skipping {patient_id}_{session_id}: Data unavailable")
                continue

            signal, rhythm, bpm = sample(
                signal,
                rhythm,
                bpm,
                length=segment_length,
                N=num_segments,
                oversampling=oversampling,
                avg_bpm=average_bpm,
            )

            signal, rhythm, bpm = filter_data(
                signal,
                rhythm,
                bpm,
                amp_thr=amplitude_threshold,
                selected_rhythms=allowed_rhythms,
            )

            label_counters[split_target].update(rhythm.tolist())

            signals_list.append(signal)
            rhythms_list.append(rhythm)
            bpms_list.append(bpm)

        if len(signals_list) == 0:
            continue

        signals = np.concatenate(signals_list, axis=0)
        rhythms = np.concatenate(rhythms_list, axis=0)
        bpms = np.concatenate(bpms_list, axis=0)

        if not os.path.exists(os.path.join(datadir, patient_id[:3])):
            os.makedirs(os.path.join(datadir, patient_id[:3]))

        if batch_size is not None:
            num_batches = signals.shape[0] // batch_size
            L = signals.shape[1]
            C = signals.shape[2]

            signals = signals[: num_batches * batch_size].reshape(num_batches, batch_size, L, C)
            rhythms = rhythms[: num_batches * batch_size].reshape(num_batches, batch_size)
            bpms = (
                bpms[: num_batches * batch_size].reshape(num_batches, batch_size)
                if average_bpm
                else bpms[: num_batches * batch_size].reshape(num_batches, batch_size, L, C)
            )

            for batch_idx in range(num_batches):
                output_path = os.path.join(datadir, split_target, f"{patient_id}__b{batch_idx+1:03d}.npz")
                np.savez(
                    file=output_path,
                    signals=signals[batch_idx].astype(np.float16),
                    labels=rhythms[batch_idx].astype(np.uint8),
                    bpms=bpms[batch_idx].astype(np.float16),
                )
        else:
            output_path = os.path.join(datadir, split_target, f"{patient_id}.npz")
            np.savez(
                file=output_path,
                signals=signals.astype(np.float16),
                labels=rhythms.astype(np.uint8),
                bpms=bpms.astype(np.float16),
            )

    # Display class distribution by split
    print("\nClass distribution per dataset split:")
    for split in ["train", "valid", "test"]:
        print(f"\n→ {split.upper()}:")
        total_labels = sum(label_counters[split].values())
        if total_labels == 0:
            print("  [Empty]")
            continue
        for label in sorted(label_counters[split].keys()):
            count = label_counters[split][label]
            percentage = 100 * count / total_labels
            print(f"  Class {label}: {count} segments ({percentage:.2f}%)")

    # Show total segment count per split
    print("\nTotal number of labeled segments per split:")
    for split in ["train", "valid", "test"]:
        total = sum(label_counters[split].values())
        print(f"→ {split.upper()}: {total} segments")

    # Validate saved vs labeled segments per split
    for split in ["train", "valid", "test"]:
        saved_files = glob(os.path.join(datadir, split, "*.npz"))
        saved_segments = len(saved_files) * batch_size
        labeled_segments = sum(label_counters[split].values())

        print(f"\nValidation for split {split.upper()}:")
        print(f"→ Labeled segments : {labeled_segments}")
        print(f"→ Saved segments   : {saved_segments} in {len(saved_files)} .npz files")

        if labeled_segments > saved_segments:
            discarded = labeled_segments - saved_segments
            print(f"Warning: {discarded} segments were discarded (not enough to form a full batch of size {batch_size}).")
        elif labeled_segments < saved_segments:
            print("Warning: more segments were saved than labeled. This should not happen.")
        else:
            print("All labeled segments were saved successfully.")
