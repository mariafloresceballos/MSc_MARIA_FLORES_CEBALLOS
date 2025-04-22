from glob import glob
import joblib
import logging
import os
import shutil
from typing import Iterable
import yaml

import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import torch
from torch.utils.data import Dataset
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


def sample(signal, rhythm, bpm, *, length=1000, oversampling=2.0, avg_bpm=False):
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
                    │   ├── x01.npx
                    │   ├── x02.npx
                    │   └── ...
                    ├── train/
                    │   ├── x03.npx
                    │   ├── x04.npx
                    │   └── ...
                    └── valid/
                        └── ...

                    Each .npz must contain three arrays:
                    'signals', 'labels' (rhythm annotations) and 'bpms' (BPM annotations).
        """
        self.keys = ["train", "valid", "test"]
        self.folder = folder

        self.files = (
            {k: sorted(glob(os.path.join(self.folder, k, "*.np*"))) for k in self.keys}
            if isinstance(self.keys, Iterable)
            else sorted(glob(os.path.join(self.folder, self.keys, "*.np*")))
        )

    @staticmethod
    def read(
        f,
        *,
        output_fmt="numpy",
        return_signal=True,
        return_label=False,
        return_bpm=False,
    ):
        """
        Read file to dictionary of arrays with same labels as npx file.

        Inputs:
            * f: path of file to be read.

        Outputs:
            * Dictionary containing data.
        """
        _valid_output_fmts = ["numpy", "torch"]
        assert (
            output_fmt in _valid_output_fmts
        ), f"Output format not recognised. Expected one of {_valid_output_fmts}, got {output_fmt}."

        if os.path.splitext(f)[1] == ".npz":
            data = np.load(f)
            signals = data["signals"] if return_signal else None
            labels = data["labels"] if return_label else None
            bpms = data["bpms"] if return_bpm else None

        elif os.path.splitext(f)[1] == ".npy":
            if return_label or return_bpm:
                raise RuntimeError(
                    "Labels and BPMs cannot be returned with file format '.npy'"
                )

        else:
            raise NotImplementedError(
                f"Unrecognised file extension found: {os.path.splitext(f)[1]}"
            )

        if output_fmt == "numpy":
            pass
        elif output_fmt == "torch":
            signals = torch.from_numpy(signals) if signals is not None else None
            labels = torch.from_numpy(labels) if labels is not None else None
            bpms = torch.from_numpy(bpms) if bpms is not None else None
        else:
            raise NotImplementedError()
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

    def __init__(self, files, *, signal=True, label=False, bpm=False):
        """
        Initializer for `LazyDataset`.

        Inputs:
            * files: list of file (paths) to load in the dataset.
            * scaler: whether to use a `Scaler` or not.
        """
        self.files = files
        self.num_files = len(files)
        self.signal = signal
        self.label = label
        self.bpm = bpm

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
        signals, labels, bpms = Reader.read(
            f=f,
            return_signal=self.signal,
            return_label=self.label,
            return_bpm=self.bpm,
            output_fmt="torch",
        )

        batch_size = [t for t in [signals, labels, bpms] if t is not None][0].shape[0]

        signals = signals if signals is not None else torch.empty(batch_size)
        labels = labels if labels is not None else torch.empty(batch_size)
        bpms = bpms if bpms is not None else torch.empty(batch_size)

        return signals, labels, bpms


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


def filter_data(signal, rhythm, bpm, amplitude_threshold=None, selected_rhythms=None):
    mask = np.ones(signal.shape[0])

    if amplitude_threshold is not None:
        amplitude = np.array(
            [signal[i].max() - signal[i].min() for i in range(signal.shape[0])]
        )
        mask = mask * (amplitude > amplitude_threshold)

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


if __name__ == "__main__":
    rootdir = os.path.split(os.path.dirname(os.path.realpath(__file__)))[0]

    with open(os.path.join(rootdir, "configs", "data.yaml"), "r") as f:
        config = yaml.safe_load(f)

    srcdir = os.path.abspath(os.path.join(config["srcdir"], "p*"))
    selected_pidx = (
        [f"p{pidx:02d}" for pidx in config['pidx']]
        if isinstance(config["pidx"], Iterable)
        else f"p{config['pidx']:02d}" if config["pidx"] is not None else None
    )
    datadir = os.path.abspath(config["datadir"])

    splits = config["splits"]
    split_sum = sum(splits.values())
    assert split_sum <= 1.0, f"Sum of splits exceeded unity: Sum={split_sum:.2f}"

    if os.path.exists(datadir):
        shutil.rmtree(datadir)
        remove_existing = True
    else:
        remove_existing = False
    os.makedirs(datadir)
    for s in ["train", "valid", "test"]:
        os.makedirs(os.path.join(datadir, s))

    logging.basicConfig(
        filename=os.path.join(datadir, "datainfo.log"), level=logging.INFO
    )

    if remove_existing:
        logger.warning(f"Removing existing directory: {datadir}")

    shutil.copy(
        os.path.join(rootdir, "configs", "data.yaml"),
        os.path.join(datadir, "data.yaml"),
    )

    rng_seed = config["seed"]
    save_npz = config["save_npz"]

    length = config["length"]
    oversampling = config["oversampling"]
    avg_bpm = config["avg_bpm"]
    ampl_thr = config["amplitude_threshold"]
    selected_rhythms = config["rhythms"]
    batch_size = config["batch_size"]

    patients = sorted(glob(os.path.join(srcdir, "p*")))

    if selected_pidx is not None:
        patients = [
            p
            for p in patients
            if os.path.split(os.path.split(p)[0])[1] in selected_pidx
        ]

    np.random.seed(rng_seed)  # reproducibility

    for p in tqdm(patients):
        target = generate_target(splits)
        if target is None:
            continue
        pid = os.path.split(p)[-1]
        sessions = [
            os.path.splitext(datfile.split("_")[-1])[0]
            for datfile in sorted(glob(os.path.join(p, "*.dat")))
        ]

        psignal = []
        prhythm = []
        pbpm = []
        for s in sessions:
            f = os.path.join(p, f"{os.path.split(p.rstrip('/'))[-1]}_{s}")
            try:
                s, r, b = construct_arrays(f)
            except:
                logger.warning(f"Skipping {pid}_{s}: Data unavailable")
                continue
            signal, rhythm, bpm = sample(
                s,
                r,
                b,
                length=length,
                oversampling=oversampling,
                avg_bpm=avg_bpm,
            )

            signal, rhythm, bpm = filter_data(
                signal,
                rhythm,
                bpm,
                amplitude_threshold=ampl_thr,
                selected_rhythms=selected_rhythms,
            )

            psignal.append(signal)
            prhythm.append(rhythm)
            pbpm.append(bpm)

        psignal = np.concatenate(psignal, axis=0)
        prhythm = np.concatenate(prhythm, axis=0)
        pbpm = np.concatenate(pbpm, axis=0)

        if batch_size is not None:
            bidx = 0
            Nbatch = psignal.shape[0] // batch_size
            L = psignal.shape[1]
            C = psignal.shape[2]

            psignal = psignal[: Nbatch * batch_size].reshape(Nbatch, batch_size, L, C)
            prhythm = prhythm[: Nbatch * batch_size].reshape(Nbatch, batch_size)
            pbpm = (
                pbpm[: Nbatch * batch_size].reshape(Nbatch, batch_size)
                if avg_bpm
                else pbpm[: Nbatch * batch_size].reshape(Nbatch, batch_size, L, C)
            )

            for i in range(Nbatch):
                bidx += 1

                if save_npz:
                    fname = os.path.join(datadir, target, pid + f"__b{bidx:03d}.npz")
                    np.savez(
                        file=fname,
                        signals=psignal[i].astype(np.float16),
                        labels=prhythm[i].astype(np.uint8),
                        bpms=pbpm[i].astype(np.float16),
                    )
                else:
                    fname = os.path.join(datadir, target, pid + f"__b{bidx:03d}.npy")
                    np.save(file=fname, arr=psignal[i].astype(np.float16))

        else:
            if save_npz:
                fname = os.path.join(datadir, target, pid + ".npz")
                np.savez(
                    file=fname,
                    signals=psignal.astype(np.float16),
                    labels=prhythm.astype(np.uint8),
                    bpms=pbpm.astype(np.float16),
                )
            else:
                fname = os.path.join(datadir, target, pid + ".npy")
                np.save(file=fname, arr=psignal.astype(np.float16))

    if config["scaler"] is not None:
        reader = Reader(folder=datadir)
        scaler = (
            StandardScaler()
            if config["scaler"] == "std"
            else MinMaxScaler if config["scaler"] == "minmax" else None
        )
        if scaler is None:
            raise NotImplementedError(f"Something went wrong... {(scaler is None)=}")

        for f in tqdm(reader(key="train")):
            _x = reader.read(f=f)
            scaler.partial_fit(_x.reshape(-1, _x.shape[-1]))

        joblib.dump(scaler, os.path.join(datadir, "scaler.save"))
        print(f"\nScaler saved to:\n\n{os.path.join(datadir, 'scaler.save')}\n\n")

        keys = ["train", "valid", "test"]

        for k in keys:
            print(f"\n\nScaling [{k.upper()}] split...\n")

            for f in tqdm(reader(key=k)):
                _x = reader.read(f=f)
                _shape = _x.shape
                x = scaler.transform(_x.reshape(-1, _shape[-1])).reshape(*_shape)
                np.save(file=f, arr=x)
