from copy import deepcopy

import torch


def longtiming(secs):
    """
    Function for expressing long time intervals, from seconds into hours:minutes:seconds.

    Inputs:
        * secs: interval in seconds (as given by `time.time()`).

    Outputs:
        * Formatted string of hours : minutes : seconds.
    """
    hours, remainder = divmod(secs, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{int(hours): 3d}:{int(minutes):02d}:{int(seconds):02d}"


class EarlyStopper:
    """
    General class for implementing early stopping in training.
    """

    def __init__(self, patience, min_delta):
        """
        Initializer for `EarlyStopper`.

        Inputs:
            * patience: number of calls (epochs) without improvement before sending stopping signal.
            * min_delta: minimum improvement to achieve before resetting step counter.
        """
        self.cnt = 0
        self.patience = patience
        self.min_delta = min_delta
        self.best_loss = torch.inf
        self.update_loss = torch.inf
        self.best_model = None

    def __call__(self, loss, state_dict):
        """
        Check if early stopping must be applied.

        Inputs:
            * loss: validation loss or metric to minimize.
            * state_dict: training model's state to be stored.

        Outputs:
            * Boolean signal for stopping (True = 'stop').
        """

        if loss < self.best_loss:
            self.best_model = deepcopy(state_dict)
            self.best_loss = loss

        if loss < self.update_loss:
            self.update_loss = loss - self.min_delta
            self.cnt = 0
            return False
        if self.patience == 0:
            return False
        else:
            self.cnt += 1
            if self.cnt == self.patience:
                return True
            else:
                return False

    def reset(self):
        """
        Reset early stopper.
        """
        self.cnt = 0
        self.best_loss = torch.inf
        self.update_loss = torch.inf
        self.best_model = None

    def save(self, filename):
        """
        Save best model (`state_dict`) to file.

        Inputs:
            * filename: path to model file.
        """
        torch.save(self.best_model, filename)
