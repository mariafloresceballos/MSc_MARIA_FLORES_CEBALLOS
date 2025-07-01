from copy import deepcopy
import logging

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


class ResNeXtBlock(nn.Module):
    """
    ResNeXt block.
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        cardinality,
        activation=nn.SiLU,
        device=None,
        dtype=None,
    ):
        """
        Initializer for `ResNeXtBlock`.

        Inputs:
            * in_channels: number of channels of input signal.
            * out_channels: number of channels of output signal. Must be divisible by `in_channels`.
            * cardinality: ResNeXt's cardinality or number of groups in convolutional layers
                (see https://arxiv.org/abs/1611.05431).
            * activation: activation function, class of `torch.nn`
                (see https://pytorch.org/docs/stable/nn.html#non-linear-activations-weighted-sum-nonlinearity).
            * device
            * dtype
        """
        super().__init__()
        factory_kwargs = {"device": device, "dtype": dtype}

        if in_channels != out_channels:
            assert (
                out_channels % in_channels == 0
            ), f"Output channels must be divisible by input channels, but got {out_channels} and {in_channels}."
            stride = int(out_channels // in_channels)
            self.downsample = nn.Sequential(
                nn.Conv1d(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=1,
                    stride=stride,
                    **factory_kwargs,
                ),
                nn.BatchNorm1d(num_features=out_channels, **factory_kwargs),
            )
        else:
            stride = 1
            self.downsample = nn.Identity()

        self.conv1x1_1 = nn.Conv1d(
            in_channels=in_channels,
            out_channels=in_channels,
            kernel_size=1,
            **factory_kwargs,
        )
        self.bn1x1_1 = nn.BatchNorm1d(num_features=in_channels, **factory_kwargs)

        self.conv3x3 = nn.Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            groups=cardinality,
            **factory_kwargs,
        )
        self.bn3x3 = nn.BatchNorm1d(num_features=out_channels, **factory_kwargs)

        self.conv1x1_2 = nn.Conv1d(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=1,
            **factory_kwargs,
        )
        self.bn1x1_2 = nn.BatchNorm1d(num_features=out_channels, **factory_kwargs)

        self.activation = activation()

    def forward(self, x):
        """
        Forward method for `ResNeXtBlock`.

        Inputs:
            * x: tensor to feed-forward through block.

        Outputs:
            * Transformed tensor.
        """
        res = self.downsample(x)

        x = self.conv1x1_1(x)
        x = self.activation(self.bn1x1_1(x))

        x = self.conv3x3(x)
        x = self.activation(self.bn3x3(x))

        x = self.conv1x1_2(x)
        x = self.bn1x1_2(x)

        x = x + res
        x = self.activation(x)

        return x


class ResNeXt(nn.Module):
    """
    Full ResNeXt model (see https://arxiv.org/abs/1611.05431).
    """

    def __init__(
        self,
        *,
        in_channels,
        h0_channels,
        out_features,
        cardinality,
        num_blocks,
        activation=nn.SiLU,
        device=None,
        dtype=None,
    ):
        """
        Initializer for `ResNeXt` model.

        Inputs:
            * in_channels: number of channels in signal input.
            * h0_channels: number of channels of the input to the first hidden layer.
            * out_features: number of classes to classify.
            * cardinality: number of branches inside each block.
            * num_blocks: number of blocks in each stage. The first block in each stage
                doubles the number of channels and halves the spatial dimensions.
            * activation: activation function.
            * device
            * dtype
        """
        super().__init__()
        factory_kwargs = {"device": device, "dtype": dtype}

        hidden_channels = h0_channels
        assert (
            hidden_channels % cardinality == 0
        ), f"Hidden channels must be divisible by cardinality, but got {hidden_channels} and {cardinality}."

        self.conv0 = nn.Conv1d(
            in_channels=in_channels,
            out_channels=hidden_channels,
            kernel_size=7,
            stride=2,
            padding=3,
            **factory_kwargs,
        )
        self.bn0 = nn.BatchNorm1d(num_features=hidden_channels, **factory_kwargs)
        self.act0 = activation()
        self.maxpool0 = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)

        blocks = []

        for nb in num_blocks:
            blocks.append(
                ResNeXtBlock(
                    in_channels=hidden_channels,
                    out_channels=2 * hidden_channels,
                    cardinality=cardinality,
                    activation=activation,
                    **factory_kwargs,
                )
            )
            hidden_channels = 2 * hidden_channels
            for _ in range(nb - 1):
                blocks.append(
                    ResNeXtBlock(
                        in_channels=hidden_channels,
                        out_channels=hidden_channels,
                        cardinality=cardinality,
                        activation=activation,
                        **factory_kwargs,
                    )
                )

        self.conv_stack = nn.Sequential(*blocks)

        self.avgpool = nn.AdaptiveAvgPool1d(output_size=1)

        self.fc = nn.Linear(
            in_features=hidden_channels, out_features=out_features, **factory_kwargs
        )

    def forward(self, x, raw=False):
        """
        Forward method for `ResNeXt` model.

        Inputs:
            * x: input data.
            * raw: whether or not to apply softmax to output.

        Outputs:
            * Transformed data for classification (logit if `raw=True`).
        """
        x = self.maxpool0(self.act0(self.bn0(self.conv0(x))))
        x = self.conv_stack(x)
        x = torch.flatten(self.avgpool(x), start_dim=1)
        x = self.fc(x)
        if not raw:
            x = F.softmax(x, dim=-1)
        return x

    def save(self, file):
        """
        Save model state.

        Inputs:
            * file: path to state file.
        """
        torch.save(deepcopy(self.state_dict()), file)

    def load(self, file):
        """
        Load model state.

        Inputs:
            * file: path to state file.
        """
        self.load_state_dict(torch.load(file, weights_only=True))
