from copy import deepcopy
import logging

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)

# ConvResNet block
# According to arXiv:1707.01836v1 there's a block used 15 times. In order to achieve efficiency, let´s code this block apart.
# All the convolutional layers have kernel size 16 and according to the paper and every alternate residual block subsamples its inputs by a factor of 2.
# Maxpool is applied to the residual connection in the subsampled blocks with the same subsample factor than in the alternate blocks.

# Regarding the original paper, some hyperparameters have been modified in order to apply the model correctly in our study. 
# Some of these hyperparameters were not specified in the original paper.

class ConvResNetBlock(nn.Module):
    """
    ConvResNet block.
    """

    def __init__(
            self, 
            in_channels, 
            out_channels,
            subsample,
            activation = nn.ReLU,
            device = None,
            dtype = None,
    ):
        """
        Inputs of the block:
            * in_channels: number of channels of input signal.
            * out_channels: number of channels of output signal. Must be divisible by `in_channels`
            * activation: activation function, class of `torch.nn`
            * subsample: boolean variable, True if subsample is performed in the actual block, False otherwise.
            * device
            * dtype
        """
        super().__init__()
        factory_kwargs = {"device": device, "dtype": dtype}

        # Batch normalization layers
        self.batchnorm1 = nn.BatchNorm1d(num_features=in_channels, **factory_kwargs).to(device)
        self.batchnorm2 = nn.BatchNorm1d(num_features=out_channels, **factory_kwargs).to(device)

        # Activation
        self.activation = activation()

        # Convolutional layers
        self.conv1 = nn.Conv1d(
            in_channels=in_channels,
            out_channels=out_channels, 
            kernel_size=15,   
            stride=2 if subsample else 1,  
            padding=7,
            **factory_kwargs,
        ).to(device)

        self.conv2 = nn.Conv1d(
            in_channels=out_channels,
            out_channels=out_channels, 
            kernel_size=15,   
            stride=1,
            padding=7,
            **factory_kwargs,
        ).to(device)

        # Maxpool
        self.maxpool = nn.MaxPool1d(kernel_size=2, stride=2).to(device) if subsample else nn.Identity().to(device)

        # Channel upsampling layer 
        self.channel_upsampling = nn.Identity() if in_channels == out_channels else nn.Conv1d(in_channels=in_channels, 
                                                                                              out_channels=out_channels, 
                                                                                              kernel_size=1, 
                                                                                              stride=1, 
                                                                                              padding=0, 
                                                                                              **factory_kwargs)

        # Dropout
        self.dropout = nn.Dropout(0.2)  

    def forward(self, x):
        """
        Forward method for `ConvResNetBlock`.

        Inputs:
            * x: tensor to feed-forward through block.

        Outputs:
            * Transformed tensor.
        """

        res = x

        x = self.batchnorm1(x) 
        x = self.activation(x)  
        x = self.dropout(x)  
        x = self.conv1(x) 
        x = self.batchnorm2(x)  
        x = self.activation(x)  
        x = self.dropout(x)  
        x = self.conv2(x)  

        res = (
            F.adaptive_max_pool1d(res, x.shape[-1])
            if x.shape[-1] != res.shape[-1]
            else res
        )

        res = self.channel_upsampling(res)

        x = x + res

        return x


class ConvResNet(nn.Module):
    """
    Full ConvResNet model (see arXiv:1707.01836v1).
    """

    def __init__(
        self,
        *,
        in_channels,
        h0_channels,
        out_features,
        num_blocks,
        activation=nn.ReLU,
        subsample = False,
        device=None,
        dtype=None,
    ):
        """
        Initializer for `ConvResNet` model.

        Inputs:
            * in_channels: number of channels in signal input.
            * h0_channels: number of channels of the input (no es output?) to the first hidden layer.
            * out_features: number of classes to classify.
            * num_blocks: number of repeated blocks. 
            * activation: activation function.
            * device
            * dtype
        """
        super().__init__()
        factory_kwargs = {"device": device, "dtype": dtype}

        # Initial definitions
        hidden_channels = h0_channels 

        # Batch normalization
        self.batchnorm0 = nn.BatchNorm1d(num_features=hidden_channels, **factory_kwargs).to(device)
        self.batchnorm1 = nn.BatchNorm1d(num_features=hidden_channels, **factory_kwargs).to(device)

        # Convolutional layers
        self.conv0 = nn.Conv1d(
            in_channels=in_channels,
            out_channels=hidden_channels,
            kernel_size=15,   
            stride=1, 
            padding=7, 
            **factory_kwargs,
        ).to(device)
 
        self.conv1 = nn.Conv1d(
            in_channels=hidden_channels,  
            out_channels=hidden_channels, 
            kernel_size=15, 
            stride=2,  
            padding=7,
            **factory_kwargs,
        ).to(device)

        self.conv2 = nn.Conv1d(
            in_channels=hidden_channels,
            out_channels=hidden_channels, 
            kernel_size=15, 
            stride=1, 
            padding=7,
            **factory_kwargs,
        ).to(device)


        # ConvResNet blocks
        blocks = []
        k = 2
        for i in range(num_blocks):  
            subsample = ((i + 1 ) % 2 == 0)  
            if i % 4 == 0 and i > 0:  
                out_channels = 64 * k
                k = k + 1
            else:
                out_channels = hidden_channels 
            
            blocks.append(ConvResNetBlock(hidden_channels, out_channels, subsample, **factory_kwargs))
            hidden_channels = out_channels 
            in_channels = out_channels

        self.conv_stack = nn.Sequential(*blocks)

        # Additional batch normalization
        self.batchnorm2 = nn.BatchNorm1d(num_features=out_channels, **factory_kwargs).to(device)

        # Fully-connected layer
        self.fc = nn.Linear(
            in_features=out_channels, out_features=out_features, **factory_kwargs
        ) 

        # Activation
        self.activation = activation()

        # Activation
        self.maxpool0 = nn.MaxPool1d(kernel_size=2, stride=2).to(device)

        # Dropout
        self.dropout = nn.Dropout(0.2) 

        # Average pooling
        self.avgpool = nn.AdaptiveAvgPool1d(output_size=1) 

    def forward(self, x, raw=False):
        """
        Forward method for `ConvResNet` model.

        Inputs:
            * x: input data.
            * raw: whether or not to apply softmax to output.

        Outputs:
            * Transformed data for classification (logit if `raw=True`).
        """
        x = self.conv0(x) 
        x = self.batchnorm0(x) 
        x = self.activation(x) 

        res = self.maxpool0(x) 

        x = self.conv1(x) 
        x = self.batchnorm1(x) 
        x = self.activation(x) 
        x = self.dropout(x) 
        x = self.conv2(x) 
        x = x + res 

        x = self.conv_stack(x) 

        x = self.batchnorm2(x) 
        x = self.activation(x) 

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
