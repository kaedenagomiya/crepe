import os
import sys
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torchaudio

from crepe.utils import get_frame, activation_to_frequency

class ConvBlock(nn.Module):
    """
    class of Convolutional block.
    """

    def __init__(self, out_channels:int, kernel_width:int, stride:tuple[int], in_channels:int):
        """
        Initialize Convolutional block.
        
        Args:
            out_channels (int): The number of output channels.
            kernel_size (int): The size of the convolution kernel.
            stride (tuple, int): The stride for each dimention.
            in_channels (int): The number of input channels.
        """
        super(ConvBlock, self).__init__()

        # Calculate padding for the height dimension (kernel width)
        pad_top = (kernel_width - 1) // 2
        pad_bottom = (kernel_width - 1) - pad_top

        # Define the block using nn.Sequential
        self.layer = nn.Sequential(
            # Add padding to the input
            nn.ZeroPad2d((0, 0, pad_top, pad_bottom)),
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=(kernel_width, 1),
                stride=stride
            ),  # Apply 2D convolution
            nn.ReLU(),  # Apply ReLU activation
            nn.BatchNorm2d(out_channels),  # Apply batch normalization
            nn.MaxPool2d(kernel_size=(2, 1)),  # Apply max pooling
            nn.Dropout(p=0.25)  # Apply dropout for regularization
        )

    def forward(self, x):
        """
        process an input tensor 'x' through the ConvBlock.
        
        Args:
            x (torch.Tensor): The input tensor for the ConvBlock.

        Returns:
            torch.Tensor: The output of the ConvBlock.
        """
        return self.layer(x)



class CREPE(nn.Module):
    """
    CREPE model.
    
    Ref:
    Kim, Jong Wook, et al.
    "Crepe: A convolutional representation for pitch estimation."
    2018 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP).
    IEEE, 2018.
    URL: https://api.semanticscholar.org/CorpusID:3344371
    """
    def __init__(self, pretrained_path:str=None, device='cpu', model_capacity:str='tiny'):
        """
        CREPE model for pitch(F0) estimation.

        Args:
            model_capacity (str): The capacity of the model ('tiny','small','medium', 'large', 'full').
        """

        super(CREPE, self).__init__()
        
        # Define model capacity.
        self.model_capacity = model_capacity
        capacity_coef = {
            'tiny': 4, 'small': 8, 'medium': 16, 'large': 24, 'full': 32,
        }[model_capacity]

        # for each layer.
        # Define the number of filters.
        filters = [n * capacity_coef for n in [32, 4, 4, 4, 8, 16]]
        filters = [1] + filters # add the input channel size, so [1, 32, 4, 4, 4, 8, 16]

        # Define the kernel size and strides.
        widths = [512, 64, 64, 64, 64, 64]
        strides = [(4, 1), (1, 1), (1, 1), (1, 1), (1, 1), (1, 1)]

        # Define list of layers for ConvBlock
        layers = []
        for i in range(len(filters) - 1):
            layers.append(
                ConvBlock(
                    out_channels=filters[i + 1],
                    kernel_width=widths[i],
                    stride=strides[i],
                    in_channels=filters[i]
                )
            )

        self.conv_blocks = nn.Sequential(*layers)

        # Define final layer to detect 360 class for pitch detection
        self.linear = nn.Linear(64 * capacity_coef, 360)

        # setting for loading model
        self.device = device
        #self.load_weight_from_cap(model_capacity)
        self.load_weight_from_path(pretrained_path)
        self.eval()


    def forward(self, x):
        """
        process an input tensor 'x' through the CREPE.
        
        Args:
            x (torch.Tensor): The input tensor for the CREPE.

        Returns:
            torch.Tensor: The output of the CREPE.
        """
        # x : shape (batch, sample)
        x = x.view(x.shape[0], 1, -1, 1)
        x = self.conv_blocks(x)

        # Reorder dimention and flatten for linear layer.
        x = x.permute(0, 3, 2, 1)
        x = x.reshape(x.shape[0], -1)

        x = self.linear(x)
        x = torch.sigmoid(x)
        
        return x

    def load_weight_from_path(self, model_weight_path:str):
        """
        Load the weight for the model from path.
        
        Args:
            model_weight_path (str): The path of weight file for model.
        """
        #package_dir = os.path.dirname(os.path.realpath(__file__))
        #full_path = os.path.join(package_dir, model_weight_path)
        full_path = Path(model_weight_path).resolve()

        try:
            self.load_state_dict(
                torch.load(
                    full_path,
                    map_location=torch.device(self.device)
                )
            )
            print(f"Successfully loaded weights from {full_path}.")
        except FileNotFoundError:
            print(f"File for model weight not found: {full_path}")
        except Exception as e:
            print(f"Error loading weights: {e}")


    def load_weight_from_cap(self, model_capacity:str):
        """
        Load the weights for the model.

        Args:
            model_capacity (str): The capacity of the model('tiny', 'small', 'medium', 'large', or 'full').
        """
        
        package_dir = os.path.dirname(os.path.realpath(__file__))
        filename = "crepe_{}.pth".format(model_capacity)
        try:
            self.load_state_dict(
                torch.load(
                    os.path.join(package_dir, filename),
                    map_location=torch.device(self.device)
                )
            )
        except:
            print(f"{filename} Not found.")


    def get_activation(self, audio, samplerate:int, center:bool=True, step_size:int=10, batch_size:int=128):
        """
        Processing for the activation stack.

        Args:
            audio (torch.Tensor): The input audio tensor. audio.shape is (N,) or (C, N).
            samplerate (int): The samplingrate of the audio signal using training data.
            center (bool): Specifies whether the frame should be centered or not. Defaults to True.
            step_size (int): The number of samples per frame. Defaults to 10.
            batch_size (int): The batch size for activation stack.
        
        Returns:
            torch.Tensor: The activation stack.
        """

        # resample to 16kHz.
        if samplerate != 16000:
            resampler = torchaudio.transforms.Resample(samplerate, 16000)
            audio = resampler(audio)

        # convert to mono if needed
        if len(audio.shape) == 2:
            if audio.shape[0] == 1:
                audio = audio[0]
            else:
                audio = audio.mean(dim=0)

        frames = get_frame(audio, step_size, center)
        activation_stack = []
        device = self.linear.weight.device

        for i in range(0, len(frames), batch_size):
            f = frames[i:min(i+batch_size, len(frames))]
            f = f.to(device)
            act = self.forward(f)
            activation_stack.append(act.cpu())

        activation = torch.cat(activation_stack, dim=0)
        return activation


    def predict(self, audio, samplerate:int, center:bool=True, step_size:int=10, batch_size:int=128):
        """
        predict pitch(F0) class from the input audio signal.

        Args:
            audio (torch.Tensor): The input audio tensor. audio.shape is (N,) or (C, N).
            samplerate (int): The samplingrate of the audio signal using training data.
            center (bool): Specifies whether the frame should be centered or not. Defaults to True.
            step_size (int): The number of samples per frame. Defaults to 10.
            batch_size (int): The batch size for activation stack.
        
        Returns:
            tuple: This tuple containing the time, frequency, confidence, and activation stack.
        """

        activation = self.get_activation(
            audio, samplerate, batch_size=batch_size, step_size=step_size)
        frequency = activation_to_frequency(activation)
        confidence = activation.max(dim=1)[0]
        time = torch.arange(confidence.shape[0]) * step_size / 1000.0
        return time, frequency, confidence, activation
