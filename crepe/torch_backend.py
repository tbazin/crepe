from __future__ import division
from __future__ import print_function

import os
import typing
from typing import List, Dict, Optional, Any
import numpy as np

import torch
from torch import nn
import torch.nn.functional as F
import h5py

# store as a global variable, since we only support a few models for now
models: Dict[str, Optional['CREPE']] = {
    'tiny': None,
    'small': None,
    'medium': None,
    'large': None,
    'full': None
}


def _get_keras_weights(weights: h5py.File, group_name: str,
                       parameter_name: str) -> torch.Tensor:
    """Retrieve weights for a given layer's parameter
    Example usage:
    >>> weights = h5py.File(PATH_TO_MODEL, 'r')
    >>> _get_keras_weights(weights, 'conv1', 'weight')
    np.ndarray([...])
    """
    parent_group = weights[group_name]
    access_key = next(iter(parent_group.keys()))
    group = parent_group[access_key]
    return torch.as_tensor(group[parameter_name + ':0'][()])


# custom conv1d since PyTorch does not support 'SAME' padding as in TensorFlow
# taken from:
# https://github.com/Gasoonjia/Tensorflow-type-padding-with-pytorch-conv2d./
#   blob/master/Conv2d_tensorflow.py
class Conv1d_samePadding(nn.Conv1d):
    def __init__(self, *args, padding: int = 0, **kwargs):
        assert padding == 0, "no additional padding on top of 'same' padding"
        kwargs['padding'] = 0
        super().__init__(*args, **kwargs)

    def same_padding_1d(self, input):
        input_duration = input.size(2)
        filter_duration = self.weight.size(2)
        out_duration = (input_duration + self.stride[0] - 1) // self.stride[0]
        padding_duration = max(
            0,
            ((out_duration - 1) * self.stride[0]
             + (filter_duration - 1) * self.dilation[0] + 1 - input_duration))
        duration_odd = padding_duration % 2

        input = F.pad(input, (padding_duration // 2,
                              padding_duration // 2 + int(duration_odd)))

        return input

    def forward(self, input):
        input = self.same_padding_1d(input)
        return super().forward(input)


class CrepeLayer(nn.Module):
    """Define a single convolutional layer for the CREPE model"""
    def __init__(self, in_channels: int, filters: int,
                 width: int, stride: int, layer_index: int):
        super().__init__()
        self.layer_index = layer_index

        self.conv = Conv1d_samePadding(
            in_channels=in_channels, out_channels=filters, stride=stride,
            kernel_size=width)
        self.batch_norm = nn.BatchNorm1d(filters)
        self.maxpool = nn.MaxPool1d(kernel_size=2, stride=None, padding=0)
        self.dropout = nn.Dropout(0.25)

    def forward(self, input):
        input = self.conv(input)
        input = F.relu(input)
        input = self.batch_norm(input)
        input = self.maxpool(input)
        input = self.dropout(input)
        return input

    def load_keras_weights(self, weights: h5py.File):
        """Load weights from a keras pretrained layer"""
        h5_layer_name = f'conv{self.layer_index}'

        # load the weights for the convolutional layer
        conv_layer_name = h5_layer_name
        conv_kernel_matrix = (
            _get_keras_weights(weights, conv_layer_name, 'kernel')
            # squeezing since the TF implementation uses 2D convolutions
            # although the kernels are actually 1D
            .squeeze(1)
            # transpose to adapt to Keras' weights shape
            .transpose(0, 2))
        assert self.conv.weight.data.shape == conv_kernel_matrix.shape
        self.conv.weight.data = conv_kernel_matrix

        self.conv.bias.data = _get_keras_weights(
            weights, conv_layer_name, 'bias')

        # load the weights for the BatchNorm layer
        bn_layer_name = h5_layer_name + '-BN'
        self.batch_norm.weight.data = _get_keras_weights(
            weights, bn_layer_name, 'gamma')
        self.batch_norm.bias.data = _get_keras_weights(
            weights, bn_layer_name, 'beta')
        self.batch_norm.running_mean.data = _get_keras_weights(
            weights, bn_layer_name, 'moving_mean')
        self.batch_norm.running_var.data = _get_keras_weights(
            weights, bn_layer_name, 'moving_variance')


class CREPE(nn.Module):
    """Define a 6-layer CREPE pitch-prediction model"""
    # the model is trained on 16kHz audio
    fs_hz: int = 16000

    # number of output classes
    num_classes: int = 360

    # the model is monophonic
    in_channels = 1

    # convolutional layers settings as used in the TF implementation
    widths = [512, 64, 64, 64, 64, 64]
    strides = [4, 1, 1, 1, 1, 1]
    # base values to be multiplied by the capacity multiplier
    num_filters_base = [32, 4, 4, 4, 8, 16]

    capacity_multipliers_per_capacity = {
            'tiny': 4, 'small': 8, 'medium': 16, 'large': 24, 'full': 32
        }

    # pad so that frames are centered around their timestamps (i.e. first frame
    # is zero centered).
    center = True

    def __init__(self, model_capacity: str, frame_duration_n: int = 1024,
                 hop_length_s: float = 10e-3,
                 capacity_multiplier: Optional[int] = None):
        super().__init__()
        self.frame_duration_n = frame_duration_n
        self.hop_length_n = int(self.fs_hz * hop_length_s)

        if capacity_multiplier is None:
            self.model_capacity = model_capacity
            self.capacity_multiplier = self.capacity_multipliers_per_capacity[
                self.model_capacity]
        else:
            self.model_capacity = 'custom'
            self.capacity_multiplier = capacity_multiplier
        self.num_filters = [f * self.capacity_multiplier
                            for f in self.num_filters_base]

        layers: List[CrepeLayer] = []

        # initial number of channels
        out_channels_prev_layer = self.in_channels
        for i, (f, w, s) in enumerate(
                zip(self.num_filters, self.widths, self.strides)):
            layer_index = i+1  # for consistency with the TF implementation
            layers.append(CrepeLayer(out_channels_prev_layer, f, w, s,
                                     layer_index))

            out_channels_prev_layer = f

        # self.crepe_layers = layers
        self.layers = nn.Sequential(*layers)

        # compute output dimension for classifier
        with torch.no_grad():
            # shape: Batch, Channels, Duration
            dummy_input = torch.zeros(1, 1, self.frame_duration_n)
            dummy_output = self.layers(dummy_input)
            output_shape = dummy_output.shape[1:]
            classifier_input_len = output_shape[0] * output_shape[1]

        self.classifier = nn.Linear(classifier_input_len, self.num_classes)

    def forward(self, input):
        if input.ndim == 2:
            # insert channel dimension
            input = input.unsqueeze(1)

        # apply successive 1D convolutional layers
        input = self.layers(input)

        # convert to logits
        input = input.transpose(1, 2)
        input = input.flatten(start_dim=1)
        logits = self.classifier(input)
        return logits

    def num_frames_in_samples(self, samples: torch.Tensor) -> int:
        if self.center:
            samples = F.pad(
                samples,
                [self.frame_duration_n//2, self.frame_duration_n//2],
                mode='constant', value=0)
        batch_size, duration_n = samples.shape[:2]
        return batch_size * (1 + int((duration_n - self.frame_duration_n)
                                     / self.hop_length_n))

    def get_frames(self, audio, sr, center=True, step_size=10):
        """Split the provided batch of audio samples into frames

        Parameters
        ----------
        audio : np.ndarray [shape=(B, N,) or (B, N, C)]
            The audio samples. Multichannel audio will be downmixed.
        sr : int
            Sample rate of the audio samples. The audio will be resampled if
            the sample rate is not 16 kHz, which is expected by the model.
        center : boolean
            - If `True` (default), the signal `audio` is padded so that frame
            `D[:, t]` is centered at `audio[t * hop_length]`.
            - If `False`, then `D[:, t]` begins at `audio[t * hop_length]`
        step_size : int
            The step size in seconds for running pitch estimation.

        Returns
        -------
        frames : np.ndarray [shape=(T, 1024)]
        """
        if len(audio.shape) == 3:
            audio = audio.mean(-1)  # make mono
        audio = audio.float()
        if sr != self.fs_hz:
            raise ValueError(f"Unexpected sampling frequency {sr}, "
                             f"expected {self.fs_hz}")
        # if sr != model_srate:
        #     # resample audio if necessary
        #     from resampy import resample
        #     audio = resample(audio, sr, model_srate)

        if self.center:
            audio = F.pad(
                audio,
                [self.frame_duration_n//2, self.frame_duration_n//2],
                mode='constant', value=0)

        # must clone to remove shared memory after unfolding with overlap
        # otherwise the normalization goes wrong
        frames = audio.unfold(1, self.frame_duration_n, self.hop_length_n
                              ).clone()
        frames = frames.flatten(0, 1)

        # normalize each frame -- this is expected by the model
        frames -= frames.mean(dim=-1, keepdim=True)
        frames /= (frames.std(dim=-1, keepdim=True) + 1e-6)
        return frames

    def load_keras_weights(self, weights_path: str):
        from h5py import File
        with File(weights_path, 'r') as f:
            # load the weights for all CREPE layers
            # ignore error on next line since
            # nn.Sequential.__iter__ is not detected by mypy
            for layer in self.layers:  # type: ignore
                layer.load_keras_weights(f)

            # load the weights for the final classifier layer
            self.classifier.weight.data = _get_keras_weights(
                f, 'classifier', 'kernel').t()
            self.classifier.bias.data = _get_keras_weights(
                f, 'classifier', 'bias')

    @torch.no_grad()
    def predict(self, frames: np.ndarray, **kwargs
                ) -> np.ndarray:
        """Return predicted activation for the provided frames.

        Provided for duck-typing with the TensorFlow backend.

        Arguments:
            frames (np.ndarray):
                an array of frames with shape
                (N_frames, self.frame_duration_n)
                or (N_frames, 1, self.frame_duration_n)
        Return:
            np.ndarray of shape (N_frames, 360):
                logits for each frame over the whole cents distribution
        """
        self.eval()

        device = torch.device('cuda:0' if torch.cuda.is_available()
                              else 'cpu')
        self.to(device)
        frames = torch.as_tensor(frames).to(device)

        parallel_model = torch.nn.DataParallel(self)
        logits = parallel_model(frames).cpu().numpy()
        activation = torch.sigmoid(logits)
        return activation

    def make_targets(self, samples: torch.Tensor,
                     midi_pitches: torch.IntTensor,
                     ) -> torch.Tensor:
        import librosa
        targets_hz_per_sample = torch.as_tensor(
            librosa.midi_to_hz(midi_pitches)).float()
        n_frames_in_sample = self.num_frames_in_samples(samples[0][None])
        return targets_hz_per_sample.repeat_interleave(n_frames_in_sample)


def build_and_load_model(model_capacity: str):
    """
    Build the CNN model and load the weights

    Parameters
    ----------
    model_capacity : 'tiny', 'small', 'medium', 'large', or 'full'
        String specifying the model capacity, which determines the model's
        capacity multiplier to 4 (tiny), 8 (small), 16 (medium), 24 (large),
        or 32 (full). 'full' uses the model size specified in the paper,
        and the others use a reduced number of filters in each convolutional
        layer, resulting in a smaller model that is faster to evaluate at the
        cost of slightly reduced pitch estimation accuracy.

    Returns
    -------
    model : tensorflow.keras.models.Model
        The pre-trained keras model loaded in memory
    """
    if models[model_capacity] is None:
        model = CREPE(model_capacity)

        package_dir = os.path.dirname(os.path.realpath(__file__))
        filename = "model-{}.h5".format(model_capacity)
        model.load_keras_weights(os.path.join(package_dir, filename))

        device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        model = model.to(device)

        models[model_capacity] = model

    return models[model_capacity]


if __name__ == '__main__':
    import torch
    device = 'cpu'
    model = build_and_load_model('tiny')
    model(torch.randn(size=(1, 1, 1024)).to('cuda'))
