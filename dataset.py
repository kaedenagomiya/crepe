import os
import glob
import json
import torch
import torchaudio
from torch.utils.data import Dataset
from torch.utils.data import Dataset, ConcatDataset
from concurrent.futures import ThreadPoolExecutor, as_completed


class MIR1KDataset(Dataset):
    """
    For [MIR-1K Dataset](http://mirlab.org/dataset/public/MIR-1K.zip).

    Args:
        Dataset (Dataset): from torch.utils.data import Dataset
    """

    def __init__(self, root_dir):
        """
        Create an instance of the MIR-1K dataset.

        This class loads and prepares the data from the MIR-1K dataset,
        which consists of audio files with corresponding pitch labels.

        Args:
            root_dir (str): The root directory containing the MIR-1K dataset.
        """
        self.root_dir = root_dir
        # self.labels = [f.replace('.wav', '.txt') for f in self.files]
        self.files = sorted(glob.glob(os.path.join(
            self.root_dir+"/Wavfile", f"*.wav")))
        self.labels = sorted(glob.glob(os.path.join(
            self.root_dir+"/PitchLabel", f"*.pv")))

    def __len__(self):
        """
        Retrieve the total number of samples in the dataset.

        Returns:
            int: The total number of samples in the dataset.
        """
        return len(self.files)

    def __getitem__(self, idx):
        """
        Retrieve a single sample from the dataset,
        consisting of an audio file and its corresponding pitch label.

        Args:
            idx (int): The index of the sample to retrieve.

        Returns:
            tuple: A tuple containing the audio file and its corresponding pitch label,
                where the audio is a tensor representing the audio waveform, and the
                pitch label is a tensor containing the pitch values for each frame.

        Note:
            The returned audio tensor has shape (1, num_frames), where num_frames is the
            number of frames in the original audio file. 
            The returned pitch label tensor has shape (num_frames,), containing one pitch value per frame.
        """
        audio_path = os.path.abspath(self.files[idx])
        label_path = self.labels[idx]

        audio, sr = torchaudio.load(audio_path)

        with open(label_path, 'r') as f:
            labels = [float(line.strip()) for line in f.readlines()]

        labels = torch.tensor(labels)

        return audio[1, :], labels
