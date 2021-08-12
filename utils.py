import librosa
import numpy as np
import pickle as pkl 
import re
from pathlib import Path
import torch
import torchvision
import torchaudio
from PIL import Image

SAMPLING_RATE = 8000
num_channels = 3
window_sizes = [25, 50, 100]
hop_sizes = [10, 25, 50]
eps = 1e-6
limits = ((-2, 2), (0.9, 1.2))

def extract_feature(file_path):
    clip, sr = librosa.load(file_path, sr=SAMPLING_RATE)
    specs = []
    for i in range(num_channels):
        window_length = int(round(window_sizes[i]*SAMPLING_RATE/1000))
        hop_length = int(round(hop_sizes[i]*SAMPLING_RATE/1000))            
        clip = torch.Tensor(clip)
        spec = torchaudio.transforms.MelSpectrogram(sample_rate=SAMPLING_RATE, n_fft=4410, win_length=window_length, hop_length=hop_length, n_mels=128)(clip)
        spec = spec.numpy()
        spec = np.log(spec+eps)
        spec = np.asarray(torchvision.transforms.Resize((128, 250))(Image.fromarray(spec)))
        specs.append(spec)
    new_entry = {}
    new_entry["audio"] = clip.numpy()
    new_entry["values"] = np.array(specs)
    return new_entry
