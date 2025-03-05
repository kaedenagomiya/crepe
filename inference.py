import os
import sys

import torch
import torchaudio
import numpy as np
import matplotlib.pyplot as plt

from crepe.model import CREPE

wav_path = './my_humm.wav'
pretrained_path = './crepe/pretrained/crepe_tiny_best.pt'

#torch.cuda.is_available()
#torch.backends.mps.is_available()
device = torch.device('cpu') # cuda, mps, cpu
crepe = CREPE(model_capacity='tiny', pretrained_path=pretrained_path).to(device)

print(f"crepe model_capacity: ", crepe.model_capacity)

waveform, samplerate = torchaudio.load(wav_path)

time, frequency, confidence, activation = crepe.predict(
    audio=waveform,
    samplerate=samplerate
)

print("\nProcess..")
print(f"activation Shape: {activation.shape}")
print(f"confidence Shape: {frequency.shape}")
print(f"frequency Shape:  {frequency.shape}")
print(f"time Shape:       {frequency.shape}")

salience = activation.flip(1)
salience_transposed = salience.transpose(0, 1)  # Transpose the axes
plt.figure(figsize=(10, 6))  # Adjust the figure size
plt.imshow(salience_transposed.detach().numpy(), cmap='inferno', aspect='auto')
plt.colorbar(label='Activation')  # Add a color bar for reference
plt.title('Salience Map')
plt.xlabel('Sample Index')  # Adjusted based on transposition
plt.ylabel('Feature Dimension')  # Adjusted based on transposition
plt.ylim(350, 300)  # Set the y-axis range from 350 to 250
plt.show()
plt.savefig('map.png')

print('fin')
