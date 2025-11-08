import os
import numpy as np
import torch
from torch.utils.data import Dataset

# Dataset Class
class SineWaveDataset(Dataset):
    def __init__(self, seq_length, num_samples):
        self.seq_length = seq_length
        self.data = []
        self.targets = []
        for _ in range(num_samples):
            phase = np.random.rand() * 2 * np.pi
            x = np.linspace(phase, phase + 4 * np.pi, seq_length + 1)
            y = np.sin(x)
            self.data.append(y[:-1])        # from starting to second last
            self.targets.append(y[1:])      # from second to last one shifted tokens
        self.data = torch.tensor(np.array(self.data), dtype=torch.float32).unsqueeze(-1)
        self.targets = torch.tensor(np.array(self.targets), dtype=torch.float32).unsqueeze(-1)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.targets[idx]

# Parameters
seq_length = 50
num_samples = 20000

# Generate Dataset
dataset = SineWaveDataset(seq_length, num_samples)

# Directory for saving the dataset
save_dir = "sinedataset"
os.makedirs(save_dir, exist_ok=True)

# Save the dataset
torch.save(dataset, os.path.join(save_dir, "sine_wave_dataset.pth"))

print(f"Dataset saved at {os.path.join(save_dir, 'sine_wave_dataset.pth')}")
