import h5py
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from tqdm import tqdm

class KWaveDataset(Dataset):
    def __init__(self, h5_file, sequence_length=128):
        self.h5_file = h5_file
        self.sequence_length = sequence_length
        with h5py.File(h5_file, 'r') as hf:
            self.t = hf['p'].shape[1]

    def __len__(self):
        return self.t // self.sequence_length

    def __getitem__(self, idx):
        with h5py.File(self.h5_file, 'r') as hf:
            start_time = idx * self.sequence_length
            p_seq = hf['p'][:, start_time:start_time + self.sequence_length, :]
            p_seq = p_seq.reshape(1, self.sequence_length, 256, 256)
            p_seq = np.transpose(p_seq, (0, 2, 3, 1))  # (1, 256, 256, 128)
            p_seq_tensor = torch.tensor(p_seq, dtype=torch.float32).permute(3, 0, 1, 2)  # (128, 1, 256, 256)
            resampled_seq = F.interpolate(p_seq_tensor, size=(128, 128), mode='bilinear', align_corners=False)
            resampled_seq = resampled_seq.permute(1, 2, 3, 0)  # (1, 128, 128, 128)
            p_min = resampled_seq.min()
            p_max = resampled_seq.max()
            normalized_seq = (resampled_seq - p_min) / (p_max - p_min)
            return normalized_seq, p_min, p_max

h5_file = '/datasets/paws/temp_hdf5/2024-06-26-19-35-16_kwave_output.h5'
sequence_length = 128
dataset = KWaveDataset(h5_file, sequence_length)
dataloader = DataLoader(dataset, batch_size=32, shuffle=False)

import torch.nn as nn

class MultiHeadSelfAttention(nn.Module):
    def __init__(self, channels, num_heads):
        super(MultiHeadSelfAttention, self).__init__()
        assert channels % num_heads == 0, "channels must be divisible by num_heads"
        self.channels = channels
        self.num_heads = num_heads
        self.per_head_channels = channels // num_heads
        self.query = nn.Conv3d(channels, channels, kernel_size=1)
        self.key = nn.Conv3d(channels, channels, kernel_size=1)
        self.value = nn.Conv3d(channels, channels, kernel_size=1)
        self.fc_out = nn.Conv3d(channels, channels, kernel_size=1)

    def forward(self, x):
        batch_size = x.shape[0]
        q = self.query(x).view(batch_size, self.num_heads, self.per_head_channels, *x.shape[2:]).permute(0, 1, 3, 4, 5, 2)
        k = self.key(x).view(batch_size, self.num_heads, self.per_head_channels, *x.shape[2:]).permute(0, 1, 3, 4, 5, 2)
        v = self.value(x).view(batch_size, self.num_heads, self.per_head_channels, *x.shape[2:]).permute(0, 1, 3, 4, 5, 2)
        attention_scores = torch.einsum('bnxyzc,bmxyzc->bnmxy', q, k) / (self.per_head_channels ** 0.5)
        attention = F.softmax(attention_scores, dim=-1)
        attended_values = torch.einsum('bnmxy,bmxyzc->bnxyzc', attention, v)
        attended_values = attended_values.permute(0, 1, 5, 2, 3, 4).reshape(batch_size, self.channels, *x.shape[2:])
        return self.fc_out(attended_values)

class ResBlockWithSelfAttention(nn.Module):
    def __init__(self, channels):
        super(ResBlockWithSelfAttention, self).__init__()
        self.conv1 = nn.Conv3d(channels, channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm3d(channels)
        self.tanh = nn.Tanh()
        self.self_attention = MultiHeadSelfAttention(channels, num_heads=4)
        self.conv2 = nn.Conv3d(channels, channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm3d(channels)

    def forward(self, x):
        identity = x
        out = self.tanh(self.bn1(self.conv1(x)))
        out = self.self_attention(out)
        out = self.bn2(self.conv2(out))
        out += identity  # Add skip connection
        return self.tanh(out)

class Conv3DNet(nn.Module):
    def __init__(self):
        super(Conv3DNet, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv3d(1, 16, kernel_size=3, stride=2, padding=1),
            nn.Tanh(),
            nn.BatchNorm3d(16),
            ResBlockWithSelfAttention(16),
            nn.Conv3d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.Tanh(),
            nn.BatchNorm3d(32),
            ResBlockWithSelfAttention(32),
            nn.Conv3d(32, 4, kernel_size=3, stride=2, padding=1),
            nn.Tanh(),
            nn.BatchNorm3d(4),
            ResBlockWithSelfAttention(4)
        )
        
        self.decoder = nn.Sequential(
            ResBlockWithSelfAttention(4),
            nn.ConvTranspose3d(4, 32, kernel_size=2, stride=2),
            nn.Tanh(),
            nn.BatchNorm3d(32),
            ResBlockWithSelfAttention(32),
            nn.ConvTranspose3d(32, 16, kernel_size=2, stride=2),
            nn.Tanh(),
            nn.BatchNorm3d(16),
            ResBlockWithSelfAttention(16),
            nn.ConvTranspose3d(16, 1, kernel_size=2, stride=2),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = Conv3DNet().to(device)
model.load_state_dict(torch.load('model.pth', map_location=device))
model.eval()

def process_batch(model, batch, device):
    batch = batch.to(device)
    with torch.no_grad():
        compressed_feature = model.encoder(batch)
    return compressed_feature.cpu().numpy()

def compress_h5_file(model, input_h5_file, output_h5_file, sequence_length=128, batch_size=32):
    dataset = KWaveDataset(input_h5_file, sequence_length)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4, prefetch_factor=2)

    compressed_data = []
    p_min_list = []
    p_max_list = []

    for data, p_min, p_max in tqdm(dataloader, desc="Compressing batches"):
        compressed_feature = process_batch(model, data, device)
        compressed_data.append(compressed_feature)
        p_min_list.append(p_min.numpy())
        p_max_list.append(p_max.numpy())

    compressed_data = np.concatenate(compressed_data, axis=0)
    p_min_list = np.concatenate(p_min_list, axis=0).flatten()
    p_max_list = np.concatenate(p_max_list, axis=0).flatten()

    with h5py.File(output_h5_file, 'w') as hf:
        hf.create_dataset('compressed_features', data=compressed_data)
        hf.create_dataset('p_min', data=p_min_list)
        hf.create_dataset('p_max', data=p_max_list)

input_h5_file = '/datasets/paws/temp_hdf5/2024-06-26-19-35-16_kwave_output.h5'
output_h5_file = 'compressed_kwave_output.h5'

compress_h5_file(model, input_h5_file, output_h5_file)
