import h5py
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

# 数据集类
class CompressedKWaveDataset(Dataset):
    def __init__(self, h5_file):
        self.h5_file = h5_file
        with h5py.File(h5_file, 'r') as hf:
            self.compressed_features = hf['compressed_features'][:]
            self.p_min = hf['p_min'][:]
            self.p_max = hf['p_max'][:]

    def __len__(self):
        return len(self.compressed_features)

    def __getitem__(self, idx):
        compressed_feature = self.compressed_features[idx]
        p_min = self.p_min[idx]
        p_max = self.p_max[idx]
        return torch.tensor(compressed_feature), torch.tensor(p_min), torch.tensor(p_max)

# 模型定义
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

# 加载模型
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = Conv3DNet().to(device)
model.load_state_dict(torch.load('model.pth', map_location=device))
model.eval()

# 处理批次
def process_decompressed_batch(model, batch, device):
    batch = batch.to(device)  # 确保 batch 是 PyTorch 张量并移至设备
    with torch.no_grad():
        decompressed_data = model.decoder(batch)

    return decompressed_data.cpu()

# 解压缩 HDF5 文件
def decompress_h5_file(model, input_h5_file, output_h5_file, batch_size=32):
    dataset = CompressedKWaveDataset(input_h5_file)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4, prefetch_factor=2)

    decompressed_data = []

    for compressed_feature, p_min, p_max in tqdm(dataloader, desc="Decompressing batches"):
        decompressed_batch = process_decompressed_batch(model, compressed_feature, device)

        current_batch_size = decompressed_batch.shape[0]  # 动态获取当前批次的大小

        # 反归一化处理
        p_min = p_min.view(current_batch_size, 1, 1, 1, 1)  # 将 p_min 和 p_max 调整为合适的形状
        p_max = p_max.view(current_batch_size, 1, 1, 1, 1)
        denormalized_data = decompressed_batch * (p_max - p_min) + p_min

        # 将解压缩后的数据形状从 (B, C, H, W, T) 调整为 (1, B*T, H, W)
        B, C, H, W, T = denormalized_data.shape
        denormalized_data = denormalized_data.permute(1, 0, 4, 2, 3).reshape(1, B * T, H, W)

        # 将每个批次的数据添加到解压缩数据列表中
        decompressed_data.append(denormalized_data.numpy())

    decompressed_data = np.concatenate(decompressed_data, axis=1)  # 将所有批次的数据连接起来，形成 (1, N*B*T, H, W)
    
    # 将解压缩后的数据保存到新的 HDF5 文件中
    with h5py.File(output_h5_file, 'w') as hf:
        hf.create_dataset('decompressed_data', data=decompressed_data)

# 解压缩数据
input_h5_file = 'compressed_kwave_output.h5'
output_h5_file = 'decompressed_kwave_output.h5'

decompress_h5_file(model, input_h5_file, output_h5_file)
