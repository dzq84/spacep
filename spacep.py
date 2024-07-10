import h5py
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, random_split
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter  
# 设置TensorBoard
writer = SummaryWriter('runs/kwave_experiment')

class KWaveDataset(Dataset):
    def __init__(self, h5_file, num_samples, sequence_length=128):
        self.h5_file = h5_file
        self.num_samples = num_samples
        self.sequence_length = sequence_length
        with h5py.File(h5_file, 'r') as hf:
            self.t = hf['p'].shape[1]

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        with h5py.File(self.h5_file, 'r') as hf:
            start_time = np.random.randint(0, self.t - self.sequence_length)
            p_seq = hf['p'][:, start_time:start_time + self.sequence_length, :]
            p_seq = p_seq.reshape(1, self.sequence_length, 256, 256)  # (1, 128, 256, 256)
            p_seq = np.transpose(p_seq, (0, 2, 3, 1))  # (1, 256, 256, 128)
            p_seq_tensor = torch.tensor(p_seq, dtype=torch.float32).permute(3, 0, 1, 2)  # (128*1*256*256) (T*C*H*W)
            resampled_seq = F.interpolate(p_seq_tensor, size=(128, 128), mode='bilinear', align_corners=False)
            resampled_seq = resampled_seq.permute(1, 2, 3, 0)  # (1*128*128*128) (C*H*W*T)

            # 归一化处理
            p_min = resampled_seq.min()
            p_max = resampled_seq.max()
            normalized_seq = (resampled_seq - p_min) / (p_max - p_min)
            
            # 返回归一化后的块以及 p_min 和 p_max
            return normalized_seq, p_min, p_max

h5_file = '/datasets/paws/temp_hdf5/2024-06-26-19-35-16_kwave_output.h5'
num_samples = 1000 # 指定需要的样本数
dataset = KWaveDataset(h5_file, num_samples)

# 将数据集分成训练集和测试集
train_size = int(0.8 * num_samples)
test_size = num_samples - train_size
train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False)

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

def compute_snr(original, reconstructed):
    signal_power = torch.sum(original ** 2)
    noise_power = torch.sum((original - reconstructed) ** 2)
    snr = 10 * torch.log10(signal_power / noise_power)
    return snr.item()

def train_one_epoch(epoch, model, dataloader, device, optimizer, criterion, writer):
    model.train()
    running_loss = 0.0
    for i, (data, p_min, p_max) in enumerate(tqdm(dataloader, desc=f"Epoch {epoch + 1} Training")):
        data = data.to(device)
        optimizer.zero_grad()
        outputs = model(data)
        loss = criterion(outputs, data)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        writer.add_scalar('Loss/train_step', loss.item(), epoch * len(dataloader) + i)
    average_loss = running_loss / len(dataloader)
    print(f"Training Loss: {average_loss:.4f}")
    return average_loss

def validate(model, dataloader, device, criterion, epoch, best_snr, writer):
    model.eval()
    test_loss = 0.0
    snr_list = []
    with torch.no_grad():
        for i, (data, p_min, p_max) in enumerate(tqdm(dataloader, desc="Validating")):
            data = data.to(device)
            outputs = model(data)
            loss = criterion(outputs, data)
            test_loss += loss.item()
            snr = compute_snr(data, outputs)
            snr_list.append(snr)
            if i == 0:  # Only log the first batch's images
                writer.add_images('Images/original', data[:, :, :, :, 0], epoch, dataformats='NCHW')
                writer.add_images('Images/reconstructed', outputs[:, :, :, :, 0], epoch, dataformats='NCHW')
    average_test_loss = test_loss / len(dataloader)
    average_snr = np.mean(snr_list)
    print(f"Validation Loss: {average_test_loss:.4f}, Average SNR: {average_snr:.4f} dB")
    if average_snr > best_snr:
        best_snr = average_snr
        torch.save(model.state_dict(), 'best_model.pth')
        print("Saved better model")
    return average_test_loss, best_snr

# Setup
device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
model = Conv3DNet().to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
writer = SummaryWriter('runs/kwave_experiment')
best_snr = -float('inf')

# Training loop
num_epochs = 1000
for epoch in range(num_epochs):
    train_one_epoch(epoch, model, train_loader, device, optimizer, criterion, writer)
    _, best_snr = validate(model, test_loader, device, criterion, epoch, best_snr, writer)

writer.close()