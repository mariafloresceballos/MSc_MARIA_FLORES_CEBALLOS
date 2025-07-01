from copy import deepcopy
import os
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from glob import glob
from torchsummary import summary
import random
import torch.nn.init as init
from tqdm import tqdm

datadir = r"C:\Users\maria\Desktop\GITHUB\Data\ECG_data_GAN" 
train_folder = os.path.join(datadir, "train")  


output_dir = "training_plots"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False  

set_seed(123)


def filter_class_3_segments(train_folder):
    class_3_signals = []
    class_3_labels = []
    class_3_bpms = []

    files = sorted(glob(os.path.join(train_folder, "*.npz")))

    for file in files:
        data = dict(np.load(file))
        signals = torch.tensor(data["signals"])
        labels = torch.tensor(data["labels"])
        bpms = torch.tensor(data["bpms"])

        mask = labels == 3
        filtered_signals = signals[mask]
        filtered_labels = labels[mask]
        filtered_bpms = bpms[mask]

        class_3_signals.append(filtered_signals)
        class_3_labels.append(filtered_labels)
        class_3_bpms.append(filtered_bpms)

    class_3_signals = torch.cat(class_3_signals, dim=0)
    class_3_labels = torch.cat(class_3_labels, dim=0)
    class_3_bpms = torch.cat(class_3_bpms, dim=0)

    return class_3_signals, class_3_labels, class_3_bpms


class EarlyStopper:
    def __init__(self, patience, min_delta):
        self.cnt = 0
        self.patience = patience
        self.min_delta = min_delta
        self.best_loss = torch.inf
        self.update_loss = torch.inf
        self.best_model = None

    def __call__(self, loss, state_dict):
        if loss < self.best_loss:
            self.best_model = deepcopy(state_dict)
            self.best_loss = loss

        if loss < self.update_loss:
            self.update_loss = loss - self.min_delta
            self.cnt = 0
            return False
        if self.patience == 0:
            return False
        else:
            self.cnt += 1
            if self.cnt == self.patience:
                return True
            else:
                return False

    def reset(self):
        self.cnt = 0
        self.best_loss = torch.inf
        self.update_loss = torch.inf
        self.best_model = None

    def save(self, filename):
        torch.save(self.best_model, filename)

class BiLSTMGenerator(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(BiLSTMGenerator, self).__init__()

        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers=1, bidirectional=True, batch_first=True)

        kernel_size = 15
        padding = (kernel_size - 1) // 2

        self.leaky_relu = nn.LeakyReLU(negative_slope=0.2)

        self.conv1 = nn.Conv1d(hidden_dim * 2, 128, kernel_size=kernel_size, padding=padding)
        self.bn1 = nn.BatchNorm1d(128)
        self.conv2 = nn.Conv1d(128, 64, kernel_size=kernel_size, padding=padding)
        self.bn2 = nn.BatchNorm1d(64)
        self.conv3 = nn.Conv1d(64, 32, kernel_size=kernel_size, padding=padding)
        self.bn3 = nn.BatchNorm1d(32)
        self.conv4 = nn.Conv1d(32, 16, kernel_size=kernel_size, padding=padding)
        self.bn4 = nn.BatchNorm1d(16)
        self.conv5 = nn.Conv1d(16, 1, kernel_size=kernel_size, padding=padding)

        self.upsample1 = nn.Upsample(scale_factor=2, mode='nearest')
        self.upsample2 = nn.Upsample(scale_factor=2, mode='nearest')

        self.dropout = nn.Dropout(p=0.3)

        self.initialize_weights()

    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm1d):
                init.ones_(m.weight)
                init.zeros_(m.bias)
            elif isinstance(m, nn.LSTM):
                for name, param in m.named_parameters():
                    if 'weight_ih' in name:
                        init.xavier_normal_(param)
                    elif 'weight_hh' in name:
                        init.orthogonal_(param)
                    elif 'bias' in name:
                        init.zeros_(param)

    def forward(self, z):

        z = z.view(z.size(0), -1, 1) 
        
        lstm_out, _ = self.lstm(z)

        x = self.leaky_relu(self.bn1(self.conv1(lstm_out.transpose(1, 2))))  
        
        x = self.dropout(x)
        x = self.leaky_relu(self.bn2(self.conv2(x)))
        
        x = self.upsample1(x) 
        
        x = self.leaky_relu(self.bn3(self.conv3(x)))
        
        x = self.leaky_relu(self.bn4(self.conv4(x)))
        
        x = self.upsample2(x)  

        output = self.conv5(x) 

        if output.size(2) == 2048:  
            last_value = output[:, :, -1:]  
            output = torch.cat([output, last_value], dim=2)
        return output.transpose(1, 2)  

class BiLSTMDiscriminator(nn.Module):
    def __init__(self):
        super(BiLSTMDiscriminator, self).__init__()

        kernel_size = 15
        padding = (kernel_size - 1) // 2

        self.conv1 = nn.Conv1d(1, 32, kernel_size=kernel_size, padding=padding)
        self.bn1 = nn.BatchNorm1d(32)

        self.conv2 = nn.Conv1d(32, 64, kernel_size=kernel_size, padding=padding)
        self.bn2 = nn.BatchNorm1d(64)

        self.conv3 = nn.Conv1d(64, 128, kernel_size=kernel_size, padding=padding)
        self.bn3 = nn.BatchNorm1d(128)

        self.conv4 = nn.Conv1d(128, 256, kernel_size=kernel_size, padding=padding)

        self.maxpool = nn.MaxPool1d(2)
        self.dropout = nn.Dropout(p=0.3)

        self.fc = nn.Linear(256 * 512, 1)  

        self.leaky_relu = nn.LeakyReLU(negative_slope=0.2)

        self.initialize_weights()

    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm1d):
                init.ones_(m.weight)
                init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                init.xavier_normal_(m.weight)
                if m.bias is not None:
                    init.zeros_(m.bias)

    def forward(self, x):
        if x.dim() == 3 and x.size(2) == 1:
            x = x.squeeze(-1) 

        if x.dim() == 2:
            x = x.unsqueeze(1) 

        x = self.leaky_relu(self.bn1(self.conv1(x)))  
        x = self.dropout(x)
        
        x = self.leaky_relu(self.bn2(self.conv2(x)))
        x = self.maxpool(x)
        
        x = self.leaky_relu(self.bn3(self.conv3(x)))
        
        x = self.leaky_relu(self.conv4(x))
        x = self.maxpool(x)

        x = x.view(x.size(0), -1)

        x = self.fc(x)
        
        return torch.sigmoid(x)
    

def generate_synthetic_samples(generator, num_samples, z_dim, batch_size, device, output_dir):
    generator.eval() 

    for batch_start in tqdm(range(0, num_samples, batch_size), desc="Generanting synthetic samples", unit="lote"):
        noise = torch.randn(batch_size, z_dim, 1).to(device) 
        fake_samples = generator(noise)  

        fake_samples_np = fake_samples.detach().cpu().numpy()

        batch_filename = os.path.join(output_dir, f"synthetic_batch_{batch_start // batch_size + 1}.npz")
        
        labels = np.ones(fake_samples_np.shape[0], dtype=np.uint8) * 3 
        bpm = np.ones(fake_samples_np.shape[0], dtype=np.float16) * 70.0  

        np.savez(batch_filename, signals=fake_samples_np.astype(np.float16), labels=labels, bpms=bpm)





device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

z_dim = 512  
input_dim = 1  
output_dim = 2049

signals, labels, bpms = filter_class_3_segments(train_folder)

signals = signals.float().to(device)
labels = labels.to(device)

batch_size = 32  
lr = 0.001 
hidden_dim = 64  

best_d_loss = float('inf')
best_g_loss = float('inf')
best_config = None

results = []

generator = BiLSTMGenerator(input_dim, hidden_dim)
discriminator = BiLSTMDiscriminator()

generator = generator.to(device)
discriminator = discriminator.to(device)

optimizer_g = torch.optim.Adam(generator.parameters(), lr=lr, betas=(0.5, 0.999))
optimizer_d = torch.optim.Adam(discriminator.parameters(), lr=lr, betas=(0.5, 0.999))

criterion = nn.BCEWithLogitsLoss() 

scheduler_g = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_g, T_max=100)
scheduler_d = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_d, T_max=100)

num_epochs = 1
synthetic_samples = []  

d_losses = [] 
g_losses = []  
real_accuracies = []  
fake_accuracies = []  
fake_accuracies_gen = [] 


for epoch in range(num_epochs):
    print(f"Epoch [{epoch+1}/{num_epochs}]")
    
    for i in tqdm(range(0, len(signals), batch_size), desc=f"Batchs en época {epoch+1}", unit="batch"):
        real_signals = signals[i:i + batch_size].to(device)
        real_labels = labels[i:i + batch_size].to(device)

        noise = torch.normal(mean=0, std=1, size=(batch_size, z_dim, 1)).to(device)  
        optimizer_d.zero_grad()

        real_preds = discriminator(real_signals)
        real_loss = criterion(real_preds, torch.ones_like(real_preds)).to(device)

        fake_signals = generator(noise)
        fake_preds = discriminator(fake_signals.detach())  
        fake_loss = criterion(fake_preds, torch.zeros_like(fake_preds)).to(device)

        real_accuracy = (real_preds.round() == 1).sum().float() / real_preds.size(0)  
        fake_accuracy = (fake_preds.round() == 0).sum().float() / fake_preds.size(0)  

        d_loss = real_loss + fake_loss
        d_loss.backward()
        optimizer_d.step()

        optimizer_g.zero_grad()

        fake_preds = discriminator(fake_signals)
        g_loss = criterion(fake_preds, torch.ones_like(fake_preds)).to(device)

        fake_accuracy_gen = (fake_preds.round() == 1).sum().float() / fake_preds.size(0)

        g_loss.backward()
        optimizer_g.step()

    scheduler_g.step()
    scheduler_d.step()

    d_losses.append(d_loss.item())
    g_losses.append(g_loss.item())
    real_accuracies.append(real_accuracy.item() * 100) 
    fake_accuracies.append(fake_accuracy.item() * 100)  
    fake_accuracies_gen.append(fake_accuracy_gen.item() * 100) 

    print(f"Discriminator Loss: {d_loss.item()}")
    print(f"Generator Loss: {g_loss.item()}")

results.append({
    'batch_size': batch_size,
    'learning_rate': lr,
    'hidden_dim': hidden_dim,
    'd_loss': d_loss.item(),
    'g_loss': g_loss.item(),
})

if g_loss.item() < best_g_loss and d_loss.item() < best_d_loss:
    best_d_loss = d_loss.item()
    best_g_loss = g_loss.item()
    best_config = {'batch_size': batch_size, 'learning_rate': lr, 'hidden_dim': hidden_dim}

config_str = f"bs_{batch_size}_lr_{lr}_hd_{hidden_dim}"

output_dir_config = os.path.join(output_dir, config_str)

if not os.path.exists(output_dir_config):
    os.makedirs(output_dir_config)

print(output_dir_config)

num_generated_samples = 1024 * 400


generate_synthetic_samples(generator, num_generated_samples, z_dim, batch_size=1024, device=device, output_dir=output_dir_config)
