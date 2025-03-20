import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.ginformer import AIA_Ginformer

class AdaptiveFrequencyBandPositionalEncoding(nn.Module):
    def __init__(self, strategy='trained-zero', F=301, sample_rate=16000):
        super(AdaptiveFrequencyBandPositionalEncoding, self).__init__()
        self.strategy = strategy
        self.sample_rate = sample_rate
        self.F = F

        self.low_freq_range = (0, 300)
        self.mid_freq_range = (300, 3400)
        self.high_freq_range = (3400, sample_rate / 2)

        low_bins = int(F * (self.low_freq_range[1] / (self.sample_rate / 2)))
        mid_bins = int(F * (self.mid_freq_range[1] / (self.sample_rate / 2))) - low_bins
        high_bins = F - (low_bins + mid_bins)

        self.low_bins = low_bins
        self.mid_bins = mid_bins
        self.high_bins = high_bins

        if self.strategy == 'fixed-sin':
            P_low = torch.sin(torch.pi / 2 * torch.arange(low_bins).float() / F)
            P_mid = torch.sin(torch.pi / 2 * torch.arange(mid_bins).float() / F)
            P_high = torch.sin(torch.pi / 2 * torch.arange(high_bins).float() / F)
        elif self.strategy == 'trained-zero':
            P_low = torch.zeros(low_bins)
            P_mid = torch.zeros(mid_bins)
            P_high = torch.zeros(high_bins)
        else:
            raise ValueError(f"Unknown strategy '{self.strategy}'")

        self.P_freq_low = nn.Parameter(P_low, requires_grad=True)
        self.P_freq_mid = nn.Parameter(P_mid, requires_grad=True)
        self.P_freq_high = nn.Parameter(P_high, requires_grad=True)

        low_mask = torch.zeros(F, dtype=torch.bool)
        low_mask[:low_bins] = True
        mid_mask = torch.zeros(F, dtype=torch.bool)
        mid_mask[low_bins:low_bins + mid_bins] = True
        high_mask = torch.zeros(F, dtype=torch.bool)
        high_mask[low_bins + mid_bins:] = True

        self.register_buffer('low_mask', low_mask)
        self.register_buffer('mid_mask', mid_mask)
        self.register_buffer('high_mask', high_mask)

    def forward(self, X):
        batch_size, C, F, T = X.size()

        if F != self.F:
            low_bins = int(F * (self.low_freq_range[1] / (self.sample_rate / 2)))
            mid_bins = int(F * (self.mid_freq_range[1] / (self.sample_rate / 2))) - low_bins
            high_bins = F - (low_bins + mid_bins)


            orig_low = self.P_freq_low.shape[0]
            orig_mid = self.P_freq_mid.shape[0]
            orig_high = self.P_freq_high.shape[0]

            use_low = min(low_bins, orig_low)
            use_mid = min(mid_bins, orig_mid)
            use_high = min(high_bins, orig_high)

            P_freq_adaptive = torch.zeros(batch_size, C, F, T, device=X.device)

            low_mask = torch.zeros(F, dtype=torch.bool, device=X.device)
            low_mask[:use_low] = True
            mid_mask = torch.zeros(F, dtype=torch.bool, device=X.device)
            mid_mask[use_low:use_low + use_mid] = True
            high_mask = torch.zeros(F, dtype=torch.bool, device=X.device)
            start_high = use_low + use_mid
            if start_high < F:
                high_mask[start_high:start_high + use_high] = True

            P_low = self.P_freq_low[:use_low].view(1, 1, use_low, 1).expand(batch_size, C, -1, T)
            P_mid = self.P_freq_mid[:use_mid].view(1, 1, use_mid, 1).expand(batch_size, C, -1, T)
            P_high = self.P_freq_high[:use_high].view(1, 1, use_high, 1).expand(batch_size, C, -1, T)

            if use_low > 0:
                P_freq_adaptive[:, :, low_mask, :] = P_low
            if use_mid > 0:
                P_freq_adaptive[:, :, mid_mask, :] = P_mid
            if use_high > 0:
                P_freq_adaptive[:, :, high_mask, :] = P_high

            return P_freq_adaptive
        else:
            P_freq_adaptive = torch.zeros(batch_size, C, F, T, device=X.device)

            P_low = self.P_freq_low.view(1, 1, self.low_bins, 1).expand(batch_size, C, -1, T)
            P_mid = self.P_freq_mid.view(1, 1, self.mid_bins, 1).expand(batch_size, C, -1, T)
            P_high = self.P_freq_high.view(1, 1, self.high_bins, 1).expand(batch_size, C, -1, T)

            P_freq_adaptive[:, :, self.low_mask, :] = P_low
            P_freq_adaptive[:, :, self.mid_mask, :] = P_mid
            P_freq_adaptive[:, :, self.high_mask, :] = P_high

            return P_freq_adaptive

class GatedPositionalEncoding(nn.Module):
    def __init__(self, in_channels, strategy='trained-zero', F=301, reduction=8, sample_rate=16000):
        super(GatedPositionalEncoding, self).__init__()
        self.positional_encoding = AdaptiveFrequencyBandPositionalEncoding(strategy, F, sample_rate)
        
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(in_channels, in_channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels // reduction, in_channels, bias=False),
            nn.Sigmoid()
        )
        
        self.gate = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, X):
        P_freq = self.positional_encoding(X)
        se = self.se(X)
        se = se.view(X.size(0), X.size(1), 1, 1)
        gate = self.gate(X)
        gated_positional_encoding = gate * P_freq * se
        X_gated = X + gated_positional_encoding
        return X_gated

class FACLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, strategy='trained-zero', F=None, sample_rate=16000):
        super(FACLayer, self).__init__()
        if in_channels == 1 or in_channels == 2:
            reduction = 1
        elif in_channels == 16:
            reduction = 8
        else:
            reduction = 16
        
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.gated_positional_encoding = GatedPositionalEncoding(in_channels, strategy, F, reduction, sample_rate)

    def forward(self, X):
        X_gated = self.gated_positional_encoding(X)
        output = self.conv(X_gated) 
        return output



class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        
        self.conv1 = FACLayer(in_channels=2, out_channels=16, kernel_size=(2,3), stride=(1,2), padding=(1,0), F=F)
        self.conv2 = FACLayer(in_channels=16, out_channels=32, kernel_size=(2,3), stride=(1,2), padding=(1,0), F=F)
        self.conv3 = FACLayer(in_channels=32, out_channels=64, kernel_size=(2,3), stride=(1,2), padding=(1,0), F=F)
        self.conv4 = FACLayer(in_channels=64, out_channels=128, kernel_size=(2,3), stride=(1,2), padding=(1,0), F=F)
        self.conv5 = FACLayer(in_channels=128, out_channels=256, kernel_size=(2,3), stride=(1,2), padding=(1,0), F=F)

        self.m1 = AIA_Ginformer(input_size=256, output_size=256, seq_len=F)


        self.de5 = nn.ConvTranspose2d(in_channels=512, out_channels=128, kernel_size=(2,3), stride=(1,2), padding=(1,0))
        self.de4 = nn.ConvTranspose2d(in_channels=256, out_channels=64, kernel_size=(2,3), stride=(1,2), padding=(1,0))
        self.de3 = nn.ConvTranspose2d(in_channels=128, out_channels=32, kernel_size=(2,3), stride=(1,2), padding=(1,0))
        self.de2 = nn.ConvTranspose2d(in_channels=64, out_channels=16, kernel_size=(2,3), stride=(1,2), padding=(1,0), output_padding=(0,1))
        self.de1 = nn.ConvTranspose2d(in_channels=32, out_channels=2, kernel_size=(2,3), stride=(1,2), padding=(1,0))

        self.bn1 = nn.BatchNorm2d(16)
        self.bn2 = nn.BatchNorm2d(32) 
        self.bn3 = nn.BatchNorm2d(64)
        self.bn4 = nn.BatchNorm2d(128)
        self.bn5 = nn.BatchNorm2d(256)

        self.bn5_t = nn.BatchNorm2d(128) 
        self.bn4_t = nn.BatchNorm2d(64)
        self.bn3_t = nn.BatchNorm2d(32)
        self.bn2_t = nn.BatchNorm2d(16)
        self.bn1_t = nn.BatchNorm2d(2)

        self.elu = nn.ELU(inplace=True)
        self.softplus = nn.Softplus()

    def forward(self, x):
        out = x

        e1 = self.elu(self.bn1(self.conv1(out)[:,:,:-1,:].contiguous()))
        e2 = self.elu(self.bn2(self.conv2(e1)[:,:,:-1,:].contiguous()))
        e3 = self.elu(self.bn3(self.conv3(e2)[:,:,:-1,:].contiguous()))
        e4 = self.elu(self.bn4(self.conv4(e3)[:,:,:-1,:].contiguous()))
        e5 = self.elu(self.bn5(self.conv5(e4)[:,:,:-1,:].contiguous()))

        out = self.m1(e5)
        out = torch.cat([out, e5], dim=1)

        d5 = self.bn5_t(F.pad(self.de5(out), [0,0,1,0]).contiguous())
        d5 = self.elu(d5)

        out = torch.cat([d5, e4], dim=1)
        d4 = self.bn4_t(F.pad(self.de4(out), [0,0,1,0]).contiguous())
        d4 = self.elu(d4)

        out = torch.cat([d4, e3], dim=1)
        d3 = self.bn3_t(F.pad(self.de3(out), [0,0,1,0]).contiguous())
        d3 = self.elu(d3)

        out = torch.cat([d3, e2], dim=1)
        d2 = self.bn2_t(F.pad(self.de2(out), [0,0,1,0]).contiguous())
        d2 = self.elu(d2)

        out = torch.cat([d2, e1], dim=1)
        d1 = self.bn1_t(F.pad(self.de1(out), [0,0,1,0]).contiguous())
        return d1 
