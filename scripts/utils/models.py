import os
import shutil
import timeit

import numpy as np
import soundfile as sf
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import DataParallel
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader
from torch.optim import Adam, lr_scheduler

from configs import exp_conf
from utils.utils import getLogger, countFrames, lossMask, lossLog, wavNormalize
from utils.pipeline_modules import NetFeeder, Resynthesizer
from utils.data_utils import AudioLoader
from utils.criteria import LossFunction
from utils.ginformer import AIA_Ginformer

def he_init(m):
    if isinstance(m, (torch.nn.Conv2d, torch.nn.Linear)):
        torch.nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
        if m.bias is not None:
            torch.nn.init.constant_(m.bias, 0)

# Network definitions (from Network.py)
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
        return X + gated_positional_encoding

class FACLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, strategy='trained-zero', F=None, sample_rate=16000):
        super(FACLayer, self).__init__()
        reduction = 1 if in_channels in [1, 2] else (8 if in_channels == 16 else 16)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.gated_positional_encoding = GatedPositionalEncoding(in_channels, strategy, F, reduction, sample_rate)

    def forward(self, X):
        return self.conv(self.gated_positional_encoding(X))



# Model definitions (from Model.py)
gpu_ids        = '0'
tr_list        =    # add directory address for training list here
cv_file        =    # add directory address for validation here
tt_list        =    # add directory address for testing list here
model_file     =    # add directory address for model_file here
est_path       =    # add directory address for estimated file here
ckpt_dir       =    # add directory address for ckpt_dir here
unit           =   'utt'
logging_period =   2
time_log       =   ''
batch_size     =   10
buffer_size    =   24
segment_size   =   4.0
segment_shift  =   1.0
lr             =   0.001
lr_decay_factor=   2
lr_decay_period=   0.98
clip_norm      =   1.0
max_n_epochs   =   100
loss_log       =   'loss.txt'
resume_model   =   ''
write_ideal    =   'False'

class CheckPoint(object):
    def __init__(self, ckpt_info=None, net_state_dict=None, optim_state_dict=None):
        self.ckpt_info = ckpt_info
        self.net_state_dict = net_state_dict
        self.optim_state_dict = optim_state_dict

    def save(self, filename, is_best, best_model=None):
        torch.save(self, filename)
        if is_best:
            shutil.copyfile(filename, best_model)

    def load(self, filename, device):
        if not os.path.isfile(filename):
            raise FileNotFoundError('No checkpoint found at {}'.format(filename))
        ckpt = torch.load(filename, map_location=device)
        self.ckpt_info = ckpt.ckpt_info
        self.net_state_dict = ckpt.net_state_dict
        self.optim_state_dict = ckpt.optim_state_dict

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
        e1 = self.elu(self.bn1(self.conv1(x)[:,:,:-1,:].contiguous()))
        e2 = self.elu(self.bn2(self.conv2(e1)[:,:,:-1,:].contiguous()))
        e3 = self.elu(self.bn3(self.conv3(e2)[:,:,:-1,:].contiguous()))
        e4 = self.elu(self.bn4(self.conv4(e3)[:,:,:-1,:].contiguous()))
        e5 = self.elu(self.bn5(self.conv5(e4)[:,:,:-1,:].contiguous()))
        out = self.m1(e5)
        out = torch.cat([out, e5], dim=1)
        d5 = self.bn5_t(F.pad(self.de5(out), [0,0,1,0]).contiguous())
        d5 = self.elu(d5)
        d4 = self.elu(self.bn4_t(F.pad(self.de4(torch.cat([d5, e4], dim=1)), [0,0,1,0]).contiguous()))
        d3 = self.elu(self.bn3_t(F.pad(self.de3(torch.cat([d4, e3], dim=1)), [0,0,1,0]).contiguous()))
        d2 = self.elu(self.bn2_t(F.pad(self.de2(torch.cat([d3, e2], dim=1)), [0,0,1,0]).contiguous()))
        d1 = self.bn1_t(F.pad(self.de1(torch.cat([d2, e1], dim=1)), [0,0,1,0]).contiguous())
        return d1

class Model(object):
    def __init__(self, tr_list, ckpt_dir, cv_file, unit):
        self.in_norm = exp_conf['in_norm']
        self.sample_rate = exp_conf['sample_rate']
        self.win_len = exp_conf['win_len']
        self.hop_len = exp_conf['hop_len']
        self.win_size = int(self.win_len * self.sample_rate)
        self.hop_size = int(self.hop_len * self.sample_rate)

    def train(self):
        with open(tr_list, 'r') as f:
            self.tr_list = [line.strip() for line in f.readlines()]
        self.tr_size = len(self.tr_list)
        self.cv_file = cv_file
        self.ckpt_dir = ckpt_dir
        self.logging_period = logging_period
        self.resume_model = resume_model
        self.time_log = time_log
        self.lr = lr
        self.lr_decay_factor = lr_decay_factor
        self.lr_decay_period = lr_decay_period
        self.clip_norm = clip_norm
        self.max_n_epochs = max_n_epochs
        self.batch_size = batch_size
        self.buffer_size = buffer_size
        self.loss_log = loss_log
        self.unit = unit
        self.segment_size = segment_size
        self.segment_shift = segment_shift
        self.gpu_ids = tuple(map(int, gpu_ids.split(',')))
        if len(self.gpu_ids) == 1 and self.gpu_ids[0] == -1:
            self.device = torch.device('cpu')
        else:
            self.device = torch.device('cuda:{}'.format(self.gpu_ids[0]))
        if not os.path.isdir(self.ckpt_dir):
            os.makedirs(self.ckpt_dir)
        logger = getLogger(os.path.join(self.ckpt_dir, 'train.log'), log_file=True)
        tr_loader = AudioLoader(self.tr_list, self.sample_rate, self.unit, self.segment_size, self.segment_shift, self.batch_size, self.buffer_size, self.in_norm, mode='train')
        cv_loader = AudioLoader(self.cv_file, self.sample_rate, unit='utt', segment_size=None, segment_shift=None, batch_size=1, buffer_size=10, in_norm=self.in_norm, mode='eval')
        net = Net()
        net.apply(he_init)
        logger.info('Model summary:\n{}'.format(net))
        net = net.to(self.device)
        if len(self.gpu_ids) > 1:
            net = DataParallel(net, device_ids=self.gpu_ids)
        feeder = NetFeeder(self.device, self.win_size, self.hop_size)
        criterion = LossFunction(device=self.device, win_size=self.win_size, hop_size=self.hop_size)
        optimizer = Adam(net.parameters(), lr=self.lr, amsgrad=False)
        scheduler = lr_scheduler.StepLR(optimizer, step_size=self.lr_decay_period, gamma=self.lr_decay_factor)
        if self.resume_model:
            logger.info('Resuming model from {}'.format(self.resume_model))
            ckpt = CheckPoint()
            ckpt.load(self.resume_model, self.device)
            state_dict = {}
            for key in ckpt.net_state_dict:
                if len(self.gpu_ids) > 1:
                    state_dict['module.'+key] = ckpt.net_state_dict[key]
                else:
                    state_dict[key] = ckpt.net_state_dict[key]
            net.load_state_dict(state_dict)
            optimizer.load_state_dict(ckpt.optim_state_dict)
            ckpt_info = ckpt.ckpt_info
            logger.info('model info: epoch {}, iter {}, cv_loss - {:.4f}\n'.format(ckpt.ckpt_info['cur_epoch']+1, ckpt.ckpt_info['cur_iter']+1, ckpt.ckpt_info['cv_loss']))
        else:
            logger.info('Training from scratch...\n')
            ckpt_info = {'cur_epoch': 0, 'cur_iter': 0, 'tr_loss': None, 'cv_loss': None, 'best_loss': float('inf')}
        start_iter = 0
        while ckpt_info['cur_epoch'] < self.max_n_epochs:
            accu_tr_loss = 0.
            accu_n_frames = 0
            net.train()
            for n_iter, egs in enumerate(tr_loader):
                n_iter += start_iter
                mix = egs['mix']
                sph = egs['sph']
                n_samples = egs['n_samples']
                mix = mix.to(self.device)
                sph = sph.to(self.device)
                n_samples = n_samples.to(self.device)
                n_frames = countFrames(n_samples, self.win_size, self.hop_size)
                start_time = timeit.default_timer()
                feat, lbl = feeder(mix, sph)
                loss_mask = lossMask(shape=lbl.shape, n_frames=n_frames, device=self.device)
                optimizer.zero_grad()
                with torch.enable_grad():
                    est = net(feat)
                loss = criterion(est, lbl, loss_mask, n_frames, mix, n_samples)
                loss.backward()
                if self.clip_norm >= 0.0:
                    clip_grad_norm_(net.parameters(), self.clip_norm)
                optimizer.step()
                running_loss = loss.data.item()
                accu_tr_loss += running_loss * sum(n_frames)
                accu_n_frames += sum(n_frames)
                end_time = timeit.default_timer()
                batch_time = end_time - start_time
                if self.time_log:
                    with open(self.time_log, 'a+') as f:
                        print('Epoch [{}/{}], Iter [{}], tr_loss = {:.4f} / {:.4f}, batch_time (s) = {:.4f}'.format(ckpt_info['cur_epoch']+1, self.max_n_epochs, n_iter, running_loss, accu_tr_loss / accu_n_frames, batch_time), file=f)
                        f.flush()
                else:
                    print('Epoch [{}/{}], Iter [{}], tr_loss = {:.4f} / {:.4f}, batch_time (s) = {:.4f}'.format(ckpt_info['cur_epoch']+1, self.max_n_epochs, n_iter, running_loss, accu_tr_loss / accu_n_frames, batch_time), flush=True)
                if (n_iter + 1) % self.logging_period == 0:
                    avg_tr_loss = accu_tr_loss / accu_n_frames
                    avg_cv_loss = self.validate(net, cv_loader, criterion, feeder)
                    net.train()
                    ckpt_info['cur_iter'] = n_iter
                    is_best = True if avg_cv_loss < ckpt_info['best_loss'] else False
                    ckpt_info['best_loss'] = avg_cv_loss if is_best else ckpt_info['best_loss']
                    latest_model = 'latest.pt'
                    best_model = 'best.pt'
                    ckpt_info['tr_loss'] = avg_tr_loss
                    ckpt_info['cv_loss'] = avg_cv_loss
                    if len(self.gpu_ids) > 1:
                        ckpt = CheckPoint(ckpt_info, net.module.state_dict(), optimizer.state_dict())
                    else:
                        ckpt = CheckPoint(ckpt_info, net.state_dict(), optimizer.state_dict())
                    logger.info('Saving checkpoint into {}'.format(os.path.join(self.ckpt_dir, latest_model)))
                    if is_best:
                        logger.info('Saving checkpoint into {}'.format(os.path.join(self.ckpt_dir, best_model)))
                    logger.info('Epoch [{}/{}], ( tr_loss: {:.4f} | cv_loss: {:.4f} )\n'.format(ckpt_info['cur_epoch']+1, self.max_n_epochs, avg_tr_loss, avg_cv_loss))
                    model_path = os.path.join(self.ckpt_dir, 'models')
                    if not os.path.isdir(model_path):
                        os.makedirs(model_path)
                    ckpt.save(os.path.join(model_path, latest_model), is_best, os.path.join(model_path, best_model))
                    lossLog(os.path.join(self.ckpt_dir, self.loss_log), ckpt, self.logging_period)
                    accu_tr_loss = 0.
                    accu_n_frames = 0
                    if n_iter + 1 == self.tr_size // self.batch_size:
                        start_iter = 0
                        ckpt_info['cur_iter'] = 0
                        break
            ckpt_info['cur_epoch'] += 1
        return

    def validate(self, net, cv_loader, criterion, feeder):
        accu_cv_loss = 0.
        accu_n_frames = 0
        if len(self.gpu_ids) > 1:
            net = net.module
        net.eval()
        for k, egs in enumerate(cv_loader):
            mix = egs['mix']
            sph = egs['sph']
            n_samples = egs['n_samples']
            mix = mix.to(self.device)
            sph = sph.to(self.device)
            n_samples = n_samples.to(self.device)
            n_frames = countFrames(n_samples, self.win_size, self.hop_size)
            feat, lbl = feeder(mix, sph)
            with torch.no_grad():
                loss_mask = lossMask(shape=lbl.shape, n_frames=n_frames, device=self.device)
                est = net(feat)
                loss = criterion(est, lbl, loss_mask, n_frames, mix, n_samples)
            accu_cv_loss += loss.data.item() * sum(n_frames)
            accu_n_frames += sum(n_frames)
        return accu_cv_loss / accu_n_frames

    def test(self):
        with open(tt_list, 'r') as f:
            self.tt_list = [line.strip() for line in f.readlines()]
        self.model_file = model_file
        self.ckpt_dir = ckpt_dir
        self.est_path = est_path
        self.write_ideal = write_ideal
        self.gpu_ids = tuple(map(int, gpu_ids.split(',')))
        if len(self.gpu_ids) == 1 and self.gpu_ids[0] == -1:
            self.device = torch.device('cpu')
        else:
            self.device = torch.device('cuda:{}'.format(self.gpu_ids[0]))
        if not os.path.isdir(self.ckpt_dir):
            os.makedirs(self.ckpt_dir)
        logger = getLogger(os.path.join(self.ckpt_dir, 'test.log'), log_file=True)
        net = Net()
        logger.info('Model summary:\n{}'.format(net))
        net = net.to(self.device)
        criterion = LossFunction(device=self.device, win_size=self.win_size, hop_size=self.hop_size)
        feeder = NetFeeder(self.device, self.win_size, self.hop_size)
        resynthesizer = Resynthesizer(self.device, self.win_size, self.hop_size)
        logger.info('Loading model from {}'.format(self.model_file))
        ckpt = CheckPoint()
        ckpt.load(self.model_file, self.device)
        net.load_state_dict(ckpt.net_state_dict)
        logger.info('model info: epoch {}, iter {}, cv_loss - {:.4f}\n'.format(ckpt.ckpt_info['cur_epoch']+1, ckpt.ckpt_info['cur_iter']+1, ckpt.ckpt_info['cv_loss']))
        net.eval()
        for i in range(len(self.tt_list)):
            tt_loader = AudioLoader(self.tt_list[i], self.sample_rate, unit='utt', segment_size=None, segment_shift=None, batch_size=1, buffer_size=10, in_norm=self.in_norm, mode='eval')
            logger.info('[{}/{}] Estimating on {}'.format(i+1, len(self.tt_list), self.tt_list[i]))
            est_subdir = os.path.join(self.est_path, self.tt_list[i].split('/')[-1].replace('.ex', ''))
            if not os.path.isdir(est_subdir):
                os.makedirs(est_subdir)
            accu_tt_loss = 0.
            accu_n_frames = 0        
            for k, egs in enumerate(tt_loader):
                mix = egs['mix']
                sph = egs['sph']
                n_samples = egs['n_samples']
                n_frames = countFrames(n_samples, self.win_size, self.hop_size)
                mix = mix.to(self.device)
                sph = sph.to(self.device)
                feat, lbl = feeder(mix, sph)
                with torch.no_grad():
                    loss_mask = lossMask(shape=lbl.shape, n_frames=n_frames, device=self.device)
                    est = net(feat)
                    loss = criterion(est, lbl, loss_mask, n_frames, mix, n_samples)
                accu_tt_loss += loss.data.item() * sum(n_frames)
                accu_n_frames += sum(n_frames)
                sph_idl = resynthesizer(lbl, mix)
                sph_est = resynthesizer(est, mix)
                mix = mix[0].cpu().numpy() 
                sph = sph[0].cpu().numpy()
                sph_est = sph_est[0].cpu().numpy()
                sph_idl = sph_idl[0].cpu().numpy()
                mix, sph, sph_est, sph_idl = wavNormalize(mix, sph, sph_est, sph_idl)
                sf.write(os.path.join(est_subdir, '{}_mix.wav'.format(k)), mix, self.sample_rate)
                sf.write(os.path.join(est_subdir, '{}_sph.wav'.format(k)), sph, self.sample_rate)
                sf.write(os.path.join(est_subdir, '{}_sph_est.wav'.format(k)), sph_est, self.sample_rate)
                if self.write_ideal:
                    sf.write(os.path.join(est_subdir, '{}_sph_idl.wav'.format(k)), sph_idl, self.sample_rate)
            logger.info('loss: {:.4f}'.format(accu_tt_loss / accu_n_frames))
        return
