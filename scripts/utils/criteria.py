import torch
import torch.nn.functional as F
from utils.pipeline_modules import Resynthesizer

def si_sdr(estimated, target, eps=1e-8):
    """
    Computes SI-SDR between estimated and target time-domain signals.
    Inputs: [B, T] or [B, 1, T]
    Output: Mean SI-SDR in dB over batch
    """
    # Ensure 2D shape [B, T]
    if estimated.dim() == 3:
        estimated = estimated.squeeze(1)
    if target.dim() == 3:
        target = target.squeeze(1)
    
    # Zero-mean
    estimated = estimated - torch.mean(estimated, dim=1, keepdim=True)
    target = target - torch.mean(target, dim=1, keepdim=True)
    
    # Projection
    alpha = (torch.sum(estimated * target, dim=1, keepdim=True) /
            (torch.sum(target ** 2, dim=1, keepdim=True) + eps))
    target_scaled = alpha * target
    
    # Noise
    noise = estimated - target_scaled
    
    # SI-SDR
    si_sdr_val = (torch.sum(target_scaled ** 2, dim=1) / 
                  (torch.sum(noise ** 2, dim=1) + eps))
    return 10 * torch.log10(si_sdr_val + eps).mean()

def si_sdr_loss(estimated, target):
    return -si_sdr(estimated, target)  # Negative for minimization

def l1_loss_complex(est, ref):
    return F.l1_loss(est[:, 0], ref[:, 0]) + F.l1_loss(est[:, 1], ref[:, 1])

class LossFunction(object):
    def __init__(self, device, win_size=320, hop_size=160):
        self.device = device
        self.resynthesizer = Resynthesizer(device, win_size, hop_size)

    def __call__(self, est, lbl, loss_mask, n_frames, mix, n_samples):
        # Mask padded spectrogram frames
        est_masked = est * loss_mask
        lbl_masked = lbl * loss_mask
        
        # Generate waveforms
        est_wave = self.resynthesizer(est_masked, mix)  # [B, T_padded]
        lbl_wave = self.resynthesizer(lbl_masked, mix)  # [B, T_padded]
        
        # Truncate to original length per sample
        B = est_wave.size(0)
        T = n_samples[0].item()  # All same in batch (training)
        est_wave = est_wave[:, :T]
        lbl_wave = lbl_wave[:, :T]
        
        # Compute losses
        loss_sisdr = si_sdr_loss(est_wave, lbl_wave)
        loss_mae = l1_loss_complex(est_masked, lbl_masked)
        
        return loss_sisdr + 0.5 * loss_mae