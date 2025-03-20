import random
from datetime import datetime

import h5py
import numpy as np
import torch

############################################################
# 1) WavReader
############################################################
class WavReader(object):
    def __init__(self, in_file, mode):

        self.mode = mode
        self.in_file = in_file
        assert self.mode in {'train', 'eval'}

        if self.mode == 'train':
            self.wav_dict = {i: wavfile for i, wavfile in enumerate(in_file)}
        else:
            reader = h5py.File(in_file, 'r')
            self.wav_dict = {i: str(i) for i in range(len(reader))}
            reader.close()
        
        self.wav_indices = sorted(list(self.wav_dict.keys()))

    def load(self, idx):
        if self.mode == 'train':
            filename = self.wav_dict[idx]
            reader = h5py.File(filename, 'r')
            mix = reader['mix'][:]
            sph = reader['sph'][:]
            reader.close()
        else:
            reader = h5py.File(self.in_file, 'r')
            reader_grp = reader[self.wav_dict[idx]]
            mix = reader_grp['mix'][:]
            sph = reader_grp['sph'][:]
            reader.close()
        return mix, sph

    def __iter__(self):
        for idx in self.wav_indices:
            yield idx, self.load(idx)

############################################################
# 2) PerUttLoader
############################################################
class PerUttLoader(object):
    def __init__(self, in_file, in_norm, shuffle=True, mode='train'):
        self.shuffle = shuffle
        self.mode = mode
        self.wav_reader = WavReader(in_file, mode)
        self.in_norm = in_norm
        self.eps = np.finfo(np.float32).eps

        self.MAX_LEN_SAMPLES = 128000  # 8 sec * 16 kHz

    def __iter__(self):
        if self.shuffle:
            random.shuffle(self.wav_reader.wav_indices)

        for idx, utt in self.wav_reader:
            mix, sph = utt[0], utt[1]
            
            # Truncate if longer than MAX_LEN_SAMPLES
            if mix.shape[0] > self.MAX_LEN_SAMPLES:
                mix = mix[:self.MAX_LEN_SAMPLES]
                sph = sph[:self.MAX_LEN_SAMPLES]
            
            # Normalization if in_norm is True
            utt_eg = dict()
            if self.in_norm:
                scale = np.max(np.abs(mix)) / 0.9
                # Avoid dividing by zero if all samples are zero
                if scale == 0:
                    scale = 1.0
                utt_eg['mix'] = mix / scale
                utt_eg['sph'] = sph / scale
            else:
                utt_eg['mix'] = mix
                utt_eg['sph'] = sph

            utt_eg['n_samples'] = utt_eg['mix'].shape[0]
            yield utt_eg

############################################################
# 3) SegSplitter
############################################################
class SegSplitter(object):
    def __init__(self, segment_size, sample_rate, hop_size):
        self.seg_len = int(sample_rate * segment_size)
        self.hop_len = int(sample_rate * hop_size) 
        
    def __call__(self, utt_eg):
        n_samples = utt_eg['n_samples']
        segs = []
        if n_samples < self.seg_len:
            pad_size = self.seg_len - n_samples
            seg = dict()
            seg['mix'] = np.pad(utt_eg['mix'], [(0, pad_size)])
            seg['sph'] = np.pad(utt_eg['sph'], [(0, pad_size)])
            seg['n_samples'] = n_samples
            segs.append(seg)
        else:
            s_point = 0
            while True:
                if s_point + self.seg_len > n_samples:
                    break
                seg = dict()
                seg['mix'] = utt_eg['mix'][s_point:s_point+self.seg_len]
                seg['sph'] = utt_eg['sph'][s_point:s_point+self.seg_len]
                seg['n_samples'] = self.seg_len
                s_point += self.hop_len
                segs.append(seg)
        return segs

############################################################
# 4) AudioLoader
############################################################
class AudioLoader(object):
    def __init__(self, 
                 in_file, 
                 sample_rate,
                 unit='seg',
                 segment_size=4.0,
                 segment_shift=1.0, 
                 batch_size=4, 
                 buffer_size=16,
                 in_norm=True,
                 mode='train'):

        self.mode = mode
        assert self.mode in {'train', 'eval'}
        self.unit = unit
        assert self.unit in {'seg', 'utt'}

        if self.mode == 'train':
            self.utt_loader = PerUttLoader(in_file, in_norm, shuffle=True, mode='train')
        else:
            self.utt_loader = PerUttLoader(in_file, in_norm, shuffle=False, mode='eval')

        if unit == 'seg':
            self.seg_splitter = SegSplitter(segment_size, sample_rate, hop_size=segment_shift)
        
        self.batch_size = batch_size
        self.buffer_size = buffer_size

    def make_batch(self, load_list):
        n_batches = len(load_list) // self.batch_size
        if n_batches == 0:
            return []
        else:
            batch_queue = [[] for _ in range(n_batches)]
            idx = 0
            for seg in load_list[0 : n_batches*self.batch_size]:
                batch_queue[idx].append(seg)
                idx = (idx + 1) % n_batches

            if self.unit == 'utt':
                # For the 'utt' mode, we pad each audio in the batch to the max length in that batch
                for batch in batch_queue:
                    sig_len = max([eg['mix'].shape[0] for eg in batch])
                    for i in range(len(batch)):
                        pad_size = sig_len - batch[i]['mix'].shape[0]
                        if pad_size > 0:
                            batch[i]['mix'] = np.pad(batch[i]['mix'], [(0, pad_size)])
                            batch[i]['sph'] = np.pad(batch[i]['sph'], [(0, pad_size)])
            return batch_queue

    def to_tensor(self, x):
        return torch.from_numpy(x).float()

    def batch_buffer(self):
        while True:
            try:
                utt_eg = next(self.load_iter)
                if self.unit == 'seg':
                    segs = self.seg_splitter(utt_eg)
                    self.load_list.extend(segs)
                else:
                    self.load_list.append(utt_eg)
            except StopIteration:
                self.stop_iter = True
                break

            if len(self.load_list) >= self.buffer_size:
                break
        
        batch_queue = self.make_batch(self.load_list)
        batch_list = []
        for eg_list in batch_queue:
            batch = {
                'mix': torch.stack([self.to_tensor(eg['mix']) for eg in eg_list], dim=0),
                'sph': torch.stack([self.to_tensor(eg['sph']) for eg in eg_list], dim=0),
                'n_samples': torch.tensor([eg['n_samples'] for eg in eg_list], dtype=torch.int64)
            }
            batch_list.append(batch)

        # Keep leftover examples that didn't form a full batch
        rn = len(self.load_list) % self.batch_size
        self.load_list = self.load_list[-rn:] if rn else []
        return batch_list

    def __iter__(self):
        self.load_iter = iter(self.utt_loader)
        self.stop_iter = False
        self.load_list = []
        while True:
            if self.stop_iter:
                break
            egs_buffer = self.batch_buffer()
            for egs in egs_buffer:
                yield egs
