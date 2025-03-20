import math
import torch
from torch import nn
import torch.nn.functional as F
from utils.ReversibleSeq import *
from sru import SRU  

def default(val, default_val):
    return val if val is not None else default_val

def init_(tensor):
    dim = tensor.shape[-1]
    std = 1 / math.sqrt(dim)
    tensor.uniform_(-std, std)
    return tensor

class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn
    def forward(self, x):
        return x + self.fn(x)

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = nn.LayerNorm(dim)
    def forward(self, x, **kwargs):
        x = self.norm(x)
        return self.fn(x, **kwargs)

class GELU_(nn.Module):
    def forward(self, x):
        return 0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))

GELU = nn.GELU if hasattr(nn, 'GELU') else GELU_

class FeedForward(nn.Module):
    def __init__(self, dim, mult=4, dropout=0., activation=None, glu=False):
        super().__init__()
        activation = default(activation, GELU)

        self.glu = glu
        self.w1 = nn.Linear(dim, dim * mult * (2 if glu else 1))
        self.act = activation()
        self.dropout = nn.Dropout(dropout)
        self.w2 = nn.Linear(dim * mult, dim)

    def forward(self, x, **kwargs):
        if not self.glu:
            x = self.w1(x)
            x = self.act(x)
        else:
            x, v = self.w1(x).chunk(2, dim=-1)
            x = self.act(x) * v

        x = self.dropout(x)
        x = self.w2(x)
        return x

class MultiHeadSRU(nn.Module):
    def __init__(self, input_dim, heads, dim_head):
        super().__init__()
        self.heads = heads
        self.dim_head = dim_head
        self.sru = SRU(
            input_size=input_dim,
            hidden_size=heads * dim_head,
            num_layers=1,
            bidirectional=False,
            dropout=0.0
        )

    def forward(self, x):
        batch_size, seq_len, input_dim = x.size()
        x = x.transpose(0, 1)  # (seq_len, batch_size, input_dim)
        sru_output, _ = self.sru(x)  # (seq_len, batch_size, heads*dim_head)
        sru_output = sru_output.transpose(0, 1)  # (batch_size, seq_len, heads*dim_head)
        sru_output = sru_output.view(batch_size, seq_len, self.heads, self.dim_head)
        sru_output = sru_output.permute(0, 2, 1, 3)
        return sru_output

class GinformerSelfAttention(nn.Module):  
    def __init__(self, dim, seq_len, k=256, heads=8, dim_head=None, one_kv_head=False, share_kv=False, dropout=0.):
        super().__init__()
        assert (dim % heads) == 0, 

        self.seq_len = seq_len
        self.k = k
        self.heads = heads

        dim_head = default(dim_head, dim // heads)
        self.dim_head = dim_head

        self.to_q = nn.Linear(dim, dim_head * heads, bias=False)

        kv_dim = dim_head if one_kv_head else (dim_head * heads)
        self.to_k = nn.Linear(dim, kv_dim, bias=False)
        self.proj_k = nn.Parameter(init_(torch.zeros(seq_len, k)))

        self.share_kv = share_kv
        if not share_kv:
            self.to_v = nn.Linear(dim, kv_dim, bias=False)
            self.proj_v = nn.Parameter(init_(torch.zeros(seq_len, k)))

        self.dropout = nn.Dropout(dropout)
        self.to_out = nn.Linear(dim_head * heads, dim)
        self.multi_head_sru = MultiHeadSRU(dim, heads, dim_head)

    def forward(self, x, context=None, **kwargs):
        b, n, d = x.shape
        d_h, h, k = self.dim_head, self.heads, self.k

        kv_input = x if context is None else context
        kv_len = kv_input.shape[1]


        queries = self.to_q(x).reshape(b, n, h, d_h).transpose(1, 2)  # (b, h, n, d_h)

        keys = self.to_k(kv_input)
        values = self.to_v(kv_input) if not self.share_kv else keys

        proj_seq_len = lambda args: torch.einsum('bnd,nk->bkd', *args)


        proj_k = self.proj_k[:kv_len]
        proj_v = self.proj_v[:kv_len] if not self.share_kv else self.proj_k[:kv_len]
        kv_projs = (proj_k, proj_v)

        keys, values = map(proj_seq_len, zip((keys, values), kv_projs))  # (b, k, d_h)

        keys = keys.reshape(b, 1, k, d_h).expand(-1, h, -1, -1)
        values = values.reshape(b, 1, k, d_h).expand(-1, h, -1, -1)

        gate_weights = torch.sigmoid(self.multi_head_sru(x))  # (b, h, n, d_h)
        queries = queries * gate_weights

        dots = torch.einsum('bhnd,bhkd->bhnk', queries, keys) * (d_h ** -0.5)
        attn = dots.softmax(dim=-1)
        attn = self.dropout(attn)
        out = torch.einsum('bhnk,bhkd->bhnd', attn, values)

        out = out.transpose(1, 2).reshape(b, n, h * d_h)
        return self.to_out(out)

class Ginformer(nn.Module):  
    def __init__(self, dim, seq_len, depth, k=256, heads=8, dim_head=None, one_kv_head=False, share_kv=False, reversible=False, dropout=0.):
        super().__init__()
        layers = nn.ModuleList([])
        for _ in range(depth):
            attn = GinformerSelfAttention(dim, seq_len, k=k, heads=heads, dim_head=dim_head, one_kv_head=one_kv_head, share_kv=share_kv, dropout=dropout)
            ff = FeedForward(dim, dropout=dropout)

            layers.append(nn.ModuleList([
                PreNorm(dim, attn),
                PreNorm(dim, ff)
            ]))

        execute_type = ReversibleSequence if reversible else SequentialSequence
        self.net = execute_type(layers)

    def forward(self, x):
        return self.net(x)

class AIA_Ginformer(nn.Module):  
    def __init__(self, input_size, output_size, seq_len, depth=2, k=256, heads=4, dim_head=None, one_kv_head=True, share_kv=True, dropout=0):
        super(AIA_Ginformer, self).__init__()

        self.input_size = input_size
        self.output_size = output_size

        self.input = nn.Sequential(
            nn.Conv2d(input_size, input_size // 2, kernel_size=1),
            nn.PReLU()
        )


        self.ginformer = Ginformer(  
            dim=input_size // 2,
            seq_len=seq_len,
            depth=depth,
            k=k,
            heads=heads,
            dim_head=dim_head,
            one_kv_head=one_kv_head,
            share_kv=share_kv,
            dropout=dropout
        )
 
        self.output = nn.Sequential(
            nn.PReLU(),
            nn.Conv2d(input_size // 2, output_size, kernel_size=1),
            nn.BatchNorm2d(output_size, eps=1e-8)
        )

    def forward(self, input):
        b, c, f, t = input.shape
        input = self.input(input)
        input_reshaped = input.permute(0, 2, 3, 1).contiguous().view(b * f, t, c // 2)
        output_reshaped = self.ginformer(input_reshaped)  
        output = output_reshaped.view(b, f, t, c // 2).permute(0, 3, 1, 2).contiguous()
        output = self.output(output)
        return output
