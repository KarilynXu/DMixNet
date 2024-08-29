import torch
from torch import nn, einsum
from torch.nn import functional as F

import einops as ein
from einops import rearrange
from einops.layers.torch import Rearrange



def init_weights(m):
    if isinstance(m, (nn.Linear, nn.Conv2d)):
        nn.init.xavier_normal_(m.weight)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

class DNM(nn.Module):
    def __init__(self, in_channel, out_channel, num_branch=5, synapse_activation=nn.Sigmoid, dendritic_activation=nn.Sigmoid, soma=nn.Softmax, adaptive_weight=False):
        super(DNM, self).__init__()
        
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.nb = num_branch


        # synapse init
        self.sn0 = nn.LayerNorm(in_channel)
        self.sn3 = nn.LayerNorm(in_channel)
        self.sa = synapse_activation
        self.sw = nn.Parameter(torch.randn(out_channel, num_branch, in_channel))
        self.sb = nn.Parameter(torch.randn(out_channel, num_branch, in_channel))
        
        
        # dendritic init
        self.dn = nn.LayerNorm([num_branch, in_channel])
        self.da = dendritic_activation
        self.dw = nn.Parameter(torch.randn(out_channel, num_branch))
        self.dl = nn.Linear(num_branch, 1)
        
        # soma init (for classification, softmax is used)
        self.soma = soma
        self.adaptive_weight = adaptive_weight
        
    def forward(self, x):
        x = self.sn0(x)
        b, _ = x.shape
        x = ein.repeat(x, 'b d -> b o m d', o=self.out_channel, m=self.nb)

        sw = ein.repeat(self.sw, 'o m d -> b o m d', b=b)
        sb = ein.repeat(self.sb, 'o m d -> b o m d', b=b)
        
        x = sw * x + sb

        if self.sa is not None:
            x = self.sa(x)

        x = self.dn(x)
        x = x.sum(dim=3)
        dw = self.dw

        if self.adaptive_weight:
            x = dw * x

        if self.da is not None:
            x = self.da(x)

        x = x.sum(dim=2)

        if self.soma is not None:
            x = self.soma(x)
        
        return x

class DNM_Conv(nn.Module):
    def __init__(self, dim, num_patches, out_channel, dropout, num_branch=2, synapse_activation=None,
                 dendritic_activation=None, soma=nn.GELU):
        super(DNM_Conv, self).__init__()

        self.in_channel = num_patches
        self.out_channel = out_channel
        self.nb = num_branch
        self.dropout = nn.Dropout(dropout)

        # synapse init
        self.sn0 = nn.LayerNorm(dim)
        self.sa = synapse_activation
        self.sw = nn.Parameter(torch.randn(out_channel, num_branch, num_patches))
        self.sb = nn.Parameter(torch.randn(out_channel, num_branch, num_patches))

        # dendritic init
        self.da = dendritic_activation
        self.dw = nn.Parameter(torch.randn(out_channel, num_branch))
        self.dl = nn.Linear(num_branch, 1)

        # soma init (for classification, softmax is used)
        self.soma = soma

    def forward(self, x):
        # input shape (b, num_patches, in_channel), output shape (b,num_patchess, out_channel)
        out = self.sn0(x)  # 0. norm
        out = out.permute(0, 2, 1)

        b, d, n = out.shape
        out = ein.repeat(out, 'b d n -> b d o m n', o=self.out_channel, m=self.nb)

        sw = self.sw     # (o,m,n)
        sb = self.sb     # (o,m,n)

        out = sw * out + sb  # 2. wx + b      # (b,d,o,m,n)
        out = self.dropout(out)

        if self.sa is not None:
            out = self.sa()(out)  # 4. activation

        out = out.sum(dim=4)  # 1. each branch sum (b d o m n -> b d o m)

        dw = self.dw     # (o, m)
        out = dw * out

        if self.da is not None:
            out = self.da()(out)  # 2. activation

        # membrane (each dnm cell sum to final result)
        out = out.sum(dim=3)

        # soma
        if self.soma is not None:
            out = self.soma()(out)

        out = out.permute(0, 2, 1)

        return x * (out + 1)

class ChannelMixer(nn.Module):
    def __init__(self, dim, num_patches,  hidden_c, dropout):
            super(ChannelMixer, self).__init__()
            self.ln = nn.LayerNorm(dim)
            self.fc1 = nn.Conv1d(num_patches, hidden_c, kernel_size=1)
            self.do1 = nn.Dropout(dropout)
            self.fc2 = nn.Conv1d(hidden_c, num_patches, kernel_size=1)
            self.do2 = nn.Dropout(dropout)
            self.act = F.gelu
    def forward(self, x):
            # x.shape ==(batch_size,num_patches,num_features)
            out = self.ln(x)
            out = self.fc1(out)
            out = self.do1(self.act(out))
            out = self.fc2(out)
            out = self.do2(out)
            return out + x

class MLPblock(nn.Module):
    def __init__(self, dim, num_patches, num_hidden, dropout):
        super(MLPblock, self).__init__()
        self.ln = nn.LayerNorm(dim)
        self.fc1 = nn.Linear(num_patches, num_hidden)
        self.do1 = nn.Dropout(dropout)
        self.fc2 = nn.Linear(num_hidden, num_patches)
        self.do2 = nn.Dropout(dropout)
        self.act = F.gelu
    def forward(self, x):
        out = self.ln(x)
        out = out.permute(0, 2, 1)
        out = self.do1(self.act(self.fc1(out)))
        out = self.do2(self.fc2(out))
        out = out.permute(0, 2, 1)
        return x + out

class TokenMixer(nn.Module):
    def __init__(self, dim, num_hidden, dropout):
        super(TokenMixer, self).__init__()
        self.ln = nn.LayerNorm(dim)
        self.fc1 = nn.Linear(dim, num_hidden)
        self.do1 = nn.Dropout(dropout)
        self.fc2 = nn.Linear(num_hidden, dim)
        self.do2 = nn.Dropout(dropout)
        self.act = F.gelu
    def forward(self, x):
        # x.shape == (batch_size, num_patches, num_features)
        out = self.ln(x)
        out = self.do1(self.act(self.fc1(out)))
        out = self.do2(self.fc2(out))
        return out + x


class MixerLayer(nn.Module):
    def __init__(self, dim, num_patches,  hidden_c, hidden_s, dropout):
        super(MixerLayer, self).__init__()
        self.channel = MLPblock(dim, num_patches, hidden_c, dropout)
        # self.channel = DNM_Conv(dim, num_patches, num_patches, dropout)
        # self.channel = ChannelMixer(dim, num_patches,  hidden_c, dropout)
        self.token = TokenMixer(dim, hidden_s, dropout)
    def forward(self,x):
        x = self.channel(x)
        x = self.token(x)
        return x



class Mixer(nn.Module):
    def __init__(self, img_size, patch_size, num_classes, dim,  depth, hidden_c, hidden_s, is_cls_token, in_channels=3,
                 dropout=0., mlp_head='original'):
        super().__init__()

        assert img_size % patch_size == 0, 'Image dimensions must be divisible by the patch size.'
        self.num_patch = (img_size // patch_size) ** 2
        self.to_patch_embedding = nn.Sequential(
            nn.Conv2d(in_channels, dim, patch_size, patch_size),
            Rearrange('b c h w -> b (h w) c'),
        )
        self.dim = dim
        self.is_cls_token = is_cls_token

        if self.is_cls_token:
            self.cls_token = nn.Parameter(torch.randn(1, 1, self.dim))
            self.num_patch += 1

        self.mlp_blocks = nn.Sequential(
            *[
                MixerLayer(dim, self.num_patch,  hidden_c, hidden_s, dropout)
             for _ in range(depth)
            ]
        )

        if mlp_head == 'original':
            self.mlp_head = nn.Sequential(
                nn.LayerNorm(dim),
                nn.Linear(dim, num_classes)
            )
        else:
            self.mlp_head = mlp_head

        self.apply(init_weights)

    def forward(self, img):
        x = self.to_patch_embedding(img)
        b, n, _ = x.shape
        if self.is_cls_token:
            cls_tokens = ein.repeat(self.cls_token, '() n d -> b n d', b=b)
            x = torch.cat((cls_tokens, x), dim=1)

        x = self.mlp_blocks(x)
        x = x[:, 0] if self.is_cls_token else x.mean(dim=1)

        return self.mlp_head(x)


class DMixNet(nn.Module):
    def __init__(self, config, device='cpu'):
        super(DMixNet, self).__init__()
        Mixer_config = config['Dixnet']
        dnm_config = self.str2func(config['dnm'])

        if config['mlp_head'] == 'original':
            self.mlp_head = 'original'

        elif config['mlp_head'] == 'dnm':
            self.mlp_head = DNM(**dnm_config).to(device)
        self.net = Mixer(**Mixer_config, mlp_head=self.mlp_head).to(device)
    def str2func(self, config):
        # sigmoid, relu, gelu, softmax
        for k, func_str in config.items():
            if not isinstance(func_str, str):
                continue
            if func_str.casefold() == 'sigmoid'.casefold():
                func_str = nn.Sigmoid()
            elif func_str.casefold() == 'relu'.casefold():
                func_str = nn.ReLU()
            elif func_str.casefold() == 'gelu'.casefold():
                func_str = nn.GELU()
            elif func_str.casefold() == 'softmax'.casefold():
                func_str = nn.Softmax(dim=1)
            elif func_str.casefold() == 'none'.casefold():
                func_str = None
            config[k] = func_str
        return config

    def forward(self, x):
        return self.net(x)


        
        
        
        
        
        
        
        
        
        
        
        
