import torch
from torch import nn

from .archs.vqgan_arch import (
    DecoderUpOthersDoubleIdentity,
    EncoderDecomposeBaseDownOthersDoubleIdentity,
    VectorQuantizer,
)


class VQGANDecomposeModel(nn.Module):

    def __init__(self, opt):
        super().__init__()

        self.encoder = EncoderDecomposeBaseDownOthersDoubleIdentity(
            ch=opt['ch'],
            num_res_blocks=opt['num_res_blocks'],
            attn_resolutions=opt['attn_resolutions'],
            ch_mult=opt['ch_mult'],
            other_ch_mult=opt['other_ch_mult'],
            in_channels=opt['in_channels'],
            resolution=opt['resolution'],
            z_channels=opt['z_channels'],
            double_z=opt['double_z'],
            dropout=opt['dropout'],
        )
        self.decoder = DecoderUpOthersDoubleIdentity(
            in_channels=opt['in_channels'],
            resolution=opt['resolution'],
            z_channels=opt['z_channels'],
            ch=opt['ch'],
            out_ch=opt['out_ch'],
            num_res_blocks=opt['num_res_blocks'],
            attn_resolutions=opt['attn_resolutions'],
            ch_mult=opt['ch_mult'],
            other_ch_mult=opt['other_ch_mult'],
            dropout=opt['dropout'],
            resamp_with_conv=True,
            give_pre_end=False,
        )
        self.quantize_identity = VectorQuantizer(
            opt['n_embed'], opt['embed_dim'], beta=0.25
        )
        self.quant_conv_identity = torch.nn.Conv2d(
            opt["z_channels"], opt['embed_dim'], 1
        )
        self.post_quant_conv_identity = torch.nn.Conv2d(
            opt['embed_dim'], opt["z_channels"], 1
        )

        self.quantize_others = VectorQuantizer(
            opt['n_embed'], opt['embed_dim'] // 2, beta=0.25
        )
        self.quant_conv_others = torch.nn.Conv2d(
            opt["z_channels"] // 2, opt['embed_dim'] // 2, 1
        )
        self.post_quant_conv_others = torch.nn.Conv2d(
            opt['embed_dim'] // 2, opt["z_channels"] // 2, 1
        )

        self.load_state_dict(torch.load(opt['pretrained_models']))
        self.eval()
