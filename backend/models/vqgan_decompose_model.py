import torch

from models.archs.vqgan_arch import (
    DecoderUpOthersDoubleIdentity,
    EncoderDecomposeBaseDownOthersDoubleIdentity,
    VectorQuantizer,
)


class VQGANDecomposeModel:

    def __init__(self, opt, device: torch.device):
        self.opt = opt

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
        ).to(device)
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
        ).to(device)
        self.quantize_identity = VectorQuantizer(
            opt['n_embed'], opt['embed_dim'], beta=0.25
        ).to(device)
        self.quant_conv_identity = torch.nn.Conv2d(
            opt["z_channels"], opt['embed_dim'], 1
        ).to(device)
        self.post_quant_conv_identity = torch.nn.Conv2d(
            opt['embed_dim'], opt["z_channels"], 1
        ).to(device)

        self.quantize_others = VectorQuantizer(
            opt['n_embed'], opt['embed_dim'] // 2, beta=0.25
        ).to(device)
        self.quant_conv_others = torch.nn.Conv2d(
            opt["z_channels"] // 2, opt['embed_dim'] // 2, 1
        ).to(device)
        self.post_quant_conv_others = torch.nn.Conv2d(
            opt['embed_dim'] // 2, opt["z_channels"] // 2, 1
        ).to(device)

    def load_pretrained_network(self):
        checkpoint = torch.load(self.opt['pretrained_models'])

        self.encoder.load_state_dict(checkpoint['encoder'])
        self.decoder.load_state_dict(checkpoint['decoder'])
        self.quantize_identity.load_state_dict(checkpoint['quantize_identity'])
        self.quant_conv_identity.load_state_dict(checkpoint['quant_conv_identity'])
        self.post_quant_conv_identity.load_state_dict(
            checkpoint['post_quant_conv_identity']
        )
        self.quantize_others.load_state_dict(checkpoint['quantize_others'])
        self.quant_conv_others.load_state_dict(checkpoint['quant_conv_others'])
        self.post_quant_conv_others.load_state_dict(
            checkpoint['post_quant_conv_others']
        )

        self.encoder.eval()
        self.decoder.eval()
        self.quantize_identity.eval()
        self.quant_conv_identity.eval()
        self.post_quant_conv_identity.eval()
        self.quantize_others.eval()
        self.quant_conv_others.eval()
        self.post_quant_conv_others.eval()
