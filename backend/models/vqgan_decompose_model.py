import torch

from models.archs.vqgan_arch import (
    DecoderUpOthersDoubleIdentity,
    EncoderDecomposeBaseDownOthersDoubleIdentity,
    VectorQuantizer,
)
from models.base_model import BaseModel


class VQGANDecomposeModel(BaseModel):

    def __init__(self, opt):
        super().__init__(opt)

        self.encoder = self.model_to_device(
            EncoderDecomposeBaseDownOthersDoubleIdentity(
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
        )
        self.decoder = self.model_to_device(
            DecoderUpOthersDoubleIdentity(
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
        )
        self.quantize_identity = self.model_to_device(
            VectorQuantizer(opt['n_embed'], opt['embed_dim'], beta=0.25)
        )
        self.quant_conv_identity = self.model_to_device(
            torch.nn.Conv2d(opt["z_channels"], opt['embed_dim'], 1)
        )
        self.post_quant_conv_identity = self.model_to_device(
            torch.nn.Conv2d(opt['embed_dim'], opt["z_channels"], 1)
        )

        self.quantize_others = self.model_to_device(
            VectorQuantizer(opt['n_embed'], opt['embed_dim'] // 2, beta=0.25)
        )
        self.quant_conv_others = self.model_to_device(
            torch.nn.Conv2d(opt["z_channels"] // 2, opt['embed_dim'] // 2, 1)
        )
        self.post_quant_conv_others = self.model_to_device(
            torch.nn.Conv2d(opt['embed_dim'] // 2, opt["z_channels"] // 2, 1)
        )

    def load_pretrained_network(self):

        self.load_network(
            self.encoder, self.opt['pretrained_models'], param_key='encoder'
        )
        self.load_network(
            self.decoder, self.opt['pretrained_models'], param_key='decoder'
        )
        self.load_network(
            self.quantize_identity,
            self.opt['pretrained_models'],
            param_key='quantize_identity',
        )
        self.load_network(
            self.quant_conv_identity,
            self.opt['pretrained_models'],
            param_key='quant_conv_identity',
        )
        self.load_network(
            self.post_quant_conv_identity,
            self.opt['pretrained_models'],
            param_key='post_quant_conv_identity',
        )
        self.load_network(
            self.quantize_others,
            self.opt['pretrained_models'],
            param_key='quantize_others',
        )
        self.load_network(
            self.quant_conv_others,
            self.opt['pretrained_models'],
            param_key='quant_conv_others',
        )
        self.load_network(
            self.post_quant_conv_others,
            self.opt['pretrained_models'],
            param_key='post_quant_conv_others',
        )
