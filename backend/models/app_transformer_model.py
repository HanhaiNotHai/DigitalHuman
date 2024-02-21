import numpy as np
import torch
import torch.distributions as dists
from sentence_transformers import SentenceTransformer
from torchvision.utils import save_image

from models.archs.transformer_arch import TransformerLanguage
from models.archs.vqgan_arch import (
    DecoderUpOthersDoubleIdentity,
    EncoderDecomposeBaseDownOthersDoubleIdentity,
    VectorQuantizer,
)


class AppTransformerModel:
    """Texture-Aware Diffusion based Transformer model."""

    def __init__(self, opt):
        self.opt = opt
        self.device = torch.device('cuda')

        # VQVAE for image
        self.img_encoder = EncoderDecomposeBaseDownOthersDoubleIdentity(
            ch=opt['img_ch'],
            num_res_blocks=opt['img_num_res_blocks'],
            attn_resolutions=opt['img_attn_resolutions'],
            ch_mult=opt['img_ch_mult'],
            other_ch_mult=opt['img_other_ch_mult'],
            in_channels=opt['img_in_channels'],
            resolution=opt['img_resolution'],
            z_channels=opt['img_z_channels'],
            double_z=opt['img_double_z'],
            dropout=opt['img_dropout'],
        ).to(self.device)
        self.img_decoder = DecoderUpOthersDoubleIdentity(
            in_channels=opt['img_in_channels'],
            resolution=opt['img_resolution'],
            z_channels=opt['img_z_channels'],
            ch=opt['img_ch'],
            out_ch=opt['img_out_ch'],
            num_res_blocks=opt['img_num_res_blocks'],
            attn_resolutions=opt['img_attn_resolutions'],
            ch_mult=opt['img_ch_mult'],
            other_ch_mult=opt['img_other_ch_mult'],
            dropout=opt['img_dropout'],
            resamp_with_conv=True,
            give_pre_end=False,
        ).to(self.device)
        self.quantize_identity = VectorQuantizer(
            opt['img_n_embed'], opt['img_embed_dim'], beta=0.25
        ).to(self.device)
        self.quant_conv_identity = torch.nn.Conv2d(
            opt["img_z_channels"], opt['img_embed_dim'], 1
        ).to(self.device)
        self.post_quant_conv_identity = torch.nn.Conv2d(
            opt['img_embed_dim'], opt["img_z_channels"], 1
        ).to(self.device)

        self.quantize_others = VectorQuantizer(
            opt['img_n_embed'], opt['img_embed_dim'] // 2, beta=0.25
        ).to(self.device)
        self.quant_conv_others = torch.nn.Conv2d(
            opt["img_z_channels"] // 2, opt['img_embed_dim'] // 2, 1
        ).to(self.device)
        self.post_quant_conv_others = torch.nn.Conv2d(
            opt['img_embed_dim'] // 2, opt["img_z_channels"] // 2, 1
        ).to(self.device)
        self.load_pretrained_image_vae()

        # define sampler
        self._denoise_fn = TransformerLanguage(
            codebook_size=opt['codebook_size'],
            bert_n_emb=opt['bert_n_emb'],
            bert_n_layers=opt['bert_n_layers'],
            bert_n_head=opt['bert_n_head'],
            block_size=opt['block_size'] * 2,
            embd_pdrop=opt['embd_pdrop'],
            resid_pdrop=opt['resid_pdrop'],
            attn_pdrop=opt['attn_pdrop'],
        ).to(self.device)

        self.shape = tuple(opt['latent_shape'])

        self.mask_id = opt['codebook_size']

        self.sample_steps = opt['sample_steps']

        self.get_fixed_language_model()

    def load_pretrained_image_vae(self):
        # load pretrained vqgan for segmentation mask
        img_ae_checkpoint = torch.load(self.opt['img_ae_path'])
        self.img_encoder.load_state_dict(img_ae_checkpoint['encoder'], strict=True)
        self.img_decoder.load_state_dict(img_ae_checkpoint['decoder'], strict=True)
        self.quantize_identity.load_state_dict(
            img_ae_checkpoint['quantize_identity'], strict=True
        )
        self.quant_conv_identity.load_state_dict(
            img_ae_checkpoint['quant_conv_identity'], strict=True
        )
        self.post_quant_conv_identity.load_state_dict(
            img_ae_checkpoint['post_quant_conv_identity'], strict=True
        )
        self.quantize_others.load_state_dict(
            img_ae_checkpoint['quantize_others'], strict=True
        )
        self.quant_conv_others.load_state_dict(
            img_ae_checkpoint['quant_conv_others'], strict=True
        )
        self.post_quant_conv_others.load_state_dict(
            img_ae_checkpoint['post_quant_conv_others'], strict=True
        )
        self.img_encoder.eval()
        self.img_decoder.eval()
        self.quantize_identity.eval()
        self.quant_conv_identity.eval()
        self.post_quant_conv_identity.eval()
        self.quantize_others.eval()
        self.quant_conv_others.eval()
        self.post_quant_conv_others.eval()

    @torch.no_grad()
    def decode(self, quant_list):
        quant_identity = self.post_quant_conv_identity(quant_list[0])
        quant_frame = self.post_quant_conv_others(quant_list[1])
        dec = self.img_decoder(quant_identity, quant_frame)
        return dec

    @torch.no_grad()
    def decode_image_indices(self, quant_identity, quant_frame):
        quant_identity = self.quantize_identity.get_codebook_entry(
            quant_identity,
            (
                quant_identity.size(0),
                self.shape[0],
                self.shape[1],
                self.opt["img_z_channels"],
            ),
        )
        quant_frame = self.quantize_others.get_codebook_entry(
            quant_frame,
            (
                quant_frame.size(0),
                self.shape[0] // 4,
                self.shape[1] // 4,
                self.opt["img_z_channels"] // 2,
            ),
        )
        dec = self.decode([quant_identity, quant_frame])

        return dec

    def get_fixed_language_model(self):
        self.language_model = SentenceTransformer('all-MiniLM-L6-v2')

    @torch.no_grad()
    def get_text_embedding(self):
        self.text_embedding = self.language_model.encode(
            self.text, show_progress_bar=False
        )
        self.text_embedding = (
            torch.Tensor(self.text_embedding).to(self.device).unsqueeze(1)
        )

    def sample_fn(self, temp=1.0, sample_steps=None):
        self._denoise_fn.eval()

        b, device = self.image.size(0), 'cuda'
        x_identity_t = (
            torch.ones((b, np.prod(self.shape)), device=device).long() * self.mask_id
        )
        x_pose_t = (
            torch.ones((b, np.prod(self.shape) // 16), device=device).long()
            * self.mask_id
        )
        unmasked_identity = torch.zeros_like(x_identity_t, device=device).bool()
        unmasked_pose = torch.zeros_like(x_pose_t, device=device).bool()
        sample_steps = list(range(1, sample_steps + 1))

        for t in reversed(sample_steps):
            print(f'Sample timestep {t:4d}', end='\r')
            t = torch.full((b,), t, device=device, dtype=torch.long)

            # where to unmask
            changes_identity = torch.rand(
                x_identity_t.shape, device=device
            ) < 1 / t.float().unsqueeze(-1)
            # don't unmask somewhere already unmasked
            changes_identity = torch.bitwise_xor(
                changes_identity, torch.bitwise_and(changes_identity, unmasked_identity)
            )
            # update mask with changes
            unmasked_identity = torch.bitwise_or(unmasked_identity, changes_identity)

            changes_pose = torch.rand(
                x_pose_t.shape, device=device
            ) < 1 / t.float().unsqueeze(-1)
            # don't unmask somewhere already unmasked
            changes_pose = torch.bitwise_xor(
                changes_pose, torch.bitwise_and(changes_pose, unmasked_pose)
            )
            # update mask with changes
            unmasked_pose = torch.bitwise_or(unmasked_pose, changes_pose)

            x_identity_0_hat_logits, x_pose_0_hat_logits = self._denoise_fn(
                self.text_embedding, x_identity_t, x_pose_t, t=t
            )

            x_identity_0_hat_logits = x_identity_0_hat_logits[
                :, 1 : 1 + self.shape[0] * self.shape[1], :
            ]
            x_pose_0_hat_logits = x_pose_0_hat_logits[
                :, 1 + self.shape[0] * self.shape[1] :
            ]

            # scale by temperature
            x_identity_0_hat_logits = x_identity_0_hat_logits / temp
            x_identity_0_dist = dists.Categorical(logits=x_identity_0_hat_logits)
            x_identity_0_hat = x_identity_0_dist.sample().long()

            x_pose_0_hat_logits = x_pose_0_hat_logits / temp
            x_pose_0_dist = dists.Categorical(logits=x_pose_0_hat_logits)
            x_pose_0_hat = x_pose_0_dist.sample().long()

            # x_t would be the input to the transformer, so the index range should be continual one
            x_identity_t[changes_identity] = x_identity_0_hat[changes_identity]
            x_pose_t[changes_pose] = x_pose_0_hat[changes_pose]

        self._denoise_fn.train()

        return x_identity_t, x_pose_t

    def sample_appearance(self, text, save_path, shape=[256, 128]):
        self._denoise_fn.eval()

        self.text = text
        self.image = torch.zeros([1, 3, shape[0], shape[1]]).to(self.device)
        self.get_text_embedding()

        with torch.no_grad():
            x_identity_t, x_pose_t = self.sample_fn(
                temp=1, sample_steps=self.sample_steps
            )

        self.get_vis_generated_only(x_identity_t, x_pose_t, save_path)

        quant_identity = self.quantize_identity.get_codebook_entry(
            x_identity_t,
            (
                x_identity_t.size(0),
                self.shape[0],
                self.shape[1],
                self.opt["img_z_channels"],
            ),
        )
        quant_frame = (
            self.quantize_others.get_codebook_entry(
                x_pose_t,
                (
                    x_pose_t.size(0),
                    self.shape[0] // 4,
                    self.shape[1] // 4,
                    self.opt["img_z_channels"] // 2,
                ),
            )
            .view(x_pose_t.size(0), self.opt["img_z_channels"] // 2, -1)
            .permute(0, 2, 1)
        )

        self._denoise_fn.train()

        return quant_identity, quant_frame

    def get_vis_generated_only(self, quant_identity, quant_frame, save_path):
        # pred image
        pred_img = self.decode_image_indices(quant_identity, quant_frame)
        img_cat = (pred_img.detach() + 1) / 2
        img_cat = img_cat.clamp_(0, 1)
        save_image(img_cat, save_path, nrow=1, padding=4)

    def load_network(self):
        checkpoint = torch.load(self.opt['pretrained_sampler'])
        self._denoise_fn.load_state_dict(checkpoint, strict=True)
        self._denoise_fn.eval()