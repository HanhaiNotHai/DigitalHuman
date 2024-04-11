import numpy as np
import torch
import torch.distributions as dists
from sentence_transformers import SentenceTransformer
from torchvision.utils import save_image

from .archs.transformer_arch import TransformerLanguage
from .vqgan_decompose_model import VQGANDecomposeModel


class AppTransformerModel:
    """Texture-Aware Diffusion based Transformer model."""

    def __init__(
        self,
        opt,
        vq_decompose_model: VQGANDecomposeModel,
        language_model: SentenceTransformer,
        device: torch.device,
    ):
        self.opt = opt
        self.device = device

        # VQVAE for image
        self.img_encoder = vq_decompose_model.encoder
        self.img_decoder = vq_decompose_model.decoder
        self.quantize_identity = vq_decompose_model.quantize_identity
        self.quant_conv_identity = vq_decompose_model.quant_conv_identity
        self.post_quant_conv_identity = vq_decompose_model.post_quant_conv_identity

        self.quantize_others = vq_decompose_model.quantize_others
        self.quant_conv_others = vq_decompose_model.quant_conv_others
        self.post_quant_conv_others = vq_decompose_model.post_quant_conv_others

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

        self.language_model = language_model

        self._denoise_fn.load_state_dict(
            torch.load(opt['pretrained_sampler']), strict=True
        )
        self._denoise_fn.eval()

    @torch.no_grad()
    def decode(self, quant_list):
        quant_identity = self.post_quant_conv_identity(quant_list[0])
        quant_frame = self.post_quant_conv_others(quant_list[1])
        dec = self.img_decoder(quant_identity, quant_frame)
        return dec

    @torch.no_grad()
    def decode_image_indices(self, quant_identity, quant_frame):
        self.quant_identity = self.quantize_identity.get_codebook_entry(
            quant_identity,
            (
                quant_identity.size(0),
                self.shape[0],
                self.shape[1],
                self.opt["img_z_channels"],
            ),
        )
        self.quant_frame = self.quantize_others.get_codebook_entry(
            quant_frame,
            (
                quant_frame.size(0),
                self.shape[0] // 4,
                self.shape[1] // 4,
                self.opt["img_z_channels"] // 2,
            ),
        )
        dec = self.decode([self.quant_identity, self.quant_frame])

        return dec

    @torch.no_grad()
    def get_text_embedding(self):
        self.text_embedding = self.language_model.encode(
            self.text, show_progress_bar=False
        )
        self.text_embedding = (
            torch.Tensor(self.text_embedding).to(self.device).unsqueeze(1)
        )

    def sample_fn(self, temp=1.0, sample_steps=None):
        b = self.image.size(0)
        x_identity_t = (
            torch.ones((b, np.prod(self.shape)), device=self.device).long()
            * self.mask_id
        )
        x_pose_t = (
            torch.ones((b, np.prod(self.shape) // 16), device=self.device).long()
            * self.mask_id
        )
        unmasked_identity = torch.zeros_like(x_identity_t, device=self.device).bool()
        unmasked_pose = torch.zeros_like(x_pose_t, device=self.device).bool()
        sample_steps = list(range(1, sample_steps + 1))

        for t in reversed(sample_steps):
            print(f'Sample timestep {t:4d}', end='\r')
            t = torch.full((b,), t, device=self.device, dtype=torch.long)

            # where to unmask
            changes_identity = torch.rand(
                x_identity_t.shape, device=self.device
            ) < 1 / t.float().unsqueeze(-1)
            # don't unmask somewhere already unmasked
            changes_identity = torch.bitwise_xor(
                changes_identity, torch.bitwise_and(changes_identity, unmasked_identity)
            )
            # update mask with changes
            unmasked_identity = torch.bitwise_or(unmasked_identity, changes_identity)

            changes_pose = torch.rand(
                x_pose_t.shape, device=self.device
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

        return x_identity_t, x_pose_t

    def sample_appearance(self, text, save_path, shape=[256, 128]):
        self.text = text
        self.image = torch.zeros([1, 3, shape[0], shape[1]]).to(self.device)
        self.get_text_embedding()

        with torch.no_grad():
            x_identity_t, x_pose_t = self.sample_fn(
                temp=1, sample_steps=self.sample_steps
            )

        self.get_vis_generated_only(x_identity_t, x_pose_t, save_path)

        self.quant_frame = self.quant_frame.view(
            x_pose_t.size(0), self.opt["img_z_channels"] // 2, -1
        ).permute(0, 2, 1)

        return self.quant_identity, self.quant_frame

    def get_vis_generated_only(self, quant_identity, quant_frame, save_path):
        # pred image
        pred_img = self.decode_image_indices(quant_identity, quant_frame)
        img_cat = (pred_img.detach() + 1) / 2
        img_cat = img_cat.clamp_(0, 1)
        save_image(img_cat, save_path, nrow=1, padding=4)
