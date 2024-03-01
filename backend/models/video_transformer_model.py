import logging
import random
from collections import defaultdict, deque
from copy import deepcopy

import torch
from sentence_transformers import SentenceTransformer
from torch import Tensor

from models.archs.dalle_transformer_arch import NonCausalTransformerLanguage
from models.archs.vqgan_arch import (
    DecoderUpOthersDoubleIdentity,
    EncoderDecomposeBaseDownOthersDoubleIdentity,
    VectorQuantizer,
)
from models.base_model import BaseModel

logger = logging.getLogger('base')


class VideoTransformerModel(BaseModel):

    def __init__(self, opt):
        super().__init__(opt)

        # VQVAE for image
        self.img_encoder = self.model_to_device(
            EncoderDecomposeBaseDownOthersDoubleIdentity(
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
            )
        )
        self.img_decoder = self.model_to_device(
            DecoderUpOthersDoubleIdentity(
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
            )
        )
        self.img_quantize_identity = self.model_to_device(
            VectorQuantizer(opt['img_n_embed'], opt['img_embed_dim'], beta=0.25)
        )
        self.img_quant_conv_identity = self.model_to_device(
            torch.nn.Conv2d(opt["img_z_channels"], opt['img_embed_dim'], 1)
        )
        self.img_post_quant_conv_identity = self.model_to_device(
            torch.nn.Conv2d(opt['img_embed_dim'], opt["img_z_channels"], 1)
        )

        self.img_quantize_others = self.model_to_device(
            VectorQuantizer(opt['img_n_embed'], opt['img_embed_dim'] // 2, beta=0.25)
        )
        self.img_quant_conv_others = self.model_to_device(
            torch.nn.Conv2d(opt["img_z_channels"] // 2, opt['img_embed_dim'] // 2, 1)
        )
        self.img_post_quant_conv_others = self.model_to_device(
            torch.nn.Conv2d(opt['img_embed_dim'] // 2, opt["img_z_channels"] // 2, 1)
        )
        self.load_pretrained_image_vae()

        # define sampler
        self.sampler = self.model_to_device(
            NonCausalTransformerLanguage(
                dim=opt['dim'],
                depth=opt['depth'],
                dim_head=opt['dim_head'],
                heads=opt['heads'],
                ff_mult=opt['ff_mult'],
                norm_out=opt['norm_out'],
                attn_dropout=opt['attn_dropout'],
                ff_dropout=opt['ff_dropout'],
                final_proj=opt['final_proj'],
                normformer=opt['normformer'],
                rotary_emb=opt['rotary_emb'],
            )
        )

        self.shape = tuple(opt['latent_shape'])
        self.single_len = self.shape[0] * self.shape[1]

        self.img_embed_dim = opt['img_embed_dim']

        self.num_inside_timesteps = opt['num_inside_timesteps']

        self.get_fixed_language_model()

        self.output_dict: defaultdict[str, list[Tensor]] = defaultdict(list)

    def save_output_frames(
        self, output_frames: Tensor, save_key: int | str, idx: int | None = None
    ) -> None:
        output_frames = ((output_frames + 1) / 2).clamp(0, 1)
        if idx is None:
            # TODO 如果要refine_synthesized interpolated记得改
            if save_key.endswith('_interpolated'):
                self.output_dict[f'{save_key}'] += [
                    output_frames[i : i + 1] for i in range(output_frames.shape[0])
                ]
            else:
                self.output_dict[f'{save_key}'] = [
                    output_frames[i : i + 1] for i in range(output_frames.shape[0])
                ]
        else:
            self.output_dict[f'{save_key}'][idx] = output_frames

    def load_pretrained_image_vae(self):
        # load pretrained vqgan for segmentation mask
        img_ae_checkpoint = torch.load(
            self.opt['img_ae_path'],
            map_location=lambda storage, loc: storage.cuda(torch.cuda.current_device()),
        )
        self.get_bare_model(self.img_encoder).load_state_dict(
            img_ae_checkpoint['encoder'], strict=True
        )
        self.get_bare_model(self.img_decoder).load_state_dict(
            img_ae_checkpoint['decoder'], strict=True
        )

        self.get_bare_model(self.img_quantize_identity).load_state_dict(
            img_ae_checkpoint['quantize_identity'], strict=True
        )
        self.get_bare_model(self.img_quant_conv_identity).load_state_dict(
            img_ae_checkpoint['quant_conv_identity'], strict=True
        )
        self.get_bare_model(self.img_post_quant_conv_identity).load_state_dict(
            img_ae_checkpoint['post_quant_conv_identity'], strict=True
        )

        self.get_bare_model(self.img_quantize_others).load_state_dict(
            img_ae_checkpoint['quantize_others'], strict=True
        )
        self.get_bare_model(self.img_quant_conv_others).load_state_dict(
            img_ae_checkpoint['quant_conv_others'], strict=True
        )
        self.get_bare_model(self.img_post_quant_conv_others).load_state_dict(
            img_ae_checkpoint['post_quant_conv_others'], strict=True
        )

        self.img_encoder.eval()
        self.img_decoder.eval()

        self.img_quantize_identity.eval()
        self.img_quant_conv_identity.eval()
        self.img_post_quant_conv_identity.eval()

        self.img_quantize_others.eval()
        self.img_quant_conv_others.eval()
        self.img_post_quant_conv_others.eval()

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

    @torch.no_grad()
    def get_quantized_frame_embedding(self, image):
        _, frame_embeddings = self.img_encoder(image)
        frame_embeddings = self.img_quant_conv_others(frame_embeddings)
        frame_embeddings, _, _ = self.img_quantize_others(frame_embeddings)

        return frame_embeddings

    def decode(self, identity_embeddings, frame_embeddings):
        quant_identity = self.img_post_quant_conv_identity(identity_embeddings)
        quant_frame = self.img_post_quant_conv_others(frame_embeddings)
        dec = self.img_decoder(quant_identity, quant_frame)
        return dec

    def sample_first_last(self, video_embeddings_pred, mask):
        sample_inside_steps = list(range(1, self.num_inside_timesteps + 1))
        # unmasked = torch.zeros(video_embeddings_pred.size(0),
        #    self.single_len).bool().to(self.device)
        unmasked_full = (~mask).clone()

        unmasked = unmasked_full[:, : self.single_len]

        for t_inside in reversed(sample_inside_steps):
            # where to unmask
            t_inside = torch.full(
                (video_embeddings_pred.size(0),), t_inside, dtype=torch.long
            ).to(self.device)

            changes = torch.rand(unmasked.shape).to(self.device) < (
                1.0 / t_inside.float().unsqueeze(-1)
            )

            # don't unmask somewhere already unmasked
            changes = torch.bitwise_xor(changes, torch.bitwise_and(changes, unmasked))

            # update mask with changes
            unmasked = torch.bitwise_or(unmasked, changes)

            unmasked_full_temp = torch.zeros(unmasked_full.shape).bool().to(self.device)
            unmasked_full_temp[:, : self.single_len] = changes
            # unmasked_full[:, -self.single_len:] = unmasked

            with torch.no_grad():
                temp_embeddings = self.sampler(
                    video_embeddings_pred,
                    self.exemplar_frame_embeddings,
                    self.text_embedding,
                    mask,
                )[:, 1 + self.single_len :]
                temp_nearest_codebook_embeddings = self.get_bare_model(
                    self.img_quantize_others
                ).get_nearest_codebook_embeddings(temp_embeddings)
                video_embeddings_pred[unmasked_full_temp, :] = (
                    temp_nearest_codebook_embeddings[unmasked_full_temp, :]
                )

            # update mask
            # mask = ~unmasked_full

        unmasked = (
            torch.zeros(video_embeddings_pred.size(0), self.single_len)
            .bool()
            .to(self.device)
        )
        for t_inside in reversed(sample_inside_steps):
            # where to unmask
            t_inside = torch.full(
                (video_embeddings_pred.size(0),), t_inside, dtype=torch.long
            ).to(self.device)

            changes = torch.rand(unmasked.shape).to(self.device) < (
                1.0 / t_inside.float().unsqueeze(-1)
            )

            # don't unmask somewhere already unmasked
            changes = torch.bitwise_xor(changes, torch.bitwise_and(changes, unmasked))

            # update mask with changes
            unmasked = torch.bitwise_or(unmasked, changes)

            # unmasked_full[:, :self.single_len] = unmasked
            unmasked_full_temp = torch.zeros(unmasked_full.shape).bool().to(self.device)
            unmasked_full_temp[:, -self.single_len :] = changes
            # print(unmasked_full_temp.sum())

            with torch.no_grad():
                temp_embeddings = self.sampler(
                    video_embeddings_pred,
                    self.exemplar_frame_embeddings,
                    self.text_embedding,
                    mask,
                )[:, 1 + self.single_len :]
                temp_nearest_codebook_embeddings = self.get_bare_model(
                    self.img_quantize_others
                ).get_nearest_codebook_embeddings(temp_embeddings)
                video_embeddings_pred[unmasked_full_temp, :] = (
                    temp_nearest_codebook_embeddings[unmasked_full_temp, :]
                )

            # update mask
            # mask = ~unmasked_full

        return video_embeddings_pred

    def sample_multinomial_text_embeddings(
        self,
        identity_embeddings,
        exemplar_frame_embeddings,
        text,
        fix_video_len,
        masked_index,
        video_embeddings_pred,
        save_key,
    ):
        # sample with the first frame given
        self.sampler.eval()

        self.text = text

        self.get_text_embedding()

        batch_size = exemplar_frame_embeddings.size(0)
        self.exemplar_frame_embeddings = exemplar_frame_embeddings.clone()

        sample_steps = list(range(1, fix_video_len // 2 + 1))

        self.fix_video_len = fix_video_len

        for t in reversed(sample_steps):
            mask = torch.zeros((self.fix_video_len,)).to(self.device)
            for idx in masked_index:
                mask[idx] = 1
            mask = mask.bool()
            mask = mask.unsqueeze(1).repeat(1, self.single_len).view(-1).unsqueeze(0)

            if t == self.fix_video_len // 2:
                video_embeddings_pred = self.sample_first_last(
                    video_embeddings_pred, mask
                )

            # where to unmask
            if t == self.fix_video_len // 2:
                unmask_list = []

                try:
                    masked_index.remove(0)
                    unmask_list.append(0)
                except:
                    pass

                try:
                    masked_index.remove(self.fix_video_len - 1)
                    unmask_list.append(self.fix_video_len - 1)
                except:
                    pass
            else:
                unmask_list = []
                try:
                    unmask_idx = random.choice(masked_index)
                    masked_index.remove(unmask_idx)
                    unmask_list.append(unmask_idx)
                except:
                    pass

                try:
                    unmask_idx = random.choice(masked_index)
                    masked_index.remove(unmask_idx)
                    unmask_list.append(unmask_idx)
                except:
                    pass

            if t == self.fix_video_len // 2:
                continue

            if len(unmask_list) == 0:
                continue

            unmask = torch.zeros((self.fix_video_len,)).to(self.device)
            for idx in unmask_list:
                unmask[idx] = 1
            unmask = unmask.bool()
            unmask = (
                unmask.unsqueeze(1).repeat(1, self.single_len).view(-1).unsqueeze(0)
            )

            with torch.no_grad():
                temp_embeddings = self.sampler(
                    video_embeddings_pred,
                    exemplar_frame_embeddings,
                    self.text_embedding,
                    mask,
                )[:, 1 + self.single_len :]
                temp_nearest_codebook_embeddings = self.get_bare_model(
                    self.img_quantize_others
                ).get_nearest_codebook_embeddings(temp_embeddings)
                video_embeddings_pred[unmask, :] = temp_nearest_codebook_embeddings[
                    unmask, :
                ]

        with torch.no_grad():
            self.nearest_codebook_embeddings = (
                self.get_bare_model(self.img_quantize_others)
                .get_nearest_codebook_embeddings(video_embeddings_pred)
                .view(
                    (
                        batch_size * self.fix_video_len,
                        self.shape[0],
                        self.shape[1],
                        self.img_embed_dim // 2,
                    )
                )
                .permute(0, 3, 1, 2)
                .contiguous()
            )

            self.identity_embeddings = identity_embeddings.repeat(
                self.fix_video_len, 1, 1, 1
            )
            self.output_frames = self.decode(
                self.identity_embeddings, self.nearest_codebook_embeddings
            )

        self.save_output_frames(self.output_frames, save_key)

        self.sampler.train()

    # TODO 有必要？fix_video_len=8？
    def refine_synthesized(self, x_identity, target_key, fix_video_len=8):
        frames = torch.cat(self.output_dict[target_key], dim=0)

        with torch.no_grad():
            frames_embedding = self.get_quantized_frame_embedding(frames)

            identity_embeddings = x_identity.repeat(fix_video_len, 1, 1, 1)

            self.output_frames = self.decode(identity_embeddings, frames_embedding)

        self.save_output_frames(self.output_frames, target_key)

    def video_stabilization(self, x_identity, num_input_motion, suf):
        for i in range(num_input_motion - 1):
            self.output_dict[f'all{suf}'] += self.output_dict[f'{i}{suf}']
            self.output_dict[f'all{suf}'] += self.output_dict[f'{i}_{i + 1}{suf}']
        self.output_dict[f'all{suf}'] += self.output_dict[
            f'{num_input_motion - 1}{suf}'
        ]

        frames = torch.cat(self.output_dict[f'all{suf}'][:4], dim=0)
        frames_embedding = self.get_quantized_frame_embedding(frames)
        frame_embeddings = deque(
            [frames_embedding[i : i + 1] for i in range(frames_embedding.shape[0])]
        )
        frame_embeddings.appendleft(torch.zeros_like(frame_embeddings[0]))
        sum_frame_embeddings = sum(frame_embeddings)

        for idx, output_fram in enumerate(self.output_dict[f'all{suf}'][2:-2], 2):
            sum_frame_embeddings -= frame_embeddings.popleft()
            frame_embeddings.append(self.get_quantized_frame_embedding(output_fram))
            sum_frame_embeddings += frame_embeddings[-1]
            frame_embeddings[2] = sum_frame_embeddings / 5.0
            with torch.no_grad():
                self.output_frame = self.decode(x_identity, frame_embeddings[2])
            self.save_output_frames(self.output_frame, f'all{suf}', idx)

    def load_network(self):
        checkpoint = torch.load(
            self.opt['pretrained_sampler'],
            map_location=lambda storage, loc: storage.cuda(torch.cuda.current_device()),
        )
        # remove unnecessary 'module.'
        for k, v in deepcopy(checkpoint).items():
            if k.startswith('module.'):
                checkpoint[k[7:]] = v
                checkpoint.pop(k)

        self.get_bare_model(self.sampler).load_state_dict(checkpoint, strict=True)
        self.get_bare_model(self.sampler).eval()
