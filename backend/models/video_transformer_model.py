import logging
import os
import random
from copy import deepcopy

import numpy as np
import torch
from PIL import Image
from sentence_transformers import SentenceTransformer
from torchvision.utils import save_image

from models.archs.dalle_transformer_arch import NonCausalTransformerLanguage
from models.archs.vqgan_arch import (
    DecoderUpOthersDoubleIdentity,
    EncoderDecomposeBaseDownOthersDoubleIdentity,
    VectorQuantizer,
)
from models.base_model import BaseModel
from utils.dist_util import master_only

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
        save_dir,
        save_idx=list(range(0, 8)),
        img_res=[512, 256],
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

        self.output_frames = self.output_frames.view(
            [1, self.fix_video_len, 3, img_res[0], img_res[1]]
        )

        for idx in range(batch_size):
            self.get_vis_generated_only_with_index(
                self.fix_video_len,
                self.output_frames[idx : idx + 1],
                save_dir,
                save_idx,
            )

        self.sampler.train()

    def load_raw_image(self, img_path, downsample=True):
        with open(img_path, 'rb') as f:
            image = Image.open(f)
            width, height = image.size
            image = image.resize(size=(width, height), resample=Image.LANCZOS)

        return image

    def refine_synthesized(
        self, x_identity, target_dir, fix_video_len=8, img_res=[512, 256]
    ):

        frames = []
        for i in range(fix_video_len):
            frame_path = f'{target_dir}/{i:03d}.png'
            frame = self.load_raw_image(frame_path, downsample=False)
            frame = np.array(frame).transpose(2, 0, 1).astype(np.float32)
            frame = frame / 127.5 - 1
            frames.append(torch.from_numpy(frame).unsqueeze(0).to(self.device))

        frames = torch.cat(frames, dim=0)

        with torch.no_grad():
            frames_embedding = self.get_quantized_frame_embedding(frames)

            identity_embeddings = x_identity.repeat(fix_video_len, 1, 1, 1)

            self.output_frames = self.decode(identity_embeddings, frames_embedding)

            self.output_frames = self.output_frames.view(
                [1, fix_video_len, 3, img_res[0], img_res[1]]
            )

        self.get_vis_generated_only(fix_video_len, self.output_frames[0:1], target_dir)

    def video_stabilization(
        self, x_identity, source_dir, target_dir, fix_video_len=728, img_res=[512, 256]
    ):

        for i in range(fix_video_len):
            if i < 2:
                frame_path = f'{source_dir}/{i:03d}.png'
                frame = self.load_raw_image(frame_path, downsample=False)
                frame = np.array(frame).transpose(2, 0, 1).astype(np.float32)
                frame = frame / 127.5 - 1
                frame_embedding = self.get_quantized_frame_embedding(
                    torch.from_numpy(frame).unsqueeze(0).to(self.device)
                )

            elif i > (fix_video_len - 3):
                frame_path = f'{source_dir}/{i:03d}.png'
                frame = self.load_raw_image(frame_path, downsample=False)
                frame = np.array(frame).transpose(2, 0, 1).astype(np.float32)
                frame = frame / 127.5 - 1
                frame_embedding = self.get_quantized_frame_embedding(
                    torch.from_numpy(frame).unsqueeze(0).to(self.device)
                )

            else:
                frame_path_1 = f'{source_dir}/{i - 2:03d}.png'
                frame_1 = self.load_raw_image(frame_path_1, downsample=False)
                frame_1 = np.array(frame_1).transpose(2, 0, 1).astype(np.float32)
                frame_1 = frame_1 / 127.5 - 1
                frame_1 = (
                    torch.from_numpy(frame_1).unsqueeze(0).to(torch.device('cuda'))
                )

                frame_embedding_1 = self.get_quantized_frame_embedding(frame_1)

                frame_path_2 = f'{source_dir}/{i - 1:03d}.png'
                frame_2 = self.load_raw_image(frame_path_2, downsample=False)
                frame_2 = np.array(frame_2).transpose(2, 0, 1).astype(np.float32)
                frame_2 = frame_2 / 127.5 - 1
                frame_2 = (
                    torch.from_numpy(frame_2).unsqueeze(0).to(torch.device('cuda'))
                )

                frame_embedding_2 = self.get_quantized_frame_embedding(frame_2)

                frame_path_3 = f'{source_dir}/{i:03d}.png'
                frame_3 = self.load_raw_image(frame_path_3, downsample=False)
                frame_3 = np.array(frame_3).transpose(2, 0, 1).astype(np.float32)
                frame_3 = frame_3 / 127.5 - 1
                frame_3 = (
                    torch.from_numpy(frame_3).unsqueeze(0).to(torch.device('cuda'))
                )

                frame_embedding_3 = self.get_quantized_frame_embedding(frame_3)

                frame_path_4 = f'{source_dir}/{i+1:03d}.png'
                frame_4 = self.load_raw_image(frame_path_4, downsample=False)
                frame_4 = np.array(frame_4).transpose(2, 0, 1).astype(np.float32)
                frame_4 = frame_4 / 127.5 - 1
                frame_4 = (
                    torch.from_numpy(frame_4).unsqueeze(0).to(torch.device('cuda'))
                )

                frame_embedding_4 = self.get_quantized_frame_embedding(frame_4)

                frame_path_5 = f'{source_dir}/{i+2:03d}.png'
                frame_5 = self.load_raw_image(frame_path_5, downsample=False)
                frame_5 = np.array(frame_5).transpose(2, 0, 1).astype(np.float32)
                frame_5 = frame_5 / 127.5 - 1
                frame_5 = (
                    torch.from_numpy(frame_5).unsqueeze(0).to(torch.device('cuda'))
                )

                frame_embedding_5 = self.get_quantized_frame_embedding(frame_5)

                frame_embedding = (
                    frame_embedding_1
                    + frame_embedding_2
                    + frame_embedding_3
                    + frame_embedding_4
                    + frame_embedding_5
                ) / 5.0

            with torch.no_grad():
                self.output_frames = self.decode(x_identity, frame_embedding)

                self.output_frames = self.output_frames.view(
                    [1, 1, 3, img_res[0], img_res[1]]
                )

            self.get_vis_generated_with_file_name(
                1, self.output_frames[0:1], f'{target_dir}/{i:03d}.png'
            )

    @master_only
    def get_vis_generated_with_file_name(self, video_len, pred_frames, save_path):
        for frame_idx in range(video_len):
            pred_img = (pred_frames[:, frame_idx] + 1) / 2
            pred_img = pred_img.clamp_(0, 1)
            save_image(pred_img, save_path, nrow=1, padding=4)

    @master_only
    def get_vis_generated_only_with_index(
        self, video_len, pred_frames, save_path, save_index
    ):
        os.makedirs(save_path, exist_ok=True)

        for idx, frame_idx in enumerate(range(video_len - len(save_index), video_len)):
            pred_img = (pred_frames[:, frame_idx] + 1) / 2
            pred_img = pred_img.clamp_(0, 1)
            save_image(
                pred_img, f'{save_path}/{save_index[idx]:03d}.png', nrow=1, padding=4
            )

    @master_only
    def get_vis_generated_only(self, video_len, pred_frames, save_path):
        os.makedirs(save_path, exist_ok=True)
        for frame_idx in range(video_len):
            pred_img = (pred_frames[:, frame_idx] + 1) / 2
            pred_img = pred_img.clamp_(0, 1)
            save_image(pred_img, f'{save_path}/{frame_idx:03d}.png', nrow=1, padding=4)

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
