import os
import pickle
from collections import defaultdict

import cv2
import numpy as np
import torch
from sentence_transformers import SentenceTransformer

from .models.app_transformer_model import AppTransformerModel
from .models.video_transformer_model import VideoTransformerModel
from .models.vqgan_decompose_model import VQGANDecomposeModel
from .utils.options import parse


class Text2Performer:
    def __init__(self) -> None:
        # 初始化结果目录
        self.results_dir = './text2performer/results'
        # 如果结果目录不存在，则创建
        os.makedirs(self.results_dir, exist_ok=True)

        # 初始化外观文件数量
        self.num_appearance_file = f'{self.results_dir}/num_appearance'
        # 如果外观文件存在，则加载文件中的外观数量
        if os.path.exists(self.num_appearance_file):
            with open(self.num_appearance_file, 'rb') as f:
                self.num_appearance = pickle.load(f)
        else:
            # 如果外观文件不存在，则初始化外观数量为-1，并保存到文件中
            self.num_appearance = -1
            self.save_num_appearance()

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        vq_decompose_model = VQGANDecomposeModel(
            parse('./text2performer/configs/vqgan/vqgan_decompose_high_res.yml')
        )

        language_model = SentenceTransformer('all-MiniLM-L6-v2')

        # 创建外观模型
        self.app_model = AppTransformerModel(
            parse('./text2performer/configs/sampler/sampler_high_res.yml'),
            vq_decompose_model,
            language_model,
            device,
        )

        # 创建运动模型
        self.motion_model = VideoTransformerModel(
            parse(
                './text2performer/configs/video_transformer/video_trans_high_res.yml'
            ),
            vq_decompose_model,
            language_model,
            device,
        )

        torch.cuda.empty_cache()

    def save_num_appearance(self) -> None:
        with open(self.num_appearance_file, 'wb') as f:
            pickle.dump(self.num_appearance, f, pickle.HIGHEST_PROTOCOL)

    def inter_sequence_inter(self, first_seq_idx: int, second_seq_idx: int) -> None:
        """两段之间插值"""
        video_embeddings_pred = torch.zeros([1, 8 * 32, 128]).to(
            self.motion_model.device
        )

        first_frame = self.motion_model.output_dict[f'{first_seq_idx}'][-1]
        first_frame_embedding = (
            self.motion_model.get_quantized_frame_embedding(first_frame)
            .view(1, self.motion_model.img_embed_dim // 2, -1)
            .permute(0, 2, 1)
            .contiguous()
        )

        video_embeddings_pred[:, :32, :] = first_frame_embedding

        end_frame = self.motion_model.output_dict[f'{second_seq_idx}'][0]
        end_frame_embedding = (
            self.motion_model.get_quantized_frame_embedding(end_frame)
            .view(1, self.motion_model.img_embed_dim // 2, -1)
            .permute(0, 2, 1)
            .contiguous()
        )

        video_embeddings_pred[:, -32:, :] = end_frame_embedding

        self.motion_model.sample_multinomial_text_embeddings(
            self.x_identity,
            self.x_pose,
            ['empty'],
            8,
            list(range(1, 7)),
            video_embeddings_pred,
            f'{first_seq_idx}_{second_seq_idx}',
        )
        self.motion_model.refine_synthesized(
            self.x_identity,
            f'{first_seq_idx}_{second_seq_idx}',
        )

    def intra_sequence_inter(self, seq_idx: int | str) -> None:
        """一段内插"""
        video_embeddings_pred = torch.zeros([1, 8 * 32, 128]).to(
            self.motion_model.device
        )

        for frame_idx in range(7):
            first_frame = self.motion_model.output_dict[f'{seq_idx}'][frame_idx]

            first_frame_embedding = (
                self.motion_model.get_quantized_frame_embedding(first_frame)
                .view(1, self.motion_model.img_embed_dim // 2, -1)
                .permute(0, 2, 1)
                .contiguous()
            )

            video_embeddings_pred[:, :32, :] = first_frame_embedding

            end_frame = self.motion_model.output_dict[f'{seq_idx}'][frame_idx + 1]

            end_frame_embedding = (
                self.motion_model.get_quantized_frame_embedding(end_frame)
                .view(1, self.motion_model.img_embed_dim // 2, -1)
                .permute(0, 2, 1)
                .contiguous()
            )

            video_embeddings_pred[:, -32:, :] = end_frame_embedding

            self.motion_model.sample_multinomial_text_embeddings(
                self.x_identity,
                self.x_pose,
                ['empty'],
                8,
                list(range(1, 7)),
                video_embeddings_pred,
                f'{seq_idx}_interpolated',
            )

        # TODO 爆显存 总共56张 但是原代码只搞8张 这一步有必要？如果要的话记得改save_output_frames
        # self.motion_model.refine_synthesized(self.x_identity, f'{seq_idx}_interpolated')

    @torch.no_grad()
    def generate_appearance(self, input_appearance: str) -> str:
        # 增加外观数量
        self.num_appearance += 1
        # 设置运动数量为-1
        self.num_motion = -1
        # 保存外观数量
        self.save_num_appearance()
        # 设置外观目录
        self.appearance_dir = f'{self.results_dir}/appearance{self.num_appearance:03d}'
        # 创建外观目录（如果目录不存在）
        os.makedirs(self.appearance_dir, exist_ok=True)

        # 设置外观文件名
        appearance_file_name = f'{self.appearance_dir}/exampler.png'
        # 从模型中采样外观
        self.app_model.to('cuda')
        self.x_identity, self.x_pose = self.app_model.sample_appearance(
            [input_appearance], appearance_file_name
        )
        self.app_model.to('cpu')
        # 返回外观文件的绝对路径
        return os.path.abspath(appearance_file_name)

    def generate_video(self, interpolate: bool) -> str:
        """
        生成视频

        参数：
        interpolate(bool): 是否插值

        返回：
        str: 视频文件的绝对路径
        """
        suf = '_interpolated' if interpolate else ''

        self.motion_model.video_stabilization(
            self.x_identity,
            self.num_input_motion,
            suf,
        )

        video_file_name = f'{self.motion_dir}/video{suf}.mp4'

        frame = np.array(
            np.around(
                self.motion_model.output_dict[f'all{suf}'][0]
                .squeeze(dim=0)[[2, 1, 0]]  # RGB => BGR
                .permute(1, 2, 0)  # [c, height, width] => [height, width, c]
                .cpu()
                * 255  # [0, 1] => [0, 255]
            ),
            dtype=np.uint8,
        )
        height, width, _ = frame.shape
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video = cv2.VideoWriter(
            video_file_name, fourcc, 48 if interpolate else 8, (width, height)
        )

        for output_frame in self.motion_model.output_dict[f'all{suf}']:
            video.write(
                np.array(
                    np.around(
                        output_frame.squeeze(dim=0)[[2, 1, 0]]  # RGB => BGR
                        .permute(1, 2, 0)  # [c, height, width] => [height, width, c]
                        .cpu()
                        * 255  # [0, 1] => [0, 255]
                    ),
                    dtype=np.uint8,
                )
            )

        video.release()

        h264video_file_name = f'{self.motion_dir}/video{suf}_h264.mp4'
        ffmpeg_log = f'{self.motion_dir}/ffmpeg{suf}.log'
        convert2h264cmd = (
            f'ffmpeg -y -i {video_file_name} {h264video_file_name} > {ffmpeg_log} 2>&1'
        )
        os.system(convert2h264cmd)

        return os.path.abspath(h264video_file_name)

    @torch.no_grad()
    def generate_motion(self, input_motion: str) -> str:
        self.motion_model.to('cuda')
        self.motion_model.output_dict = defaultdict(list)

        # 生成运动视频
        self.num_motion += 1
        self.motion_dir = f'{self.appearance_dir}/motion{self.num_motion:03d}'
        os.makedirs(self.motion_dir, exist_ok=True)

        # 将输入的运动字符串按行分割
        input_motion_lines = input_motion.splitlines()
        self.num_input_motion = len(input_motion_lines)

        # 对每一行运动进行处理
        for idx, motion in enumerate(input_motion_lines):
            # 使用multinomial_text_embeddings方法生成视频嵌入
            video_embeddings_pred = torch.zeros([1, 8 * 32, 128]).to(
                self.motion_model.device
            )
            self.motion_model.sample_multinomial_text_embeddings(
                self.x_identity,
                self.x_pose,
                [motion],
                8,
                list(range(0, 8)),
                video_embeddings_pred,
                f'{idx}',
            )
            # 对生成的视频进行细化
            self.motion_model.refine_synthesized(self.x_identity, f'{idx}')

        # 对相邻的两个序列进行交互
        for i in range(self.num_input_motion - 1):
            self.inter_sequence_inter(i, i + 1)

        # 生成最终的视频
        video_path = self.generate_video(False)
        self.motion_model.to('cpu')
        return video_path

    @torch.no_grad()
    def interpolate(self) -> str:
        self.motion_model.to('cuda')

        # 遍历输入运动序列
        for i in range(self.num_input_motion - 1):
            # 在序列中进行内部插值
            self.intra_sequence_inter(i)
            # 在序列中进行内部插值
            self.intra_sequence_inter(f'{i}_{i + 1}')
        # 在序列中进行内部插值
        self.intra_sequence_inter(self.num_input_motion - 1)

        # 生成视频
        video_path = self.generate_video(True)
        self.motion_model.to('cpu')
        return video_path


text2performer = Text2Performer()
