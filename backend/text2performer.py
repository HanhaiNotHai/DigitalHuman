import os
import pickle
import shutil

import cv2
import numpy as np
import torch
from PIL import Image

from models import create_model
from models.app_transformer_model import AppTransformerModel
from models.video_transformer_model import VideoTransformerModel
from utils.options import parse


class Text2Performer:
    def __init__(self) -> None:
        # 初始化结果目录
        self.results_dir = './results'
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

        # 创建外观模型
        self.app_model: AppTransformerModel = create_model(
            parse('./configs/sampler/sampler_high_res.yml')
        )
        # 加载外观模型的网络
        self.app_model.load_network()

        # 创建运动模型
        self.motion_model: VideoTransformerModel = create_model(
            parse('./configs/video_transformer/video_trans_high_res.yml')
        )
        # 加载运动模型的网络
        self.motion_model.load_network()

    def save_num_appearance(self) -> None:
        with open(self.num_appearance_file, 'wb') as f:
            pickle.dump(self.num_appearance, f, pickle.HIGHEST_PROTOCOL)

    def load_raw_image(self, img_path: str, downsample: bool = False) -> Image.Image:
        """
        加载原始图像

        参数:
            img_path (str): 图像路径
            downsample (bool, optional): 是否进行下采样，默认为False

        返回:
            Image.Image: 加载的图像
        """
        with open(img_path, 'rb') as f:
            image = Image.open(f)
            if downsample:
                image = image.resize(
                    (image.width // 2, image.height // 2), resample=Image.LANCZOS
                )
            else:
                image = image.copy()

        return image

    def inter_sequence_inter(self, first_seq_idx: int, second_seq_idx: int) -> None:
        """两段之间插值"""
        video_embeddings_pred = torch.zeros([1, 8 * 32, 128]).to(
            self.motion_model.device
        )

        first_frame_path = f'{self.motion_dir}/sequence{first_seq_idx}/007.png'
        first_frame = self.load_raw_image(first_frame_path)
        first_frame = np.array(first_frame).transpose(2, 0, 1).astype(np.float32)
        first_frame = first_frame / 127.5 - 1
        first_frame = (
            torch.from_numpy(first_frame).unsqueeze(0).to(torch.device('cuda'))
        )

        first_frame_embedding = (
            self.motion_model.get_quantized_frame_embedding(first_frame)
            .view(1, self.motion_model.img_embed_dim // 2, -1)
            .permute(0, 2, 1)
            .contiguous()
        )

        video_embeddings_pred[:, :32, :] = first_frame_embedding

        end_frame_path = f'{self.motion_dir}/sequence{second_seq_idx}/000.png'
        end_frame = self.load_raw_image(end_frame_path)
        end_frame = np.array(end_frame).transpose(2, 0, 1).astype(np.float32)
        end_frame = end_frame / 127.5 - 1
        end_frame = torch.from_numpy(end_frame).unsqueeze(0).to(torch.device('cuda'))

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
            f'{self.motion_dir}/sequence{first_seq_idx}_{second_seq_idx}',
        )
        self.motion_model.refine_synthesized(
            self.x_identity,
            f'{self.motion_dir}/sequence{first_seq_idx}_{second_seq_idx}',
        )

    def intra_sequence_inter(self, seq_idx: int | str) -> None:
        """一段内插"""
        video_embeddings_pred = torch.zeros([1, 8 * 32, 128]).to(
            self.motion_model.device
        )

        for frame_idx in range(7):
            first_frame_path = (
                f'{self.motion_dir}/sequence{seq_idx}/{frame_idx:03d}.png'
            )
            first_frame = self.load_raw_image(first_frame_path)
            first_frame = np.array(first_frame).transpose(2, 0, 1).astype(np.float32)
            first_frame = first_frame / 127.5 - 1
            first_frame = (
                torch.from_numpy(first_frame).unsqueeze(0).to(torch.device('cuda'))
            )

            first_frame_embedding = (
                self.motion_model.get_quantized_frame_embedding(first_frame)
                .view(1, self.motion_model.img_embed_dim // 2, -1)
                .permute(0, 2, 1)
                .contiguous()
            )

            video_embeddings_pred[:, :32, :] = first_frame_embedding

            end_frame_path = (
                f'{self.motion_dir}/sequence{seq_idx}/{frame_idx+1:03d}.png'
            )
            end_frame = self.load_raw_image(end_frame_path)
            end_frame = np.array(end_frame).transpose(2, 0, 1).astype(np.float32)
            end_frame = end_frame / 127.5 - 1
            end_frame = (
                torch.from_numpy(end_frame).unsqueeze(0).to(torch.device('cuda'))
            )

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
                f'{self.motion_dir}/sequence{seq_idx}_interpolated',
                save_idx=list(range(frame_idx * 8, (frame_idx + 1) * 8)),
            )

        self.motion_model.refine_synthesized(
            self.x_identity, f'{self.motion_dir}/sequence{seq_idx}_interpolated'
        )

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
        self.x_identity, self.x_pose = self.app_model.sample_appearance(
            [input_appearance], appearance_file_name
        )
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

        images = []
        for seq_idx in range(self.num_input_motion - 1):
            print(f'{self.motion_dir}/sequence{seq_idx}{suf}')
            for frame_idx in range(56 if interpolate else 8):
                images.append(
                    f'{self.motion_dir}/sequence{seq_idx}{suf}/{frame_idx:03d}.png'
                )

            print(f'{self.motion_dir}/sequence{seq_idx}_{seq_idx + 1}{suf}')
            for frame_idx in range(56 if interpolate else 8):
                images.append(
                    f'{self.motion_dir}/sequence{seq_idx}_{seq_idx + 1}{suf}/{frame_idx:03d}.png'
                )
        print(f'{self.motion_dir}/sequence{self.num_input_motion-1}{suf}')
        for frame_idx in range(56 if interpolate else 8):
            images.append(
                f'{self.motion_dir}/sequence{self.num_input_motion-1}{suf}/{frame_idx:03d}.png'
            )
        num_images = len(images)

        all_frames_dir = f'{self.motion_dir}/all_frames{suf}'
        os.makedirs(all_frames_dir, exist_ok=True)
        for idx, image in enumerate(images):
            shutil.copy(image, f'{all_frames_dir}/{idx:03d}.png')

        target_dir = f'{self.motion_dir}/all_frames{suf}_stabilized'
        os.makedirs(target_dir, exist_ok=True)
        self.motion_model.video_stabilization(
            self.x_identity, all_frames_dir, target_dir, fix_video_len=num_images
        )

        video_file_name = f'{self.motion_dir}/video{suf}.mp4'

        images = []
        for i in range(num_images):
            images.append(f'{target_dir}/{i:03d}.png')

        frame = cv2.imread(images[0])
        height, width, _ = frame.shape
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video = cv2.VideoWriter(
            video_file_name, fourcc, 48 if interpolate else 8, (width, height)
        )

        for image in images:
            video.write(cv2.imread(image))

        video.release()

        h264video_file_name = f'{self.motion_dir}/video{suf}_h264.mp4'
        ffmpeg_log = f'{self.motion_dir}/ffmpeg{suf}.log'
        convert2h264cmd = (
            f'ffmpeg -y -i {video_file_name} {h264video_file_name} > {ffmpeg_log} 2>&1'
        )
        os.system(convert2h264cmd)

        return os.path.abspath(h264video_file_name)

    def generate_motion(self, input_motion: str) -> str:
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
                f'{self.motion_dir}/sequence{idx}',
            )
            # 对生成的视频进行细化
            self.motion_model.refine_synthesized(
                self.x_identity, f'{self.motion_dir}/sequence{idx}'
            )

        # 对相邻的两个序列进行交互
        for i in range(self.num_input_motion - 1):
            self.inter_sequence_inter(i, i + 1)

        # 生成最终的视频
        return self.generate_video(False)

    def interpolate(self) -> str:
        # 遍历输入运动序列
        for i in range(self.num_input_motion - 1):
            # 在序列中进行内部插值
            self.intra_sequence_inter(i)
            # 在序列中进行内部插值
            self.intra_sequence_inter(f'{i}_{i + 1}')
        # 在序列中进行内部插值
        self.intra_sequence_inter(self.num_input_motion - 1)

        # 生成视频
        return self.generate_video(True)
