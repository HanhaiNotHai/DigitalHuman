import gradio as gr
import numpy as np
import torch
from PIL import Image, ImageOps

from magicanimate import MagicAnimate
from styleganhuman import StyleGANHuman
from text2performer import Text2Performer

style_gan_human = StyleGANHuman()
text2performer = Text2Performer()
magic_animate = MagicAnimate()

appearance_placeholder = 'The dress the person wears has long sleeves and it is of short length. Its texture is pure color.'
motion_placeholder = '''The lady moves to the right.
The person is moving to the center from the right.
She turns right from the front to the side.
She turns right from the side to the back.'''


def expand(image: np.ndarray) -> Image.Image:
    return ImageOps.expand(
        Image.fromarray(image), border=128, fill=(255, 255, 255)
    ).crop((0, 128, 512, 640))


with gr.Blocks() as demo:
    gr.HTML(
        '<p align="center" style="font-size:33px">基于深度学习的可控数字人生成平台</p>'
    )

    with gr.Tab('生成数字人形象'):
        with gr.Row():
            ########## StyleGAN-Human ##########
            with gr.Column():
                seed = gr.Slider(
                    label='随机种子', minimum=0, maximum=65535, step=1, value=0
                )
                truncation_psi = gr.Slider(
                    label='Truncation psi', minimum=0, maximum=2, step=0.05, value=0.7
                )
                style_gan_human_generate_image = gr.Button('随机生成数字人形象')
                style_gan_human2motion = gr.Button('用于驱动数字人')
            style_gan_human_appearence = gr.Image(
                'styleganhuman/example/image.png', label='数字人形象', interactive=True
            )

            ########## text2performer ##########
            with gr.Column():
                text2performer_appearence_input = gr.Textbox(
                    appearance_placeholder,
                    lines=3,
                    placeholder=appearance_placeholder,
                    label='数字人形象描述(English)',
                    interactive=True,
                )
                text2performer_generate_appearance = gr.Button('生成数字人形象')
                text2performer2motion = gr.Button('用于驱动数字人')
            text2performer_appearance = gr.Image(
                'text2performer/example/exampler.png',
                label='数字人形象',
                interactive=True,
            )

    with gr.Tab('驱动数字人'):
        with gr.Row():
            text2performer_motion_input_appearence = gr.Image(
                'text2performer/example/exampler.png',
                label='数字人形象',
                interactive=True,
            )
            with gr.Column():
                text2performer_motion_input_text = gr.Textbox(
                    motion_placeholder,
                    lines=5,
                    placeholder=motion_placeholder,
                    label='数字人动作描述(English)',
                    interactive=True,
                )
                text2performer_generate_motion = gr.Button('驱动数字人')
                text2performer_interpolate = gr.Button('视频插帧')
            text2performer_motion = gr.Video(
                'text2performer/example/video.mp4',
                height=512,
                width=256,
                label='数字人动作',
                autoplay=True,
            )

        ########## magic animate ##########
        with gr.Row():
            magic_animate_input_appearence = gr.Image(
                'magicanimate/inputs/applications/source_image/monalisa.png',
                label='数字人形象',
                interactive=True,
            )
            magic_animate_input_motion_sequence = gr.Video(
                'magicanimate/inputs/applications/driving/densepose/running.mp4',
                format='mp4',
                label='动作序列(从下面选择)',
            )

            with gr.Column():
                random_seed = gr.Slider(
                    label='随机种子', minimum=0, maximum=65535, step=1, value=0
                )
                sampling_steps = gr.Slider(
                    label='采样步数', minimum=1, maximum=100, step=1, value=25
                )
                guidance_scale = gr.Slider(
                    label='Guidance scale', minimum=0, maximum=10, step=0.1, value=7.5
                )
                magic_animate_generate_motion = gr.Button('驱动数字人')

        gr.Examples(
            examples=[
                'magicanimate/inputs/applications/driving/densepose/running.mp4',
                'magicanimate/inputs/applications/driving/densepose/demo4.mp4',
                'magicanimate/inputs/applications/driving/densepose/running2.mp4',
                'magicanimate/inputs/applications/driving/densepose/dancing2.mp4',
                'magicanimate/inputs/applications/driving/densepose/multi_dancing.mp4',
            ],
            inputs=magic_animate_input_motion_sequence,
            label='动作序列',
        )

        magic_animate_animation = gr.Video(
            'magicanimate/inputs/example.mp4',
            format='mp4',
            label='数字人动作',
            autoplay=True,
        )

    empty_cache = gr.Button('清空缓存')

    ########## StyleGAN-Human ##########
    style_gan_human_generate_image.click(
        style_gan_human.generate_image,
        [seed, truncation_psi],
        style_gan_human_appearence,
    )
    style_gan_human2motion.click(
        expand,
        style_gan_human_appearence,
        magic_animate_input_appearence,
    )

    ########## text2performer ##########
    text2performer_generate_appearance.click(
        text2performer.generate_appearance,
        text2performer_appearence_input,
        text2performer_appearance,
    )
    text2performer2motion.click(
        lambda image: (image, expand(image)),
        text2performer_appearance,
        [text2performer_motion_input_appearence, magic_animate_input_appearence],
    )

    text2performer_generate_motion.click(
        text2performer.generate_motion,
        text2performer_motion_input_text,
        text2performer_motion,
    )
    text2performer_interpolate.click(
        text2performer.interpolate,
        None,
        text2performer_motion,
    )

    ########## magic animate ##########
    # when `first_frame` is updated
    magic_animate_input_appearence.upload(
        lambda image: Image.fromarray(image).resize((512, 512)),
        magic_animate_input_appearence,
        magic_animate_input_appearence,
    )
    # when the `submit` button is clicked
    magic_animate_generate_motion.click(
        magic_animate,
        [
            magic_animate_input_appearence,
            magic_animate_input_motion_sequence,
            random_seed,
            sampling_steps,
            guidance_scale,
        ],
        magic_animate_animation,
    )

    empty_cache.click(torch.cuda.empty_cache)

demo.launch()
