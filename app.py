import gradio as gr
import numpy as np
from PIL import Image

from magicanimate import MagicAnimate
from text2performer import Text2Performer

text2performer = Text2Performer()
animator = MagicAnimate()

input_appearance = 'The dress the person wears has long sleeves and it is of short length. Its texture is pure color.'
input_motion = '''The lady moves to the right.
The person is moving to the center from the right.
She turns right from the front to the side.
She turns right from the side to the back.'''

with gr.Blocks() as demo:
    ########## text2performer ##########
    with gr.Row():
        with gr.Column():
            input_appearance = gr.Textbox(
                input_appearance,
                placeholder=input_appearance,
                label='外貌',
                interactive=True,
            )
            generate_appearance = gr.Button('生成外貌')
        appearance = gr.Image(
            'text2performer/example/exampler.png',
            interactive=True,
        )

        with gr.Column():
            input_motion = gr.Textbox(
                input_motion,
                lines=5,
                placeholder=input_motion,
                label='动作',
                interactive=True,
            )
            generate_motion = gr.Button('生成动作')
            interpolate = gr.Button('插帧')
        motion = gr.Video(
            'text2performer/example/video.mp4',
            height=512,
            width=256,
            autoplay=True,
        )

    generate_appearance.click(
        text2performer.generate_appearance,
        input_appearance,
        appearance,
    )
    generate_motion.click(
        text2performer.generate_motion,
        input_motion,
        motion,
    )
    interpolate.click(
        text2performer.interpolate,
        None,
        motion,
    )

    text2performer2magicanimate = gr.Button('text2performer -> magic animate')

    ########## magic animate ##########
    with gr.Row():
        reference_image = gr.Image(label="Reference Image")
        motion_sequence = gr.Video(format="mp4", label="Motion Sequence")

        with gr.Column():
            random_seed = gr.Textbox(label="Random seed", value=1, info="default: -1")
            sampling_steps = gr.Textbox(
                label="Sampling steps", value=25, info="default: 25"
            )
            guidance_scale = gr.Textbox(
                label="Guidance scale", value=7.5, info="default: 7.5"
            )
            submit = gr.Button("Animate")

    animation = gr.Video(format="mp4", label="Animation Results", autoplay=True)

    text2performer2magicanimate.click(
        lambda image: Image.fromarray(image).crop((0, 0, 256, 256)).resize((512, 512)),
        appearance,
        reference_image,
    )
    # when `first_frame` is updated
    reference_image.upload(
        lambda image: Image.fromarray(image).resize((512, 512)),
        reference_image,
        reference_image,
    )
    # when the `submit` button is clicked
    submit.click(
        animator,
        [reference_image, motion_sequence, random_seed, sampling_steps, guidance_scale],
        animation,
    )

    # Examples
    gr.Markdown("## Examples")
    gr.Examples(
        examples=[
            [
                "magicanimate/inputs/applications/source_image/monalisa.png",
                "magicanimate/inputs/applications/driving/densepose/running.mp4",
            ],
            [
                "magicanimate/inputs/applications/source_image/demo4.png",
                "magicanimate/inputs/applications/driving/densepose/demo4.mp4",
            ],
            [
                "magicanimate/inputs/applications/source_image/dalle2.jpeg",
                "magicanimate/inputs/applications/driving/densepose/running2.mp4",
            ],
            [
                "magicanimate/inputs/applications/source_image/dalle8.jpeg",
                "magicanimate/inputs/applications/driving/densepose/dancing2.mp4",
            ],
            [
                "magicanimate/inputs/applications/source_image/multi1_source.png",
                "magicanimate/inputs/applications/driving/densepose/multi_dancing.mp4",
            ],
        ],
        inputs=[reference_image, motion_sequence],
        outputs=animation,
    )

demo.launch()
