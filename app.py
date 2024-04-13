import gradio as gr

from text2performer import text2performer

input_appearance = 'The dress the person wears has long sleeves and it is of short length. Its texture is pure color.'
input_motion = '''The lady moves to the right.
The person is moving to the center from the right.
She turns right from the front to the side.
She turns right from the side to the back.'''

with gr.Blocks() as demo:
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

demo.launch()
