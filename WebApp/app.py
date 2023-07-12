import gradio as gr


def sepia(input_img, slider: int):
    return input_img + slider


demo = gr.Interface(
                    fn=sepia,
                    inputs=[gr.Image(shape=(200, 200)), gr.Slider(0, 100)],
                    outputs=gr.Image(shape=(200, 200)),
                    )

demo.launch()