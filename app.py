from __future__ import annotations

import gradio as gr

from app_text_to_video import create_demo_text_to_video
from css.theme import Seafoam
from css import theme

image_path=theme.image_path

seafoam=Seafoam()


with gr.Blocks(css="css/style.css", theme=seafoam) as demo:
    create_demo_text_to_video()


print("Generating Gradio app LINK:")
_, _, link = demo.queue(api_open=False).launch(share=True,allowed_paths=[image_path])
print(link)
