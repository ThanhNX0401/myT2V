from __future__ import annotations


import gradio as gr



import gradio as gr
import argparse


from css.theme import Seafoam
from css import theme

image_path=theme.image_path

seafoam=Seafoam()

parser = argparse.ArgumentParser()
parser.add_argument('--public_access', action='store_true',
                    help="if enabled, the app can be access from a public url", default=False)
args = parser.parse_args()



with gr.Blocks(css="css/style.css", theme=seafoam) as demo:
    create_demo_text_to_video()


print("Generating Gradio app LINK:")
_, _, link = demo.queue(api_open=False).launch(share=args.public_access,allowed_paths=[image_path])
print(link)
