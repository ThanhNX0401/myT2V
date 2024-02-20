import gradio as gr
import imageio
import torch

from pipeline import Pipeline

pipe = Pipeline.from_pretrained("runwayml/stable-diffusion-v1-5", torch_dtype=torch.float16).to("cuda")
DESCRIPTION = "Text to Video generation using Stable diffusion and multimodel"
            
def process_text2video(prompt, negative_prompt, seed, guidance_scale, num_inference_steps, video_length, fps, t0, t1):
    frame = pipe(   prompt=prompt,
                    negative_prompt=negative_prompt,
                    num_inference_steps=num_inference_steps,
                    guidance_scale=guidance_scale, 
                    generator = torch.Generator('cuda').manual_seed(seed),
                    video_length = video_length,
                    t0 = t0,
                    t1 = t1,
                ).images
    result = [(r * 255).astype("uint8") for r in frame]
    imageio.mimsave("output_video.mp4", result, fps=fps)
    return "output_video.mp4"

def create_demo_text_to_video():
    with gr.Blocks() as demo:
        title = gr.HTML(
            f"""<h1><span>{DESCRIPTION}</span></h1>""",
            elem_id="title",
        )
        with gr.Row():
            with gr.Column(variant="panel",scale=2):
                with gr.Tab("Txt2Vid"):
                    with gr.Group():
                        prompt = gr.Text(
                            label="Prompt",
                            max_lines=5,
                            placeholder="Enter your prompt",
                        )
                        negative_prompt = gr.Text(
                            label="Negative Prompt",
                            max_lines=5,
                            placeholder="Enter a negative prompt",
                        )
                with gr.Tab("Advanced Settings"):
                    with gr.Group():
                        video_length = gr.Slider(
                            label="Video length",  minimum=1, maximum=16, step=1, value=4)
                    with gr.Group():
                        seed = gr.Slider(
                            label="Seed", minimum=0, maximum=100000, step=1, value=40
                        )
                    with gr.Group():
                        fps = gr.Slider(
                            label="FPS",  minimum=0, maximum=8, step=1, value=1)
                    with gr.Group():
                        with gr.Row():
                            guidance_scale = gr.Slider(
                                label="Guidance scale",
                                minimum=1,
                                maximum=12,
                                step=0.1,
                                value=7.5,
                            )
                            num_inference_steps = gr.Slider(
                                label="Number of inference steps",
                                minimum=1,
                                maximum=100,
                                step=1,
                                value=50,
                            )
                    with gr.Group():
                        gr.Markdown("Perform DDPM steps from t0 to t1. The larger the gap between t0 and t1, the more variance between the frames. Ensure t0 < t1 ")
                        t0 = gr.Slider(label="Timestep t0", minimum=0, maximum=100, value=21, step=1,)
                        t1 = gr.Slider(label="Timestep t1", minimum=1, maximum=100, value=27, step=1,)

            with gr.Column(variant="panel",scale=3):
                with gr.Blocks():
                    run_button = gr.Button("Generate", variant="primary")
                result = gr.Video(label="Generated Video")

        inputs = [
            prompt,
            negative_prompt,
            seed,
            guidance_scale,
            num_inference_steps,
            video_length,
            fps,
            t0,
            t1
        ]

        run_button.click(fn=process_text2video,
                        inputs=inputs,
                        outputs=result,)
    return demo

