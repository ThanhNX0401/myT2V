'''!pip install -q transformers==4.36.0
!pip install -q bitsandbytes==0.41.3 accelerate==0.25.0'''

import torch
from transformers import BitsAndBytesConfig
from transformers import pipeline
from PIL import Image
from transformers import AutoProcessor, LlavaForConditionalGeneration
quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16
)

model_id = "llava-hf/llava-1.5-7b-hf"
processor = AutoProcessor.from_pretrained(model_id)
model = LlavaForConditionalGeneration.from_pretrained(model_id, quantization_config=quantization_config, device_map="auto")
#pipe = pipeline("image-to-text", model=model_id, model_kwargs={"quantization_config": quantization_config})

def get_label(image, prompt):
    prompts = f"USER: <image>\n Given a user prompt, identify the moving objects or parts. Give me the labels name only. Prompt: {prompt}” \nASSISTANT:"
    inputs = processor(prompts, images=image, padding=True, return_tensors="pt").to("cuda")
    output = model.generate(**inputs, max_new_tokens=20)
    generated_text = processor.batch_decode(output, skip_special_tokens=True)
    for text in generated_text:
        print(text.split("ASSISTANT:")[-1])
    return text.split("ASSISTANT:")[-1]
def get_motion(image,prompt):
    label = get_label(image, prompt)
    prompts = f'''USER: <image>\nGiven the following image prompt: {prompt}, elaborate the movement direction of the {label}.
                Motions should be consistent. Directions should be one of followings:[“motionless”,
                “left”, “right”, “up”, “down”, “left down”, “left up”,“right down”, “right up”].
                \nASSISTANT:'''
    inputs = processor(prompts, images=image, padding=True, return_tensors="pt").to("cuda")
    output = model.generate(**inputs, max_new_tokens=20)
    generated_text = processor.batch_decode(output, skip_special_tokens=True)
    for text in generated_text:
        print(text.split("ASSISTANT:")[-1])
    return text.split("ASSISTANT:")[-1]
    