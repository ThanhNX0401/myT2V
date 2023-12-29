import openai
import time
import torch
import requests
from PIL import Image
from transformers import BlipProcessor, BlipForQuestionAnswering

# Initialize the client
client = openai.OpenAI(api_key="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")

 
def get_label(user_prompt):
    # Step 1: Create an Assistant
    assistant = client.beta.assistants.create(
        name="The Director",
        instructions=f"Given a user prompt, identify the moving objects or parts. Skip the background object, give me the label of all the moving object only, don't answer anything else except the label",
        model="gpt-3.5-turbo"
    )
    # Step 2: Create a Thread
    thread = client.beta.threads.create()

    # Step 3: Add a Message to a Thread
    message = client.beta.threads.messages.create(
        thread_id=thread.id,
        role="user",
        content=user_prompt
    )
    print("running assistant")
    # Step 4: Run the Assistant
    run = client.beta.threads.runs.create(
        thread_id=thread.id,
        assistant_id=assistant.id,
        instructions=f"Given a user prompt, identify the moving objects or parts. Skip the background object, give me the label of all the moving object only, don't answer anything else except the label"
    )


    while True:
        # Wait for 5 seconds
        time.sleep(1)  

        # Retrieve the run status
        run_status = client.beta.threads.runs.retrieve(
            thread_id=thread.id,
            run_id=run.id
        )

        # If run is completed, get messages
        if run_status.status == 'completed':
            messages = client.beta.threads.messages.list(
                thread_id=thread.id
            )

            break
        
    prompt = messages.data[0].content[0].text.value
    print(prompt)
    label_list = prompt.split(',')
    return label_list

def get_motion(image):
    processor = BlipProcessor.from_pretrained("ybelkada/blip-vqa-capfilt-large")
    model = BlipForQuestionAnswering.from_pretrained("ybelkada/blip-vqa-capfilt-large", torch_dtype=torch.float16).to("cuda")

    question = "what is the direction of the main object in the image?"
    inputs = processor(image, question, return_tensors="pt",do_rescale=False).to("cuda", torch.float16)

    out = model.generate(**inputs)
    print(processor.decode(out[0], skip_special_tokens=True))
    motion = processor.decode(out[0], skip_special_tokens=True)
    return motion

