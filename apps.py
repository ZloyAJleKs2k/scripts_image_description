from transformers import BlipProcessor, BlipForConditionalGeneration
import torch
from PIL import Image

processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

image_path = "img.png"  # Name image
image = Image.open(image_path)

inputs = processor(images=image, use_fast=False, return_tensors="pt")

with torch.no_grad():
    output = model.generate(
        **inputs,
        max_new_tokens=100,
        min_new_tokens=30,
        do_sample=True,
        temperature=0.7,
        top_k=50,
        top_p=0.92,
        repetition_penalty=1.2,
        num_return_sequences=1
    ) #You can set detailed parameters

description = processor.decode(output[0], skip_special_tokens=True)

print("Image description:", description)
