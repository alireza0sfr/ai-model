from unsloth import FastVisionModel
from PIL import Image
import torch
import os

from radigenius.model_utils import initialize_model

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model, tokenizer = initialize_model("inference")
FastVisionModel.for_inference(model)

def predict_radiology_description(image, instruction):
    try:
        messages = [{"role": "user", "content": [
            {"type": "image"},
            {"type": "text", "text": instruction}
        ]}]
        input_text = tokenizer.apply_chat_template(messages, add_generation_prompt=True)

        inputs = tokenizer(
            image,
            input_text,
            add_special_tokens=False,
            return_tensors="pt",
        ).to(device)

        output_ids = model.generate(
            **inputs,
            max_new_tokens=256,
            temperature=1.5,
            min_p=0.1
        )

        generated_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
        return generated_text.replace("assistant", "\n\nassistant").strip()
    except Exception as e:
        return f"Error: {str(e)}"


def main():
    images = os.listdir('images')
    image_options = '\n'.join([f'{image.split(".")[0]}' for image in images])

    while True:
      input_image = input(f'Enter image name. Available options are:\n{image_options}\n> ')

      default_prompt = "You are an expert radiographer. Describe accurately what you see in this image"
      user_prompt = input(f'Enter your prompt or press enter to use default prompt: (default: {default_prompt})')
      instruction = user_prompt if user_prompt else default_prompt

      image_path = f'images/{input_image}.jpg'

      image = Image.open(image_path).convert("RGB")

      print(f'using image: {image_path}')
      print(f'using prompt: {instruction}')

      print('generating description...')

      output = predict_radiology_description(image, instruction)
      print(output)
      print('\n--------------------------------\n')


if __name__ == "__main__":
    main()
