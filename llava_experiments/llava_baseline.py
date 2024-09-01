import argparse
import os
import random
import sys
sys.path.append('..')

import cv2
import numpy as np
import torch
import torch.backends.cudnn as cudnn
from PIL import Image
import json
import pandas as pd

def rtp_read(text_file):
    dataset = []
    lines = open(text_file).read().split("\n")
    for li in lines:
        obj = json.loads(li)
        if obj['challenging']:
            dataset.append(obj['prompt']['text'])
    return dataset

def parse_args():

    parser = argparse.ArgumentParser(description="Demo")
    parser.add_argument("--model-path", type=str, default="ckpts/llava_llama_2_13b_chat_freeze")
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--mode", type=int, default=1, help="testing against benign prompts (mode=0) and adversarial prompts (mode=1).")
    parser.add_argument("--image_path", type=str, default='./image.bmp',
                        help="Image file")
    parser.add_argument("--output_file", type=str, default='./result.jsonl',
                        help="Output file.")


    

    args = parser.parse_args()
    return args


def load_image(image_path):
    image = Image.open(image_path).convert('RGB')
    return image


# ========================================
#             Model Initialization
# ========================================

print('>>> Initializing Models')

from llava.utils import get_model
args = parse_args()

print('model = ', args.model_path)

tokenizer, model, image_processor, model_name = get_model(args)
model = model.cuda()
print(model)
print(model_name)
model.eval()

print('[Initialization Finished]\n')


from llava_utils import prompt_wrapper, generator


my_generator = generator.Generator(model=model, tokenizer=tokenizer)


# ========================================
#             Inference
# ========================================

##  TODO: expose interface.
if args.mode==1:
    prompts = rtp_read('harmful_corpus/rtp_prompts.jsonl')
else:
    lines = open('prompts/benign').read().split("\n")
    prompts = [li for li in lines]

#text_prompt = 'Please extend the following sentence: %s'
text_prompt = '%s'

out = []
with torch.no_grad():

    for i, user_message in enumerate(prompts):

        print(f" ----- {i} ----")
        print(" -- prompt: ---")

        print(text_prompt % user_message)

        # blur kernel baseline
        #image = cv2.imread(args.image_path+str(np.random.randint(25))+'.bmp')
        #image = cv2.blur(image,(3,3))

        # compression decompression baseline
        image = load_image(args.image_path+str(np.random.randint(25))+'.bmp')
        image = image.save('baseline/compressed_q10.jpg', quality=10)
        image = load_image('baseline/compressed_q10.jpg')

        safe_image = image_processor.preprocess(image, return_tensors='pt')['pixel_values'].cuda()
        
        text_prompt_template = prompt_wrapper.prepare_text_prompt(text_prompt % user_message)
        print(text_prompt_template)
        prompt = prompt_wrapper.Prompt(model, tokenizer, text_prompts=text_prompt_template, device=model.device)

        response = my_generator.generate(prompt, safe_image)

        print(" -- continuation: ---")
        print(response)
        out.append({'prompt': user_message, 'continuation': response})
        print()

with open(args.output_file, 'w') as f:
    f.write(json.dumps({
        "args": vars(args),
        "prompt": text_prompt
    }))
    f.write("\n")

    for li in out:
        f.write(json.dumps(li))
        f.write("\n")
