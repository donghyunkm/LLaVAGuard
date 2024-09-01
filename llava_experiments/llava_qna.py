import argparse
import os
import random
import cv2
import sys
sys.path.append('..')

import numpy as np
import torch.nn as nn
import torch
import torch.backends.cudnn as cudnn
from PIL import Image
import json
from torchvision.utils import save_image
from utils import data_read

def denormalize(images):
    mean = torch.tensor([0.48145466, 0.4578275, 0.40821073]).cuda()
    std = torch.tensor([0.26862954, 0.26130258, 0.27577711]).cuda()
    new_images = (images - mean[None, :, None, None])/ std[None, :, None, None]
    return new_images

def normalize(images):
    mean = torch.tensor([0.48145466, 0.4578275, 0.40821073]).cuda()
    std = torch.tensor([0.26862954, 0.26130258, 0.27577711]).cuda()
    new_images = (images * std[None, :, None, None])+ mean[None, :, None, None]
    return new_images

def parse_args():

    parser = argparse.ArgumentParser(description="Demo")
    parser.add_argument("--model-path", type=str, default="ckpts/llava_llama_2_13b_chat_freeze")
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--gpu-id", type=int, default=0, help="specify the gpu to load the model.")
    parser.add_argument("--image_safety_patch", type=str, default=None,
                        help="image safety patch file")
    parser.add_argument("--text_safety_patch", type=str, default=None,
                        help="text safety patch file") 
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
model.eval()

print('[Initialization Finished]\n')

from llava_utils import prompt_wrapper, generator_qna

my_generator = generator_qna.Generator(model=model, tokenizer=tokenizer)

# ========================================
#             Inference
# ========================================

##  TODO: expose interface.
qna = data_read('datasets/aokvqa/aokvqa_v1p0_val.json','question',K=1000)

if args.text_safety_patch!=None:
    with open(args.text_safety_patch, 'r') as file:
        text_safety_patch = file.read().rstrip()


responses = []
acc = []
text_prompt = '%s'
with torch.no_grad():

    for i in range(len(qna)):
        
        image_id,question,choices,answer_id,direct_answers = qna[i]
        image_id = (12-len(str(image_id)))*'0'+str(image_id)
        image = load_image('datasets/coco/val2017/'+image_id+'.jpg')
        image.save("llava_original.png")

        image = image_processor.preprocess(image, return_tensors='pt')['pixel_values'].cuda()
        if args.image_safety_patch!=None:
            image = normalize(image)
            safety_patch = torch.load(args.image_safety_patch).cuda()
            safe_image = denormalize((image + safety_patch).clamp(0,1))
        else:
            safe_image = image

        print(f" ----- {i} ----")
        print(" -- prompt: ---")

        if args.text_safety_patch!=None:
            #use the below for optimal text safety patch
#            question = text_safety_patch + '\n' + question + '\nA. '+choices[0]+'\nB. '+choices[1]+'\nC. '+choices[2]+'\nD. '+choices[3] + "\nAnswer with the option's letter from the given choices directly."
            #use the below for heuristic text safety patch
            question = question + '\nA. '+choices[0]+'\nB. '+choices[1]+'\nC. '+choices[2]+'\nD. '+choices[3] + "\nAnswer with the option's letter from the given choices directly." + '\n' + text_safety_patch    
        else:
            question = question + '\nA. '+choices[0]+'\nB. '+choices[1]+'\nC. '+choices[2]+'\nD. '+choices[3] + "\nAnswer with the option's letter from the given choices directly."

        text_prompt_template = prompt_wrapper.prepare_text_prompt(question)
        
        print(text_prompt_template)
        prompt = prompt_wrapper.Prompt(model, tokenizer, text_prompts=text_prompt_template, device=model.device)

        response = my_generator.generate(prompt, safe_image).replace("[INST]","").replace("[/INST]","").replace("<<SYS>>","").replace("<</SYS>>","").replace("[SYS]","").replace("[/SYS]","").strip()
        if args.text_safety_patch!=None:
            response = response.replace(text_safety_patch,"")
        print(" -- response: ---")
        responses.append({'prompt': question, 'continuation': response})
        maxv = 99999
        response_id = -1
        for idx in range(4):
            loc = response.find(chr(ord('A')+idx)+'.')
            if loc!=-1 and maxv>loc:
                maxv = loc
                response_id = chr(ord('A')+idx)
 
            loc = response.find(chr(ord('A')+idx)+'\n')
            if loc!=-1 and maxv>loc:
                maxv = loc
                response_id = chr(ord('A')+idx)
    
            loc = response.find(chr(ord('A')+idx)+' ')
            if loc==0 and maxv>loc:
                maxv = loc
                response_id = chr(ord('A')+idx)
 
            loc = response.find(chr(ord('A')+idx)+'\t')
            if loc==0 and maxv>loc:
                maxv = loc
                response_id = chr(ord('A')+idx)
                           
        answer_id = chr(ord('A')+answer_id)
        print(response,response_id,answer_id)
        acc.append(response_id==answer_id if len(response)!=0 else 0)

print('overall_accuracy = {}'.format(np.average(acc)))

with open(args.output_file, 'w') as f:
    print('overall_accuracy = {}'.format(np.average(acc)),file=f)
    f.write(json.dumps({
        "args": vars(args),
        "prompt": text_prompt
    }))
    f.write("\n")

    for li in responses:
        f.write(json.dumps(li))
        f.write("\n")

