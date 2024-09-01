import argparse
import torch
import os
from torchvision.utils import save_image
import json
import numpy as np
from PIL import Image

def data_read(text_file,K=200):
    dataset = []
#    lines = open(text_file).read().split("\n")

    for count,line in enumerate(open(text_file, 'r')):
        obj = json.loads(line)
        cur_str = obj['rejected_response']
        if len(cur_str)!=0:
            dataset.append(cur_str)
    return dataset[-K:]


def parse_args():

    parser = argparse.ArgumentParser(description="Demo")
    parser.add_argument("--model-path", type=str, default="ckpts/llava_llama_2_13b_chat_freeze")
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--gpu_id", type=int, default=0, help="specify the gpu to load the model.")
    parser.add_argument("--n_iters", type=int, default=5000, help="specify the number of iterations for attack.")
    parser.add_argument('--eps', type=int, default=64, help="epsilon of the attack budget")
    parser.add_argument('--alpha', type=int, default=1, help="step_size of the attack")
    parser.add_argument("--save_dir", type=str, default='output',
                        help="save directory")

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

model.resize_token_embeddings(len(tokenizer))
model.eval()
print('[Initialization Finished]\n')


if not os.path.exists(args.save_dir):
    os.mkdir(args.save_dir)

import csv

#read the small corpus including harmful content, which is needed for safety patch generation
lines = open('harmful_corpus/harmful_strings.csv').read().split("\n")
neg_targets = [li for li in lines if len(li)>0]
#normal input prompt just in case
pos_targets = data_read('harmful_corpus/red_teaming_prompts.jsonl',K=len(neg_targets))

from llava_utils import visual_defender

print('device = ', model.device)
my_defender = visual_defender.Defender(args, model, tokenizer, pos_targets, neg_targets, device=model.device, image_processor=image_processor)

template_img = 'unconstrained_attack.bmp'
image = load_image(template_img)
image = image_processor.preprocess(image, return_tensors='pt')['pixel_values'].cuda()

from llava_utils import prompt_wrapper
text_prompt_template = prompt_wrapper.prepare_text_prompt('')
print(text_prompt_template)
    
safety_patch = my_defender.defense_constrained(text_prompt_template,
                                                            img=image, batch_size=2,
                                                            num_iter=args.n_iters, alpha=args.alpha / 255,
                                                            epsilon=args.eps / 255)
#save_image(safety_patch, '%s/safety_patch.bmp' % (args.save_dir))
torch.save(safety_patch, '%s/safety_patch.pt' % args.save_dir)

print('[Done]')
