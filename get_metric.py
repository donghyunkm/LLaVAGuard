
import json
from metric.perspective_api import PerspectiveApiScorer
from metric.detoxify import DetoxifyScorer
import argparse
from tqdm import tqdm
from evaluate import load
import os

api_key = "AIzaSyCRQyj4x_oxe7mcqoBv4-YJ9YqKdmDiSDQ"
perspective_scorer = PerspectiveApiScorer(api_key)
detoxify_scorer = DetoxifyScorer()
perplexity = load("perplexity", module_type="metric")

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluation")
    parser.add_argument("--input", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--perplexity", type=str, default="")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--device", type=str, default="cuda", help="cuda or cpu or gpu")
    parser.add_argument("--model_path", type=str, default="/workingdir/models_hf/liuhaotian/llava-v1.5-13b")

    parser.add_argument("--load_existing_generation", action="store_true")

    args = parser.parse_args()
    return args

args = parse_args()
input_file = args.input
output_file = args.output
if output_file == "":
    output_file = input_file.split('.jsonl')[0] + '_eval.jsonl'

def float_it(score):
    for key in score.keys():
        score[key] = float(score[key])
    return score

print(f"input file: {input_file}")
print(f"Output file {output_file}")
inputs = open(input_file).read().strip().split('\n')
all_text = []

if args.load_existing_generation:

    with open(args.output, 'r') as f:
        lines = f.readlines()
        for i, line in enumerate(lines):
            if i == 0 or (i <= 3 and "_qna" in args.input):
                print(f"Skipping {i}-th line")
                continue
            obj = json.loads(line)
            all_text += [obj['continuation']]

else:

    with open(output_file, 'w') as f:
        for i, li in enumerate(tqdm(inputs)):


            if i == 0 or (i <= 3 and "_qna" in args.input):
                print(f"Skipping {i}-th line")
                continue
            obj = json.loads(li)
            if 'continuation' in obj:

                text = obj['continuation']
                if type(text) == list:
                    text = text[0]

                if text != "":
                    all_text.append(text)
                    res = {
                        'detoxify': float_it(detoxify_scorer.get_scores(text)),
                        'perspective_api': float_it(perspective_scorer.get_scores(text))
                    }
                    obj['continuation_metric'] = res

            f.write(json.dumps(obj))
            f.write('\n')

print("Output file", output_file)
with open(args.perplexity, 'a') as f:
    perplexity = perplexity.compute(predictions=all_text, model_id=args.model_path,device=args.device, batch_size=args.batch_size)['mean_perplexity']
    f.write(f"Perplexity: {perplexity}\n")

print("Perplexity:", perplexity)
