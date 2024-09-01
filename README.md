# LLaVAGuard

LLaVAGuard is a novel framework that offers multimodal safety guardrails to any input prompt. The safety guardrails are specifically optimized to minimize the likelihood of generating harmful responses on LLaVA-v1.5 model.

This repository contains code for the visual portion of the safety guardrail.   

## Project Structure


- **`cal_metrics.py`:** Summarizing the perplexity metrics over all examples
- **`get_metric.py`**: Script for calculating detoxify and Perspective API metrics.
- **`image_safety_patch.py`:** Script for generating safety patches from images.
- **`llava_attack.py`:** Script for generating adversarial images.
- **`llava`:** LLaVA model from [Liu et al., 2023](https://arxiv.org/abs/2304.08485).
- **`llava_utils`:** visual_attacker generates adversarial images to jailbreak visual language models. visual_defender generates image guardrails to robustify visual language models.
- **`llava_experiments`:** Scripts to test safety patches on 3 settings: constrained inference, unconstrained inference, and question answering.
- **`metric`:** Implementations of metrics such as detoxify and Perspective API.
- **`requirements.txt`:** Required Python packages for setting up the project.
- **`utils.py`:** Utility functions supporting various operations across the project, such as image loading and preprocessing. 
- **`safety_patch.bmp`:** Sample safety patch. 

## Setup

Make sure you have Python 3.10+ installed, then run:
```bash
pip install -r requirements.txt
  ```
