# CSE8803 LLM Project (2024 Fall)

This repository contains code for training the difussion model with [DoT](https://arxiv.org/abs/2402.07754) using the generated training samples by language models. 

## Overview
Large Language Models (LLMs) have achieved impressive success in System-1 tasks requiring intuitive and rapid processing but continue to face significant challenges in System-2 tasks that demand deliberate reasoning, particularly in mathematical domains. Existing approaches, such as Chain-of-Thought (CoT) reasoning, enhance reasoning by generating intermediate steps. However, these methods often suffer from inefficiencies, error propagation, and limited scalability. To address these challenges, Diffusion of Thoughts (DoT) was recently proposed, reframing combinatorial search problems as continuous optimization tasks. While DoT shows promise in improving reasoning, it struggles with generating high-quality synthetic data, thereby constraining its scalability. In this work, we introduce a hybrid framework that integrates a System-1 language model for synthetic data generation with a DoT-inspired model for reasoning. Our iterative training pipeline incorporates self-correction and verification mechanisms to ensure the quality of the generated synthetic data, fostering synergistic improvement between the two sub-models. Using the GSM8K dataset as a benchmark, our preliminary results highlight the framework's potential to enhance reasoning performance while scaling data efficiently. 

<img src = "fig/pipeline.pdf" align = "center" width="80%" hight="80%">

## Setup
All required packages can be found in requirements.txt. You can install them in a new environment with
```
conda create -n dot_sync python=3.10
conda activate dot_sync

# The following line to be replaced depending on your cuda version.
cd 8803_LLM_project
pip install -r requirements.txt -f https://download.pytorch.org/whl/torch_stable.html
```

Install NVIDIA Apex with fused kernels:
```
git clone https://github.com/NVIDIA/apex
cd apex
git checkout 2386a912164b0c5cfcd8be7a2b890fbac5607c82
pip install -v --disable-pip-version-check --no-cache-dir --no-build-isolation --config-settings "--build-option=--cpp_ext" --config-settings "--build-option=--cuda_ext" ./

```

## Finetuning Plaid 1B
First download the weights from here: [Plaid 1B Weights Download Page](https://github.com/igul222/plaid/releases/tag/v1.0.0). Download data from here: [4by4/5by4/GSM8k-Aug data](https://github.com/da03/implicit_chain_of_thought/tree/main/data) and put them in the ./data folder with names 4by4/5by5/gsm8k.

Extract them:
```
cat plaid1b_weights.tar.gz.* | tar xvzf -
```

Then run the following code:

```
# DoT
SAMPLE_NUM=2000 GENERATE=0 GENERATE_DATA_PATH="DATA_PATH_OF_GENERATED_DATA" python train.py --digit --fix_src --dataset gsm8k --steps 120000 --weights_path plaid1b_weights 

# DoT-MP
SAMPLE_NUM=2000 GENERATE=0 GENERATE_DATA_PATH="DATA_PATH_OF_GENERATED_DATA" python train.py --digit --fix_src --cot --dataset gsm8k --steps 31000 --weights_path plaid1b_weights 
```

Please refer to `run_train.sh` for more training commands.


## Data Generation
Here are commands for fine-tuning the system-1 model and generating new training amples.
```
# supervised-fine-tuning 
python sft.py \
  --model_name_or_path "meta-llama/Llama-2-7b-hf" \
  --output_dir "./finetuned_llama_model" \
  --generated_json_path "./generated_samples.json"

# generate new samples
python generate_samples.py \
  --fine_tuned_model_path "./finetuned_llama_model" \
  --base_model_name "meta-llama/Llama-2-7b-hf" \
  --num_samples_to_generate 2000 \
  --output_json_file "./generated_samples.json" \
  --quality_threshold 0.8
```
Please feel free to change the model to be fine-tuned and other arguments if necessary.
