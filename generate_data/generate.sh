python generate_samples.py \
  --fine_tuned_model_path "./finetuned_llama_model" \
  --base_model_name "meta-llama/Llama-2-7b-hf" \
  --num_samples_to_generate 2000 \
  --output_json_file "./generated_samples.json" \
  --quality_threshold 0.8