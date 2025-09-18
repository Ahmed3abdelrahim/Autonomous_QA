python3 sample.py \
  --out_dir=out-autonet-char \
  --data_dir=data/autonet_qa \
  --start=$'Q: What is an autonomous network in simple terms?\nA:' \
  --num_samples=1 \
  --max_new_tokens=500 \
  --temperature=0.7 \
  --top_k=100 \
  --device=cuda