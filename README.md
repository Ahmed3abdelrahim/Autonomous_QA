# Autonomous QA - LLM From Scratch

This project demonstrates how to build a small LLM from scratch for question-answering over a custom dataset (TM Forum whitepaper on Autonomous Networks). It includes training, evaluation, and API serving capabilities.

## Features
- Trains a GPT-style transformer from scratch (no pre-trained weights)
- Data preprocessing from TM Forum whitepaper
- Evaluation pipeline with multiple metrics (F1, EM, perplexity, hallucination rate, etc.)
- FastAPI-based REST server for inference
- Cloud deployment strategy (AWS ECS or SageMaker)

---

## Project Structure
```
Autonomous_QA/
├── train.py                  # Training loop with checkpointing
├── configurator.py          # Training configuration loader
├── model.py                 # GPT model and configuration
├── data/
│   └── autonet_qa/          # Prepared tokenized dataset (bin/txt)
├── out-autonet-char/        # Output directory with model checkpoints
├── server/
│   ├── app.py               # FastAPI entrypoint
│   └── generator.py         # Text generation handler
├── evaluate.py              # Evaluation script with metrics
├── eval.jsonl               # Evaluation Q&A samples
├── run.sh                   # Script to launch API server
├── test.sh                  # Script to test model endpoint
├── Deployment_AWS.md        # Deployment architecture and strategy
└── README.md                # This file
```

---

## Setup
```bash
# 1. Clone this repo
$ git clone https://github.com/Ahmed3abdelrahim/Autonomous_QA.git
$ cd Autonomous_QA

# 2. Create virtualenv and install dependencies
$ python3 -m venv .venv
$ source .venv/bin/activate
$ pip install -r requirements.txt

# 3. (Optional) Prepare dataset
$ python data/autonet_qa/prepare.py
```

---

## Training
```bash
python train.py \
  out_dir=out-autonet-char \
  dataset=autonet_char \
  data_dir=data/autonet_qa \
  gradient_accumulation_steps=1 \
  batch_size=64 block_size=256 \
  n_layer=4 n_head=4 n_embd=256 \
  dropout=0.1 bias=False \
  learning_rate=3e-3 max_iters=5000 \
  weight_decay=1e-1 beta2=0.99 warmup_iters=100 \
  compile=False init_from=scratch device=cuda
```

> Checkpoints will be saved to `out-autonet-char/ckpt.pt`.

---

## Evaluation
```bash
python evaluate.py --eval_file eval.jsonl --model_dir out-autonet-char
```
Outputs:
- Perplexity on validation text
- F1/EM for QA
- Keyword hit rate
- Hallucination & refusal metrics

---

## API Server
```bash
# Start server
$ bash run.sh

# Test endpoint
$ bash test.sh
```

---

## Deployment
See `Deployment_AWS.md` for:
- AWS ECS + ECR strategy
- SageMaker architecture (with draw.io + Mermaid)
- CI/CD and GPU scaling suggestions

---

## License
This project is for assessment and demonstration purposes only. All rights reserved.

---

## Author
Ahmed Abdelrahim
- [GitHub](https://github.com/Ahmed3abdelrahim)
- [LinkedIn](https://www.linkedin.com/in/ahmed3abdelrahim/)