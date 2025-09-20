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
│   └── autonet_qa/         # Prepared tokenized dataset (bin/txt)
|        ├── prepare.py               #convert file to train and val
│        └── prepare_char_bin.py     #convert file to bin files
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


## Data Preparation

> I train on a Q&A corpus distilled from the TM Forum *Autonomous Networks* white paper. The pipeline stays simple (char-level) and reproducible, with checks to **prevent evaluation leakage**.

### 1) Convert source PDF(s) to Q&A dataset
If starting from the original PDF, extracted Question and anwser via PDF.

### 2) Convert Jsonl file to train and val files and convert them to binary files

```bash
# prepare jsonl dataset file
$ python prepare.py

# convert it to binary files
$ python prepare_char_bin.py
```


## Training


```bash
python3 train.py config/train_autonet_char.py
```

> Checkpoints will be saved to `out-autonet-char/ckpt.pt`.

---

## Evaluation
```bash
python evaluate.py 
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

![In Draw IO Diagram](.\resources\drawio.png)
![In Mermaid Diagram](.\resources\mermaid.png)

## License
This project is for assessment and demonstration purposes only. All rights reserved.

---

## Author
Ahmed Abdelrahim
- [GitHub](https://github.com/Ahmed3abdelrahim)
- [LinkedIn](https://www.linkedin.com/in/ahmed-abdelrahim-elsayed-5a9673175/)