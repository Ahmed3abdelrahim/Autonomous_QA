# config/train_autonet_char.py
# Tiny GPT config for a small dataset (~120 Q&As)

out_dir = 'out-autonet-char'
eval_interval = 100
eval_iters = 100
log_interval = 10

# data
dataset = 'autonet_char'   # reuse the char-level dataset loader
data_dir = 'data/autonet_qa'   # path to your prepared .txt/.bin/meta.pkl files

# optimization
gradient_accumulation_steps = 1
batch_size = 64
block_size = 256   # context length; Q&A pairs are short

# model (very small)
n_layer = 4
n_head = 4
n_embd = 256
dropout = 0.1
bias = False

# optimizer
learning_rate = 3e-3
max_iters = 5000
weight_decay = 1e-1
beta2 = 0.99
warmup_iters = 100

# compilation
compile = False

# init from scratch (no pretrained weights)
init_from = 'scratch'
