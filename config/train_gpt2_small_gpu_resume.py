# config for training GPT-2 on 8GB GPU - RESUME FROM CHECKPOINT
# Use this config to continue training from checkpoint
# optimized for single 8GB GPU (RTX 3070, RTX 4060, etc.)

wandb_log = True
wandb_project = 'owt'
wandb_run_name = 'gpt2-small-gpu'
wandb_run_id = 'u6c0dakr'  # fixed run ID for resume

# reduced batch size for 8GB VRAM
batch_size = 4
block_size = 512
gradient_accumulation_steps = 10

# training params
max_iters = 100000
lr_decay_iters = 100000

# resume from checkpoint
out_dir = 'out'
init_from = 'resume'

# eval settings
eval_interval = 500
eval_iters = 200
log_interval = 10

# learning rate
learning_rate = 6e-4
weight_decay = 1e-1
beta1 = 0.9
beta2 = 0.95

# lr decay
warmup_iters = 2000
min_lr = 6e-5

# system
device = 'cuda'
dtype = 'float16'
compile = False
