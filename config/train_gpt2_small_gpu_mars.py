# config for training GPT-2 on 8GB GPU with MARS optimizer - START FROM SCRATCH
# Use this config for first-time training with MARS optimizer
# optimized for single 8GB GPU (RTX 3070, RTX 4060, etc.)

wandb_log = True
wandb_project = 'owt'
wandb_run_name = 'gpt2-small-gpu-mars'

# reduced batch size for 8GB VRAM
batch_size = 4
block_size = 512
gradient_accumulation_steps = 10

# training params
max_iters = 100000
lr_decay_iters = 100000

# start from scratch
out_dir = 'out-mars'
init_from = 'scratch'

# eval settings
eval_interval = 500
eval_iters = 200
log_interval = 10

# MARS optimizer settings
optimizer_name = 'mars'
learning_rate = 6e-3  # MARS typically uses higher lr than AdamW (5-10x)
weight_decay = 1e-2   # MARS paper recommends 0.01
beta1 = 0.95          # MARS paper recommends 0.95
beta2 = 0.99          # MARS paper recommends 0.99

# MARS specific parameters
mars_gamma = 0.025           # gradient correction strength (default: 0.025)
mars_is_approx = True        # use approximate MARS (no extra forward pass)
mars_type = 'mars-adamw'     # 'mars-adamw' or 'mars-lion'
mars_optimize_1d = False     # use AdamW for 1D params (more stable)
mars_lr_1d = 3e-3           # learning rate for 1D parameters
mars_betas_1d = (0.9, 0.95) # betas for 1D parameters
mars_weight_decay_1d = 0.1  # weight decay for 1D parameters

# lr decay
warmup_iters = 2000
min_lr = 6e-4  # 10% of max_lr

# gradient clipping
grad_clip = 1.0

# system
device = 'cuda'
dtype = 'float16'
compile = False  # disable for first run, enable after successful test
