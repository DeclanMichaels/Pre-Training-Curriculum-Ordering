# Training config: Classical Curriculum SHUFFLED condition (primary baseline)
# ScratchTrainingCuration experiment
#
# Usage: python train.py config/train_classical_shuffled.py
#
# IDENTICAL to sequenced config except dataset and output dir.
# Same architecture, same hyperparameters, same training duration.
# The ONLY difference is chunk ordering in the training data.

# Output
out_dir = 'out-classical-shuffled'
eval_interval = 250
eval_iters = 100
log_interval = 10
always_save_checkpoint = True

# Logging
wandb_log = False
wandb_project = 'scratch-training-curation'
wandb_run_name = 'shuffled'

# Data
dataset = 'classical_shuffled'
sequential_data = False  # random sampling (default nanoGPT behavior)
batch_size = 4
block_size = 2048
gradient_accumulation_steps = 8

# Model: identical to sequenced
n_layer = 12
n_head = 12
n_embd = 768
dropout = 0.0
bias = True

# Training: identical to sequenced
init_from = 'scratch'
max_iters = 5000

# Optimizer: identical to sequenced
learning_rate = 3e-4
weight_decay = 0.1
beta1 = 0.9
beta2 = 0.95
grad_clip = 1.0

# LR schedule: identical to sequenced
decay_lr = True
warmup_iters = 200
lr_decay_iters = 5000
min_lr = 3e-5

# System: Apple M2 Air
device = 'mps'
dtype = 'float32'
compile = False
