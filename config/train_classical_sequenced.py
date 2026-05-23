# Training config: Classical Curriculum SEQUENCED condition
# ScratchTrainingCuration experiment
#
# Usage: python train.py config/train_classical_sequenced.py
#
# GPT-2 small architecture (12 layers, 12 heads, 768 embed)
# with custom 8K vocab tokenizer trained on classical corpus.
# Runs on Apple M2 Air 24GB via MPS.

# Output
out_dir = 'out-classical-sequenced'
eval_interval = 250
eval_iters = 100
log_interval = 10
always_save_checkpoint = True

# Logging (disable wandb by default, enable with --wandb_log=True)
wandb_log = False
wandb_project = 'scratch-training-curation'
wandb_run_name = 'sequenced'

# Data
dataset = 'classical_sequenced'
sequential_data = True   # step through file in curriculum order (patched get_batch)
batch_size = 4
block_size = 2048
gradient_accumulation_steps = 8
# Effective batch: 8 * 2048 * 4 = 65,536 tokens/step

# Model: GPT-2 small (124M with standard vocab, ~85M with 8K vocab)
# Both conditions use identical architecture. Parameter count difference
# vs standard GPT-2 is entirely from smaller embedding layer.
n_layer = 12
n_head = 12
n_embd = 768
dropout = 0.0  # pretraining, not finetuning
bias = True

# Training
init_from = 'scratch'
max_iters = 5000
# ~25M train tokens, 65K tokens/step = ~385 steps/epoch, so 5000 = ~13 epochs
# Multiple epochs appropriate for small curated corpus (cf. Phi, LIMA)

# Optimizer
learning_rate = 3e-4
weight_decay = 0.1
beta1 = 0.9
beta2 = 0.95
grad_clip = 1.0

# LR schedule
decay_lr = True
warmup_iters = 200
lr_decay_iters = 5000
min_lr = 3e-5  # learning_rate / 10

# System: Apple M2 Air
device = 'mps'
dtype = 'float32'  # MPS does not support bfloat16
compile = False     # torch.compile not reliable on MPS
