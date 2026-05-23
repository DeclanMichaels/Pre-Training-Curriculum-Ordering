"""
Sample from a trained classical curriculum model.
Uses our SentencePiece tokenizer for proper encoding/decoding
instead of nanoGPT's default tiktoken (which is GPT-2's BPE).

Usage:
  python scripts/sample_model.py --model sequenced --prompt "The nature of justice"
  python scripts/sample_model.py --model shuffled --prompt "Water flows downhill"
  python scripts/sample_model.py --model sequenced --interactive
  python scripts/sample_model.py --compare --prompt "The virtue of a man"

Requires: trained model checkpoint in nanoGPT/out-classical-{model}/ckpt.pt
"""
import os
import sys
import argparse

import torch
import sentencepiece as spm

# Add nanoGPT to path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.join(SCRIPT_DIR, "..")
NANOGPT_DIR = os.path.join(PROJECT_ROOT, "nanoGPT")
sys.path.insert(0, NANOGPT_DIR)

from model import GPTConfig, GPT

TOKENIZER_MODEL = os.path.join(PROJECT_ROOT, "corpus", "tokenized", "tokenizer.model")

MODEL_DIRS = {
    "sequenced": os.path.join(NANOGPT_DIR, "out-classical-sequenced"),
    "shuffled": os.path.join(NANOGPT_DIR, "out-classical-shuffled"),
    "shuffled-docs": os.path.join(NANOGPT_DIR, "out-classical-shuffled-docs"),
}


def load_model(model_name=None, device='mps', ckpt_path=None):
    """Load a trained checkpoint."""
    if ckpt_path is None:
        model_dir = MODEL_DIRS.get(model_name)
        if not model_dir or not os.path.isdir(model_dir):
            print(f"Model directory not found: {model_dir}")
            print(f"Available: {list(MODEL_DIRS.keys())}")
            sys.exit(1)
        ckpt_path = os.path.join(model_dir, "ckpt.pt")

    if not os.path.exists(ckpt_path):
        print(f"No checkpoint found at {ckpt_path}")
        sys.exit(1)

    print(f"Loading from {ckpt_path}...")
    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
    config = GPTConfig(**checkpoint['model_args'])
    model = GPT(config)
    model.load_state_dict(checkpoint['model'])
    model.eval()
    model.to(device)

    iter_num = checkpoint.get('iter_num', '?')
    best_val = checkpoint.get('best_val_loss', '?')
    print(f"  Loaded: iter {iter_num}, best val loss {best_val}")
    print(f"  Config: {config.n_layer}L, {config.n_head}H, {config.n_embd}E, vocab {config.vocab_size}")

    return model, config


def load_tokenizer():
    """Load SentencePiece tokenizer."""
    sp = spm.SentencePieceProcessor()
    sp.load(TOKENIZER_MODEL)
    return sp


@torch.no_grad()
def generate(model, sp, prompt, max_tokens=500, temperature=0.8, top_k=40, device='mps'):
    """Generate text from a prompt."""
    # Encode prompt; use BOS token if prompt is empty
    if prompt.strip():
        token_ids = sp.encode(prompt)
    else:
        token_ids = [sp.bos_id()]

    x = torch.tensor([token_ids], dtype=torch.long, device=device)

    # Generate
    for _ in range(max_tokens):
        # Crop to block_size if needed
        x_cond = x if x.size(1) <= model.config.block_size else x[:, -model.config.block_size:]

        logits, _ = model(x_cond)
        logits = logits[:, -1, :] / temperature

        # Top-k filtering
        if top_k is not None:
            v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            logits[logits < v[:, [-1]]] = -float('Inf')

        probs = torch.nn.functional.softmax(logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)

        # Stop on EOS
        if next_token.item() == sp.eos_id():
            break

        x = torch.cat([x, next_token], dim=1)

    # Decode
    output_ids = x[0].tolist()
    return sp.decode(output_ids)


def main():
    parser = argparse.ArgumentParser(description="Sample from trained classical model")
    parser.add_argument("--model", type=str, default="sequenced",
                        choices=["sequenced", "shuffled", "shuffled-docs"])
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Explicit path to checkpoint file. Overrides --model.")
    parser.add_argument("--prompt", type=str, default=None)
    parser.add_argument("--interactive", action="store_true")
    parser.add_argument("--max-tokens", type=int, default=500)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=40)
    parser.add_argument("--device", type=str, default="mps")
    parser.add_argument("--seed", type=int, default=1337,
                        help="Random seed for generation (ensures identical sampling across models in --compare)")
    parser.add_argument("--compare", action="store_true",
                        help="Sample from BOTH sequenced and shuffled with same prompt and same random state")
    args = parser.parse_args()

    sp = load_tokenizer()

    if args.compare:
        model_s, _ = load_model("sequenced", args.device)
        model_r, _ = load_model("shuffled", args.device)

        prompt = args.prompt or "The virtue of a man"
        print(f"\nPrompt: {prompt}")
        print(f"Seed: {args.seed}\n")

        print("=" * 60)
        print("SEQUENCED MODEL:")
        print("=" * 60)
        torch.manual_seed(args.seed)
        out_s = generate(model_s, sp, prompt, args.max_tokens, args.temperature, args.top_k, args.device)
        print(out_s)

        print("\n" + "=" * 60)
        print("SHUFFLED MODEL:")
        print("=" * 60)
        torch.manual_seed(args.seed)
        out_r = generate(model_r, sp, prompt, args.max_tokens, args.temperature, args.top_k, args.device)
        print(out_r)
        return

    model, config = load_model(args.model, args.device, ckpt_path=args.checkpoint)

    if args.interactive:
        print("\nInteractive mode. Type a prompt and press Enter. Type 'quit' to exit.\n")
        while True:
            try:
                prompt = input(">>> ")
            except (EOFError, KeyboardInterrupt):
                print()
                break
            if prompt.strip().lower() in ('quit', 'exit', 'q'):
                break
            if not prompt.strip():
                continue
            torch.manual_seed(args.seed)
            output = generate(model, sp, prompt, args.max_tokens, args.temperature, args.top_k, args.device)
            print(output)
            print()
    elif args.prompt:
        torch.manual_seed(args.seed)
        output = generate(model, sp, args.prompt, args.max_tokens, args.temperature, args.top_k, args.device)
        print(output)
    else:
        torch.manual_seed(args.seed)
        output = generate(model, sp, "", args.max_tokens, args.temperature, args.top_k, args.device)
        print(output)


if __name__ == "__main__":
    main()
