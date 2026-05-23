"""
Patch nanoGPT's train.py to support AttnRes model variant.

Adds a config variable 'use_attnres' (default False). When True,
imports GPT and GPTConfig from model_attnres.py instead of model.py.

The import is deferred to after the configurator runs, so CLI args
like --use_attnres=True work correctly.

Usage: python scripts/patch_attnres.py --nanogpt-dir nanoGPT

Prerequisites: model_attnres.py must be in the nanoGPT directory.
"""
import os
import sys
import argparse
import shutil


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--nanogpt-dir", default="nanoGPT")
    args = parser.parse_args()

    train_py = os.path.join(args.nanogpt_dir, "train.py")
    model_attnres = os.path.join(args.nanogpt_dir, "model_attnres.py")

    if not os.path.exists(train_py):
        print(f"ERROR: {train_py} not found")
        sys.exit(1)

    # Copy model_attnres.py into nanoGPT if not already there
    src = os.path.join(os.path.dirname(__file__), "..", "model_attnres.py")
    if os.path.exists(src) and not os.path.exists(model_attnres):
        shutil.copy2(src, model_attnres)
        print(f"  Copied model_attnres.py to {args.nanogpt_dir}/")

    with open(train_py, 'r') as f:
        content = f.read()

    if 'use_attnres' in content:
        print("  train.py already has AttnRes support.")
        return

    # Find the configurator line
    config_marker = "exec(open('configurator.py').read())"
    if config_marker not in content:
        config_marker = 'exec(open("configurator.py").read())'
    if config_marker not in content:
        print("ERROR: Cannot find configurator line in train.py")
        sys.exit(1)

    # Step 1: Remove the top-level model import
    old_import = "from model import GPTConfig, GPT"
    if old_import not in content:
        print("ERROR: Cannot find model import line in train.py")
        sys.exit(1)
    content = content.replace(old_import, "# model import moved below configurator (AttnRes patch)")

    # Step 2: Add use_attnres config variable before configurator
    content = content.replace(
        config_marker,
        "use_attnres = False  # True = use AttnRes model variant\n" + config_marker
    )

    # Step 3: Add conditional import AFTER configurator
    deferred_import = (
        "\n# --- AttnRes patch: deferred import after config is resolved ---\n"
        "if use_attnres:\n"
        "    from model_attnres import GPTConfig, GPT\n"
        "    print('Using AttnRes model variant')\n"
        "else:\n"
        "    from model import GPTConfig, GPT\n"
    )
    content = content.replace(
        config_marker,
        config_marker + deferred_import
    )

    with open(train_py, 'w') as f:
        f.write(content)

    print(f"  Patched {train_py} with AttnRes support.")
    print("  Set use_attnres=True in config or CLI to use AttnRes model.")


if __name__ == "__main__":
    main()
