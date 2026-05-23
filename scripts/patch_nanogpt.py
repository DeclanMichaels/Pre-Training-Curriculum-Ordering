"""
Patch nanoGPT's train.py to support sequential (curriculum-ordered) data loading.

The default nanoGPT data loader picks random positions from the .bin file.
For the curriculum ordering experiment, the sequenced condition needs to
step through the file in order.

This script applies a minimal patch to train.py:
  1. Adds sequential_data config variable in the config section (BEFORE
     the configurator runs, so CLI and config file overrides work).
  2. Modifies get_batch() to read consecutive chunks when sequential_data=True.
  3. Validation always uses random sampling regardless of setting.

Usage: python scripts/patch_nanogpt.py [--nanogpt-dir nanoGPT]
       python scripts/patch_nanogpt.py --revert
"""
import os
import sys
import argparse
import shutil


def patch_train_py(content):
    """Apply the sequential data loading patch. Returns patched content or None on failure."""

    if 'sequential_data' in content:
        print("  Already patched.")
        return None

    # ---- Part 1: Add config variable BEFORE the configurator ----
    # The configurator line is: exec(open('configurator.py').read())
    # We insert the default BEFORE it so config files and CLI args can override.
    config_marker = "exec(open('configurator.py').read())"
    if config_marker not in content:
        # Try alternate form
        config_marker = 'exec(open("configurator.py").read())'
    if config_marker not in content:
        print("ERROR: Cannot find configurator.py exec line in train.py.")
        print("This version of nanoGPT may not be compatible with the patch.")
        return None

    config_insert = (
        "# --- PATCH: curriculum ordering experiment ---\n"
        "sequential_data = False  # True = step through file in order; False = random sampling\n"
        "\n"
    )
    content = content.replace(config_marker, config_insert + config_marker)

    # ---- Part 2: Add sequential state and modify get_batch ----
    # Find the random index line inside get_batch
    random_line = "    ix = torch.randint(len(data) - block_size, (batch_size,))"
    if random_line not in content:
        print("ERROR: Cannot find the random index line in get_batch().")
        print("Expected: " + random_line)
        return None

    # Add state variables just before get_batch
    get_batch_def = "def get_batch(split):"
    state_insert = (
        "# Sequential loading state\n"
        "_seq_pos = 0\n"
        "_seq_epoch = 0\n"
        "\n"
    )
    content = content.replace(
        get_batch_def,
        state_insert + get_batch_def,
        1
    )

    # Add global declaration at the start of get_batch body
    # Find the first line after def get_batch(split):
    content = content.replace(
        get_batch_def + "\n",
        get_batch_def + "\n"
        "    global _seq_pos, _seq_epoch\n",
        1
    )

    # Replace the random index line with conditional logic
    sequential_logic = (
        "    if split == 'train' and sequential_data:\n"
        "        # Sequential: step through file in curriculum order\n"
        "        max_pos = len(data) - block_size - 1\n"
        "        ix = []\n"
        "        for _ in range(batch_size):\n"
        "            if _seq_pos > max_pos:\n"
        "                _seq_pos = 0\n"
        "                _seq_epoch += 1\n"
        "            ix.append(_seq_pos)\n"
        "            _seq_pos += block_size\n"
        "        ix = torch.tensor(ix, dtype=torch.long)\n"
        "    else:\n"
        "        # Random: default nanoGPT behavior\n"
        "        ix = torch.randint(len(data) - block_size, (batch_size,))"
    )
    content = content.replace(random_line, sequential_logic)

    return content


def main():
    parser = argparse.ArgumentParser(description="Patch nanoGPT train.py for sequential loading")
    parser.add_argument("--nanogpt-dir", type=str, default="nanoGPT")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--revert", action="store_true", help="Revert to original")
    args = parser.parse_args()

    train_py = os.path.join(args.nanogpt_dir, "train.py")
    backup = train_py + ".original"

    if not os.path.exists(train_py):
        print(f"ERROR: {train_py} not found")
        sys.exit(1)

    if args.revert:
        if os.path.exists(backup):
            shutil.copy2(backup, train_py)
            print("Reverted to original train.py")
        else:
            print("No backup found. Cannot revert.")
        return

    with open(train_py, 'r') as f:
        content = f.read()

    patched = patch_train_py(content)
    if patched is None:
        return

    if args.dry_run:
        print("DRY RUN. Patch looks good. Run without --dry-run to apply.")
        return

    # Backup original
    if not os.path.exists(backup):
        shutil.copy2(train_py, backup)
        print(f"  Backed up original to {backup}")

    with open(train_py, 'w') as f:
        f.write(patched)

    print(f"  Patched {train_py}")
    print("")
    print("  Config variable 'sequential_data' added (default False).")
    print("  Set sequential_data=True in config or CLI for curriculum-ordered training.")
    print("  Validation always uses random sampling.")


if __name__ == "__main__":
    main()
