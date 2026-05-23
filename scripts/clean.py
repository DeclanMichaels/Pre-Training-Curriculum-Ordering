"""
Clean raw Gutenberg texts: strip headers/footers, normalize whitespace,
remove editorial artifacts. Saves to corpus/clean/.

Usage: python scripts/clean.py [--stage N] [--verbose]

Two-pass cleaning:
  Pass 1 (regex): Gutenberg boilerplate removal, line normalization.
  Pass 2 (optional, future): LLM-assisted removal of editorial insertions,
    OCR artifacts, translator footnotes. Not implemented yet.
"""
import os
import re
import sys
import json
import argparse

RAW_DIR = "corpus/raw"
CLEAN_DIR = "corpus/clean"
LOG_DIR = "corpus/clean/_logs"

# Max lines to scan for preamble residue after Gutenberg marker stripping.
# Prevents eating legitimate text that happens to match a preamble pattern.
MAX_PREAMBLE_LINES = 30

GUTENBERG_START_MARKERS = [
    r'\*\*\* ?START OF (?:THE|THIS) PROJECT GUTENBERG',
    r'\*\*\*START OF (?:THE|THIS) PROJECT GUTENBERG',
]
GUTENBERG_END_MARKERS = [
    r'\*\*\* ?END OF (?:THE|THIS) PROJECT GUTENBERG',
    r'\*\*\*END OF (?:THE|THIS) PROJECT GUTENBERG',
    r'End of (?:the )?Project Gutenberg',
]

# Patterns that indicate Gutenberg production metadata (not book content)
PREAMBLE_PATTERNS = [
    re.compile(r'^Produced by\b', re.IGNORECASE),
    re.compile(r'^This eBook\b', re.IGNORECASE),
    re.compile(r'^This ebook\b', re.IGNORECASE),
    re.compile(r'^Updated editions\b', re.IGNORECASE),
    re.compile(r'^Character set\b', re.IGNORECASE),
    re.compile(r'^Credits:', re.IGNORECASE),
    re.compile(r'^Transcriber', re.IGNORECASE),
    re.compile(r'^E-?text prepared by\b', re.IGNORECASE),
    re.compile(r'^Language:', re.IGNORECASE),
    re.compile(r'^Release [Dd]ate:', re.IGNORECASE),
    re.compile(r'^Italic text is denoted', re.IGNORECASE),
    re.compile(r'^Small uppercase', re.IGNORECASE),
    re.compile(r'^Blank pages have been', re.IGNORECASE),
    re.compile(r'^Variations in spelling', re.IGNORECASE),
    re.compile(r'^A few typographical', re.IGNORECASE),
    re.compile(r'^The cover page was', re.IGNORECASE),
    re.compile(r'^Note:', re.IGNORECASE),
]


def strip_gutenberg_boilerplate(text):
    """Remove Gutenberg header and footer, return body text."""
    lines = text.split('\n')
    start_idx = 0
    end_idx = len(lines)

    # Find start marker
    for i, line in enumerate(lines):
        for pat in GUTENBERG_START_MARKERS:
            if re.search(pat, line, re.IGNORECASE):
                start_idx = i + 1
                # Skip blank lines after marker
                while start_idx < len(lines) and lines[start_idx].strip() == '':
                    start_idx += 1
                break
        if start_idx > 0:
            break

    # Find end marker (search from end to avoid matching mid-text references)
    for i in range(len(lines) - 1, max(start_idx, len(lines) - 200), -1):
        for pat in GUTENBERG_END_MARKERS:
            if re.search(pat, lines[i], re.IGNORECASE):
                end_idx = i
                while end_idx > start_idx and lines[end_idx - 1].strip() == '':
                    end_idx -= 1
                break
        if end_idx < len(lines):
            break

    body = '\n'.join(lines[start_idx:end_idx])
    header_removed = start_idx
    footer_removed = len(lines) - end_idx
    return body, header_removed, footer_removed


def normalize_whitespace(text):
    """Normalize line endings, collapse excessive blank lines, strip trailing spaces."""
    text = text.replace('\r\n', '\n').replace('\r', '\n')
    lines = [line.rstrip() for line in text.split('\n')]
    result = []
    blank_count = 0
    for line in lines:
        if line == '':
            blank_count += 1
            if blank_count <= 2:
                result.append(line)
        else:
            blank_count = 0
            result.append(line)
    return '\n'.join(result).strip() + '\n'


def strip_gutenberg_preamble_text(text):
    """Remove Gutenberg production metadata that survives marker stripping.

    Handles multi-line transcriber notes where continuation lines (like
    '=equal signs=.' or 'original.') don't match any pattern themselves.

    Only examines the first MAX_PREAMBLE_LINES lines to avoid eating
    legitimate text that happens to match a pattern.
    """
    lines = text.split('\n')
    if not lines:
        return text

    cut_at = 0
    matched_any_pattern = False

    for i, line in enumerate(lines):
        if i >= MAX_PREAMBLE_LINES:
            break
        stripped = line.strip()

        # Blank lines: always skip in preamble zone
        if stripped == '':
            cut_at = i + 1
            continue

        # Known preamble patterns
        is_preamble = any(pat.match(stripped) for pat in PREAMBLE_PATTERNS)
        if is_preamble:
            matched_any_pattern = True
            cut_at = i + 1
            continue

        # Short continuation lines after a pattern match (e.g. '=equal signs=.',
        # 'original.', '  A few typos corrected.'). These are wrapped continuations
        # of preamble lines, not book content.
        if matched_any_pattern and len(stripped) < 80:
            cut_at = i + 1
            continue

        # First substantial line that doesn't match: stop
        break

    # Skip trailing blank lines after preamble
    while cut_at < len(lines) and lines[cut_at].strip() == '':
        cut_at += 1

    return '\n'.join(lines[cut_at:])


def clean_file(raw_path, clean_path, log_path=None):
    """Clean a single file. Returns stats dict."""
    with open(raw_path, 'r', encoding='utf-8', errors='replace') as f:
        raw = f.read()

    raw_chars = len(raw)

    body, header_lines, footer_lines = strip_gutenberg_boilerplate(raw)
    body = strip_gutenberg_preamble_text(body)
    body = normalize_whitespace(body)

    clean_chars = len(body)

    os.makedirs(os.path.dirname(clean_path), exist_ok=True)
    with open(clean_path, 'w', encoding='utf-8') as f:
        f.write(body)

    stats = {
        "raw_chars": raw_chars,
        "clean_chars": clean_chars,
        "reduction_pct": round((1 - clean_chars / raw_chars) * 100, 1) if raw_chars > 0 else 0,
        "header_lines_removed": header_lines,
        "footer_lines_removed": footer_lines,
    }

    if log_path:
        os.makedirs(os.path.dirname(log_path), exist_ok=True)
        with open(log_path, 'w') as f:
            json.dump(stats, f, indent=2)

    return stats


def main():
    parser = argparse.ArgumentParser(description="Clean raw Gutenberg texts")
    parser.add_argument("--stage", type=int, default=None, help="Clean only this stage")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--force", action="store_true", help="Re-clean existing files")
    args = parser.parse_args()

    if not os.path.isdir(RAW_DIR):
        print(f"Raw directory not found: {RAW_DIR}")
        print("Run scripts/fetch.py first.")
        sys.exit(1)

    total_raw = 0
    total_clean = 0
    file_count = 0
    skip_count = 0

    if args.stage is not None:
        stage_name = f"stage-{args.stage}"
        if not os.path.isdir(os.path.join(RAW_DIR, stage_name)):
            print(f"Warning: {os.path.join(RAW_DIR, stage_name)} does not exist.")
            print(f"Available stages: {sorted(d for d in os.listdir(RAW_DIR) if d.startswith('stage-'))}")
            sys.exit(1)
        stages = [stage_name]
    else:
        stages = sorted(d for d in os.listdir(RAW_DIR) if d.startswith('stage-'))

    for stage_dir_name in stages:
        raw_stage = os.path.join(RAW_DIR, stage_dir_name)
        clean_stage = os.path.join(CLEAN_DIR, stage_dir_name)
        log_stage = os.path.join(LOG_DIR, stage_dir_name)

        files = sorted(f for f in os.listdir(raw_stage) if f.endswith('.txt'))
        print(f"\n{stage_dir_name}: {len(files)} files")

        for fname in files:
            raw_path = os.path.join(raw_stage, fname)
            clean_path = os.path.join(clean_stage, fname)
            log_path = os.path.join(log_stage, fname.replace('.txt', '.json'))

            if os.path.exists(clean_path) and not args.force:
                skip_count += 1
                if args.verbose:
                    print(f"  [SKIP] {fname}")
                continue

            stats = clean_file(raw_path, clean_path, log_path)
            file_count += 1
            total_raw += stats["raw_chars"]
            total_clean += stats["clean_chars"]

            if args.verbose:
                print(f"  {fname}: {stats['raw_chars']//1024}KB > {stats['clean_chars']//1024}KB ({stats['reduction_pct']}% removed)")

    print(f"\nCleaned {file_count} files" + (f", skipped {skip_count} (already clean)" if skip_count else ""))
    if total_raw > 0:
        print(f"Total: {total_raw//1024}KB raw > {total_clean//1024}KB clean ({round((1 - total_clean/total_raw) * 100, 1)}% reduction)")


if __name__ == "__main__":
    main()
