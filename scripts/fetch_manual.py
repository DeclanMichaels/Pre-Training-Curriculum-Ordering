"""
Fetch the manual texts that aren't on Project Gutenberg.
Each text has a custom fetch strategy (MIT Classics HTML, archive.org, ESP PDF).

Usage: python scripts/fetch_manual.py [--dry-run] [--text SLUG]

Handles deduplication: Newton's Principia and Galileo's Dialogue appear in
both Stage 0 and Stage 5. Fetched once, copied to both locations.
"""
import html
import json
import os
import re
import sys
import time
import urllib.request
import urllib.error
import argparse

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
RAW_DIR = os.path.join(PROJECT_ROOT, "corpus", "raw")
DELAY = 2

# User-Agent for polite fetching
HEADERS = {"User-Agent": "ScratchTrainingCuration/1.0 (research; declan@moral-os.com)"}


def fetch_url(url, as_bytes=False):
    """Fetch a URL, return text or bytes."""
    req = urllib.request.Request(url, headers=HEADERS)
    with urllib.request.urlopen(req, timeout=60) as resp:
        data = resp.read()
        if as_bytes:
            return data
        return data.decode('utf-8', errors='replace')


def strip_html(text):
    """Remove HTML tags and decode entities. Simple but sufficient for MIT Classics."""
    text = re.sub(r'<script[^>]*>.*?</script>', '', text, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r'<style[^>]*>.*?</style>', '', text, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r'<[^>]+>', '\n', text)
    text = html.unescape(text)
    # Collapse whitespace
    lines = [line.strip() for line in text.split('\n')]
    lines = [l for l in lines if l]
    return '\n'.join(lines)


def save_text(dest_path, text):
    """Save text to file, creating directories as needed."""
    os.makedirs(os.path.dirname(dest_path), exist_ok=True)
    with open(dest_path, 'w', encoding='utf-8') as f:
        f.write(text)
    size_kb = len(text.encode('utf-8')) / 1024
    print(f"  Saved: {size_kb:.0f} KB -> {os.path.relpath(dest_path, PROJECT_ROOT)}")


def copy_to_stage(src_path, stage, seq, slug):
    """Copy a fetched file to another stage directory (for duplicates)."""
    import shutil
    dest_dir = os.path.join(RAW_DIR, f"stage-{stage}")
    os.makedirs(dest_dir, exist_ok=True)
    dest_path = os.path.join(dest_dir, f"{seq:02d}-{slug}.txt")
    if os.path.exists(dest_path):
        print(f"  [SKIP COPY] Already exists: {os.path.relpath(dest_path, PROJECT_ROOT)}")
        return
    shutil.copy2(src_path, dest_path)
    print(f"  Copied to: {os.path.relpath(dest_path, PROJECT_ROOT)}")


# ============================================================
# Individual fetch functions for each manual text
# ============================================================

def fetch_aristotle_history_of_animals(dest_path, dry_run=False):
    """Archive.org: Thompson's Oxford 1910 translation (full OCR text)."""
    url = "https://archive.org/download/worksofaristotle04arisuoft/worksofaristotle04arisuoft_djvu.txt"
    print(f"  Fetching from archive.org (Thompson 1910 translation)...")
    if dry_run:
        print(f"  [DRY RUN] Would fetch {url}")
        return True
    try:
        text = fetch_url(url)
        # The archive text includes the full Oxford volume.
        # Add header note about subsetting.
        header = (
            "HISTORY OF ANIMALS\n"
            "by Aristotle\n"
            "translated by D'Arcy Wentworth Thompson (Oxford, 1910)\n\n"
            "NOTE: This is the full Oxford volume including editorial apparatus.\n"
            "The actual text of History of Animals begins after the introduction.\n\n"
        )
        save_text(dest_path, header + text)
        return True
    except Exception as e:
        print(f"  FAILED: {e}")
        return False


def fetch_hippocrates_airs(dest_path, dry_run=False):
    """GitHub mirror of MIT Classics (site itself is broken as of 2023)."""
    url = "https://raw.githubusercontent.com/TheMITTech/classics/master/Hippocrates/airwatpl.mb.txt"
    print(f"  Fetching from MIT Classics GitHub mirror...")
    if dry_run:
        print(f"  [DRY RUN] Would fetch {url}")
        return True
    try:
        text = fetch_url(url)
        if len(text) < 1000:
            raise ValueError(f"Text too short ({len(text)} chars)")
        save_text(dest_path, text)
        return True
    except Exception as e:
        print(f"  FAILED: {e}")
        return False


def fetch_newton_principia(dest_path, dry_run=False):
    """Archive.org: Motte translation. Fetch full text, extract Rules + General Scholium."""
    # The 1846 American edition has full OCR text available
    url = "https://archive.org/download/newtonspmathema00newtrich/newtonspmathema00newtrich_djvu.txt"
    print(f"  Fetching full Principia text from archive.org...")
    if dry_run:
        print(f"  [DRY RUN] Would fetch {url}")
        return True
    try:
        text = fetch_url(url)
        # The full text is ~1.5MB. We want Rules of Reasoning and General Scholium.
        # Save full text first, note in header that subsetting is recommended.
        header = (
            "NEWTON'S PRINCIPIA\n"
            "The Mathematical Principles of Natural Philosophy\n"
            "Translated by Andrew Motte (1729)\n\n"
            "NOTE: This is the full text. For the training corpus, extract:\n"
            "  - Rules of Reasoning in Philosophy (Book III, opening)\n"
            "  - General Scholium (Book III, end)\n"
            "  - Definitions and Laws of Motion (Book I, opening)\n\n"
        )
        save_text(dest_path, header + text)
        return True
    except Exception as e:
        print(f"  FAILED: {e}")
        return False


def fetch_galileo_dialogue(dest_path, dry_run=False):
    """Archive.org: Dialogue Concerning the Two Chief World Systems."""
    # The Crew translation (1953) might not be PD, but there are older translations
    # Try the 1661 Salusbury or the Stillman Drake. Let's try full text from archive.org
    url = "https://archive.org/download/dialogueconcerni00telerich/dialogueconcerni00telerich_djvu.txt"
    print(f"  Fetching Galileo's Dialogue from archive.org...")
    if dry_run:
        print(f"  [DRY RUN] Would fetch {url}")
        return True
    try:
        text = fetch_url(url)
        header = (
            "DIALOGUE CONCERNING THE TWO CHIEF WORLD SYSTEMS\n"
            "by Galileo Galilei\n\n"
            "NOTE: This is the full text. For the training corpus, select key passages\n"
            "on observation vs authority, falling bodies, and the nature of evidence.\n\n"
        )
        save_text(dest_path, header + text)
        return True
    except Exception as e:
        print(f"  FAILED: {e}")
        # Fallback: try alternate archive.org ID
        alt_url = "https://archive.org/download/galileodialoguco00galirich/galileodialoguco00galirich_djvu.txt"
        print(f"  Trying alternate: {alt_url}")
        try:
            text = fetch_url(alt_url)
            save_text(dest_path, header + text)
            return True
        except Exception as e2:
            print(f"  FAILED again: {e2}")
            print(f"  Manual download needed. Try: https://archive.org/search?query=galileo+dialogue+two+chief+world+systems")
            return False


def fetch_pilpay_fables(dest_path, dry_run=False):
    """Archive.org: Fables of Bidpai / Pilpay, Jacobs 1888 reprint."""
    url = "https://archive.org/download/faboribidpaipilp00bidprich/faboribidpaipilp00bidprich_djvu.txt"
    print(f"  Fetching Pilpay's Fables from archive.org...")
    if dry_run:
        print(f"  [DRY RUN] Would fetch {url}")
        return True
    try:
        text = fetch_url(url)
        save_text(dest_path, text)
        return True
    except Exception as e:
        print(f"  FAILED: {e}")
        # Try alternate
        alt = "https://archive.org/download/faboribidpaipilp00bidpuoft/faboribidpaipilp00bidpuoft_djvu.txt"
        print(f"  Trying alternate...")
        try:
            text = fetch_url(alt)
            save_text(dest_path, text)
            return True
        except:
            print(f"  Manual download needed: https://archive.org/search?query=fables+bidpai+pilpay")
            return False


def fetch_weber_protestant_ethic(dest_path, dry_run=False):
    """Weber's Protestant Ethic - Parsons 1930 trans, PD as of Jan 2026."""
    # Check archive.org for the Parsons translation
    url = "https://archive.org/download/protestantethics0000webe/protestantethics0000webe_djvu.txt"
    print(f"  Fetching Weber's Protestant Ethic from archive.org...")
    if dry_run:
        print(f"  [DRY RUN] Would fetch {url}")
        return True
    try:
        text = fetch_url(url)
        if len(text) < 5000:
            raise ValueError(f"Text too short ({len(text)} chars), likely an error page")
        save_text(dest_path, text)
        return True
    except Exception as e:
        print(f"  FAILED: {e}")
        print(f"  The Parsons 1930 translation entered PD Jan 1, 2026.")
        print(f"  Search: https://archive.org/search?query=weber+protestant+ethic+parsons")
        print(f"  Or try Project Gutenberg (may have been added recently).")
        return False


def fetch_mendel_experiments(dest_path, dry_run=False):
    """Mendel's Experiments in Plant Hybridization - ESP or MendelWeb."""
    # MendelWeb has a clean HTML version
    url = "http://www.mendelweb.org/Mendel.plain.html"
    print(f"  Fetching Mendel from MendelWeb...")
    if dry_run:
        print(f"  [DRY RUN] Would fetch {url}")
        return True
    try:
        raw = fetch_url(url)
        text = strip_html(raw)
        if len(text) < 1000:
            raise ValueError("Too short")
        header = "EXPERIMENTS IN PLANT HYBRIDIZATION\nby Gregor Mendel (1865)\n\n"
        save_text(dest_path, header + text)
        return True
    except Exception as e:
        print(f"  FAILED: {e}")
        # Fallback to ESP
        alt = "http://www.esp.org/foundations/genetics/classical/gm-65-a.html"
        print(f"  Trying ESP...")
        try:
            raw = fetch_url(alt)
            text = strip_html(raw)
            save_text(dest_path, "EXPERIMENTS IN PLANT HYBRIDIZATION\nby Gregor Mendel (1865)\n\n" + text)
            return True
        except Exception as e2:
            print(f"  FAILED: {e2}")
            return False


def fetch_einstein_relativity(dest_path, dry_run=False):
    """Einstein's Relativity - Lawson 1920 translation, PD."""
    # This popular exposition is on archive.org and also Gutenberg
    # Check Gutenberg first: "Relativity: The Special and General Theory"
    gutenberg_url = "https://www.gutenberg.org/cache/epub/5001/pg5001.txt"
    print(f"  Trying Gutenberg #5001 (Relativity: Special and General Theory)...")
    if dry_run:
        print(f"  [DRY RUN] Would fetch {gutenberg_url}")
        return True
    try:
        text = fetch_url(gutenberg_url)
        if len(text) > 5000:
            save_text(dest_path, text)
            return True
    except:
        pass

    # Fallback to archive.org
    url = "https://archive.org/download/cu31924011804774/cu31924011804774_djvu.txt"
    print(f"  Trying archive.org...")
    try:
        text = fetch_url(url)
        save_text(dest_path, text)
        return True
    except Exception as e:
        print(f"  FAILED: {e}")
        return False


# ============================================================
# Main: dispatch table
# ============================================================

MANUAL_TEXTS = [
    {
        "slug": "history-of-animals-selections",
        "title": "History of Animals",
        "stage": 0, "seq": 6,
        "fetcher": fetch_aristotle_history_of_animals,
    },
    {
        "slug": "principia-rules-and-general-scholium",
        "title": "Principia (Rules & General Scholium)",
        "stage": 0, "seq": 13,
        "fetcher": fetch_newton_principia,
        "copies": [{"stage": 5, "seq": 6, "slug": "principia-mathematica-selections"}],
    },
    {
        "slug": "dialogue-on-the-two-chief-world-systems-excerpts",
        "title": "Galileo's Dialogue",
        "stage": 0, "seq": 14,
        "fetcher": fetch_galileo_dialogue,
        "copies": [{"stage": 5, "seq": 5, "slug": "dialogue-concerning-two-chief-world-systems"}],
    },
    {
        "slug": "pilpay-s-fables-bidpai",
        "title": "Pilpay's Fables (Bidpai)",
        "stage": 1, "seq": 8,
        "fetcher": fetch_pilpay_fables,
    },
    {
        "slug": "the-protestant-ethic-and-spirit-of-capitalism",
        "title": "Protestant Ethic (Weber)",
        "stage": 4, "seq": 10,
        "fetcher": fetch_weber_protestant_ethic,
    },
    {
        "slug": "on-airs-waters-and-places",
        "title": "On Airs, Waters, and Places (Hippocrates)",
        "stage": 5, "seq": 1,
        "fetcher": fetch_hippocrates_airs,
    },
    {
        "slug": "experiments-in-plant-hybridization",
        "title": "Experiments in Plant Hybridization (Mendel)",
        "stage": 5, "seq": 8,
        "fetcher": fetch_mendel_experiments,
    },
    {
        "slug": "the-general-theory-of-relativity",
        "title": "Relativity (Einstein)",
        "stage": 5, "seq": 9,
        "fetcher": fetch_einstein_relativity,
    },
]


def main():
    parser = argparse.ArgumentParser(description="Fetch manual (non-Gutenberg) texts")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--text", type=str, default=None,
                        help="Fetch only this text (by slug substring)")
    args = parser.parse_args()

    texts = MANUAL_TEXTS
    if args.text:
        texts = [t for t in texts if args.text.lower() in t["slug"].lower()]
        if not texts:
            print(f"No text matching '{args.text}'. Available:")
            for t in MANUAL_TEXTS:
                print(f"  {t['slug']}")
            sys.exit(1)

    print(f"Fetching {len(texts)} manual texts\n")

    fetched = 0
    failed = 0

    for text in texts:
        stage_dir = os.path.join(RAW_DIR, f"stage-{text['stage']}")
        os.makedirs(stage_dir, exist_ok=True)
        dest = os.path.join(stage_dir, f"{text['seq']:02d}-{text['slug']}.txt")

        if os.path.exists(dest) and not args.dry_run:
            size = os.path.getsize(dest) / 1024
            print(f"[SKIP] {text['title']} (already exists, {size:.0f} KB)")
            fetched += 1
            continue

        print(f"[FETCH] {text['title']}")
        ok = text["fetcher"](dest, dry_run=args.dry_run)

        if ok:
            fetched += 1
            # Handle duplicate copies
            if "copies" in text and not args.dry_run:
                for copy in text["copies"]:
                    print(f"  Copying to Stage {copy['stage']}...")
                    copy_to_stage(dest, copy["stage"], copy["seq"], copy["slug"])
        else:
            failed += 1

        if not args.dry_run:
            time.sleep(DELAY)

        print()

    print(f"Done. Fetched: {fetched}, Failed: {failed}")


if __name__ == "__main__":
    main()
