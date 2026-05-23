"""
Fetch texts from Project Gutenberg plain-text endpoint.
Reads manifest.json and downloads auto-fetchable texts into corpus/raw/stage-N/.

Usage: python scripts/fetch.py [--dry-run] [--stage N]

Gutenberg plain text URL pattern:
  https://www.gutenberg.org/cache/epub/{id}/pg{id}.txt

Rate limiting: 2 second delay between requests (be polite to Gutenberg).
"""
import json
import os
import sys
import time
import urllib.request
import urllib.error
import argparse

MANIFEST = "config/manifest.json"
RAW_DIR = "corpus/raw"
GUTENBERG_TXT_URL = "https://www.gutenberg.org/cache/epub/{gid}/pg{gid}.txt"
GUTENBERG_UTF8_URL = "https://www.gutenberg.org/files/{gid}/{gid}-0.txt"
DELAY = 2  # seconds between requests
MIN_FILE_BYTES = 1024  # reject downloads smaller than 1KB (likely error pages)


def fetch_gutenberg(gid, dest_path, dry_run=False):
    """Try primary URL, fall back to alternate. Validates file size."""
    urls = [
        GUTENBERG_TXT_URL.format(gid=gid),
        GUTENBERG_UTF8_URL.format(gid=gid),
    ]

    if dry_run:
        print(f"  [DRY RUN] Would fetch {urls[0]}")
        return True

    for url in urls:
        try:
            req = urllib.request.Request(url, headers={
                "User-Agent": "ScratchTrainingCuration/1.0 (research; declan@moral-os.com)"
            })
            with urllib.request.urlopen(req, timeout=30) as resp:
                data = resp.read()

                if len(data) < MIN_FILE_BYTES:
                    print(f"  REJECTED: {len(data)} bytes from {url} (below {MIN_FILE_BYTES}B minimum)")
                    continue

                with open(dest_path, "wb") as f:
                    f.write(data)
                size_kb = len(data) / 1024
                print(f"  OK: {size_kb:.0f} KB from {url}")
                return True
        except urllib.error.HTTPError as e:
            if e.code == 404:
                continue
            print(f"  HTTP {e.code} from {url}")
            continue
        except Exception as e:
            print(f"  Error: {e}")
            continue

    print(f"  FAILED: no working URL for Gutenberg ID {gid}")
    return False


def main():
    parser = argparse.ArgumentParser(description="Fetch texts from Project Gutenberg")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be fetched")
    parser.add_argument("--stage", type=int, default=None, help="Fetch only this stage")
    parser.add_argument("--force", action="store_true", help="Re-download existing files")
    args = parser.parse_args()

    with open(MANIFEST) as f:
        manifest = json.load(f)

    auto_texts = [t for t in manifest["texts"] if t["fetch_status"] == "auto"]
    if args.stage is not None:
        auto_texts = [t for t in auto_texts if t["stage"] == args.stage]

    print(f"Fetching {len(auto_texts)} texts from Project Gutenberg")
    if args.dry_run:
        print("(DRY RUN)")
    print()

    fetched = 0
    skipped = 0
    failed = 0

    for text in auto_texts:
        stage_dir = os.path.join(RAW_DIR, f"stage-{text['stage']}")
        os.makedirs(stage_dir, exist_ok=True)

        # seq is guaranteed int from build_manifest, but belt-and-suspenders
        seq = int(text["seq"])
        filename = f"{seq:02d}-{text['slug']}.txt"
        dest = os.path.join(stage_dir, filename)

        if os.path.exists(dest) and not args.force:
            print(f"[SKIP] {text['title']} (already exists)")
            skipped += 1
            continue

        print(f"[FETCH] Stage {text['stage']}, #{seq}: {text['title']}")
        print(f"  Gutenberg ID: {text['gutenberg_id']}")

        ok = fetch_gutenberg(text["gutenberg_id"], dest, dry_run=args.dry_run)

        if ok:
            fetched += 1
        else:
            failed += 1

        if not args.dry_run:
            time.sleep(DELAY)

    print(f"\nDone. Fetched: {fetched}, Skipped: {skipped}, Failed: {failed}")

    manual = [t for t in manifest["texts"] if t["fetch_status"] == "manual"]
    if args.stage is not None:
        manual = [t for t in manual if t["stage"] == args.stage]
    if manual:
        print(f"\n{len(manual)} texts need manual download:")
        for t in manual:
            seq = int(t["seq"])
            print(f"  Stage {t['stage']}: {t['title']}")
            print(f"    Hint: {t['source_url']}")
            stage_dir = os.path.join(RAW_DIR, f"stage-{t['stage']}")
            filename = f"{seq:02d}-{t['slug']}.txt"
            print(f"    Save to: {os.path.join(stage_dir, filename)}")


if __name__ == "__main__":
    main()
