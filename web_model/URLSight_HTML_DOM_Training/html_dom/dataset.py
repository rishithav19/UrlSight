from pathlib import Path
from .features import extract_dom_features
import time


def label_from_folder(name):
    if name.startswith("P"):
        return 1
    elif name.startswith("L"):
        return 0
    else:
        raise ValueError(f"Unknown class prefix: {name}")


def load_split(dataset_root, split):

    root = Path(dataset_root) / split

    if not root.exists():
        raise FileNotFoundError(f"Dataset folder not found: {root}")

    folders = sorted(p for p in root.iterdir() if p.is_dir())

    records = []
    skipped = []

    total = len(folders)
    start = time.time()

    print(f"\n[{split.upper()}] Starting HTML extraction...", flush=True)
    print(f"Total samples found: {total}", flush=True)

    for index, folder in enumerate(folders, start=1):

        try:
            label = label_from_folder(folder.name)
        except ValueError as e:
            skipped.append({
                "split": split,
                "sample": folder.name,
                "reason": str(e)
            })
            continue

        html_path = folder / "RAW-HTML/index.html"
        url_path = folder / "URL/url.txt"
        label_path = folder / "Label/label.txt"

        if not html_path.is_file() or html_path.stat().st_size == 0:
            skipped.append({
                "split": split,
                "sample": folder.name,
                "reason": "missing_or_empty_html"
            })

        else:
            try:
                html = html_path.read_text(
                    encoding="utf-8",
                    errors="replace"
                )

                url = (
                    url_path.read_text(
                        encoding="utf-8-sig",
                        errors="replace"
                    ).strip()
                    if url_path.is_file() else ""
                )

                category = (
                    label_path.read_text(
                        encoding="utf-8-sig",
                        errors="replace"
                    ).strip()
                    if label_path.is_file() else ""
                )

                if not html.strip():
                    raise ValueError("Empty HTML")

                dom_features, html_text = extract_dom_features(
                    html, url
                )

                records.append({
                    "sample_id": folder.name,
                    "label": label,
                    "class_name": (
                        "phishing" if label == 1 else "legitimate"
                    ),
                    "category": category,
                    "page_url": url,
                    "html_text": html_text,
                    "dom_features": dom_features
                })

            except Exception as e:
                skipped.append({
                    "split": split,
                    "sample": folder.name,
                    "reason": f"parse_error: {type(e).__name__}: {e}"
                })

        if index % 100 == 0 or index == total:
            elapsed = time.time() - start
            print(
                f"[{split.upper()}] Processed {index}/{total} | "
                f"Extracted: {len(records)} | "
                f"Skipped: {len(skipped)} | "
                f"Time: {elapsed:.1f}s",
                flush=True
            )

    elapsed = time.time() - start

    print(f"\n[{split.upper()}] Extraction completed!", flush=True)
    print(f"Successfully extracted: {len(records)}", flush=True)
    print(f"Skipped: {len(skipped)}", flush=True)
    print(f"Time taken: {elapsed:.2f} seconds", flush=True)

    return records, skipped
