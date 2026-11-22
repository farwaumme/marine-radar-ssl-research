
import hashlib
import json
from collections import Counter
from pathlib import Path

def all5_inventory(a):
    document = json.loads(a.training_manifest.read_text())
    external = document["records"]

    # Validate external group independence and path uniqueness.
    groups = {}
    paths = set()
    for r in external:
        key = (r["source"], r["group"])
        if key in groups and groups[key] != r["split"]:
            raise ValueError(f"External group crosses splits: {key}")
        groups[key] = r["split"]
        if r["path"] in paths:
            raise ValueError(f"Duplicate external path: {r['path']}")
        paths.add(r["path"])

    # Match MAE's training-record fields and ordering exactly.
    training = []
    for r in external:
        if r["split"] == "train":
            training.append({
                "path": str(Path(r["path"]).resolve()),
                "source": r["source"],
                "group": r["group"],
                "strip11": bool(r["strip11"]),
                "representation": r["representation"].lower(),
            })

    if len({r["source"] for r in training}) != 4:
        raise ValueError("Expected four external dataset sources")

    root = a.professor_root.expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(root)

    extensions = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
    for directory in sorted(root.rglob("*")):
        if not directory.is_dir() or directory.name.lower() != "train":
            continue
        parts = [p.upper() for p in directory.relative_to(root).parts]
        representation = "polar" if "POLAR" in parts else "cartesian"
        for path in sorted(directory.iterdir()):
            if path.is_file() and path.suffix.lower() in extensions:
                training.append({
                    "path": str(path.resolve()),
                    "source": "Professor",
                    "group": str(directory.parent.relative_to(root)),
                    "strip11": False,
                    "representation": representation,
                })

    counts = Counter(r["source"] for r in training)
    if counts["Professor"] != 2520:
        raise ValueError(f"Expected 2520 professor training images: {counts}")
    if len(training) != 415244:
        raise ValueError(f"Expected 415244 training paths; found {len(training)}")

    selected_paths = [r["path"] for r in training]
    if len(selected_paths) != len(set(selected_paths)):
        raise ValueError("Duplicate training paths")

    # Exact path/metadata comparison against the MAE inventory when supplied.
    if a.mae_manifest:
        mae_records = json.loads(a.mae_manifest.read_text())["records"]
        if training != mae_records:
            raise ValueError("Training records differ from the saved MAE inventory")
        print("CONFIRMED: exact training-record match with MAE.", flush=True)

    for r in training:
        if not Path(r["path"]).is_file():
            raise FileNotFoundError(r["path"])

    print("AVAILABLE TRAINING IMAGES", flush=True)
    for source, count in sorted(counts.items()):
        print(f"  {source}: {count:,}", flush=True)
    print(f"Total: {len(training):,}", flush=True)
    print("Sampling: without replacement; final batch retained.", flush=True)

    # Both objectives require at least two images in a batch.
    if len(training) < a.batch_size or len(training) % a.batch_size == 1:
        raise ValueError(
            "Batch setting creates an invalid batch of fewer than two images. "
            "Choose another batch size, such as 8, 16, 32 or 64."
        )

    digest = hashlib.sha256(
        json.dumps(training, sort_keys=True).encode()
    ).hexdigest()
    print("Training selection SHA256:", digest, flush=True)

    return [dict(r, split="train") for r in training], training
