"""
evaluate_test_set.py
====================
Runs model.val() on BOTH the validation and test splits of the
WasteManagement-2 dataset and counts bounding-box instances per class
for all three splits (train / val / test).

Run from the D:/final_project directory:
    python evaluate_test_set.py

Outputs a summary you can paste directly into the paper / report.
"""

import os
import sys
from pathlib import Path

# ── Paths ────────────────────────────────────────────────────────────────────
MODEL_PATH  = r"D:\final_project\codebase\BEST_MODELS\best_0944_final_wiou_bifpn_100ep.pt"
DATA_YAML   = r"D:\final_project\wastemanagement-2\data.yaml"
DATASET_ROOT = r"D:\final_project"   # parent of train / valid / test folders

# ── Imports ──────────────────────────────────────────────────────────────────
try:
    from ultralytics import YOLO
except ImportError:
    sys.exit("ultralytics not installed. Run:  pip install ultralytics")

# ── Helper: count bounding boxes per class in a label directory ───────────────
def count_boxes(label_dir: Path, class_names: list[str]) -> dict:
    counts = {name: 0 for name in class_names}
    total  = 0
    n_files = 0
    for txt in label_dir.glob("*.txt"):
        n_files += 1
        for line in txt.read_text().splitlines():
            line = line.strip()
            if not line:
                continue
            cls_id = int(line.split()[0])
            if cls_id < len(class_names):
                counts[class_names[cls_id]] += 1
                total += 1
    return {"files": n_files, "counts": counts, "total": total}


def print_split_stats(split_name: str, images_dir: Path, label_dir: Path,
                      class_names: list[str]):
    stats = count_boxes(label_dir, class_names)
    print(f"\n  {split_name} split")
    print(f"    Images : {stats['files']}")
    for cls, n in stats["counts"].items():
        print(f"    {cls:10s}: {n} boxes")
    print(f"    Total  : {stats['total']} boxes")
    return stats


# ── Main ─────────────────────────────────────────────────────────────────────
def main():
    print("=" * 60)
    print("  Smart Waste Segregation — Test-Set Evaluation")
    print("=" * 60)

    model = YOLO(MODEL_PATH)
    class_names = model.names  # {0: 'Bio', 1: 'Non Bio'}
    names_list  = [class_names[i] for i in sorted(class_names)]

    # ── 1. Bounding-box counts for all splits ─────────────────────────────
    print("\n[1] Dataset instance statistics")
    root = Path(DATASET_ROOT)

    splits = {
        "Train" : (root / "train"  / "images", root / "train"  / "labels"),
        "Valid" : (root / "valid"  / "images", root / "valid"  / "labels"),
        "Test"  : (root / "test"   / "images", root / "test"   / "labels"),
    }

    split_stats = {}
    for split_name, (img_dir, lbl_dir) in splits.items():
        if not lbl_dir.exists():
            print(f"\n  WARNING: {lbl_dir} not found — skipping")
            continue
        split_stats[split_name] = print_split_stats(
            split_name, img_dir, lbl_dir, names_list)

    # ── 2. Validation metrics (already known, but re-confirm) ─────────────
    print("\n" + "=" * 60)
    print("[2] Validation set metrics  (split='val')")
    print("=" * 60)
    val_results = model.val(
        data=DATA_YAML,
        split="val",
        imgsz=640,
        conf=0.25,
        iou=0.5,
        verbose=True,
        save_json=False,
    )

    val_map50    = val_results.box.map50
    val_map5095  = val_results.box.map
    val_p        = val_results.box.mp
    val_r        = val_results.box.mr
    val_ap_class = val_results.box.ap50   # per-class AP@0.5

    print(f"\n  Val  mAP@0.5      : {val_map50:.4f}")
    print(f"  Val  mAP@0.5:0.95 : {val_map5095:.4f}")
    print(f"  Val  Precision    : {val_p:.4f}")
    print(f"  Val  Recall       : {val_r:.4f}")
    for i, name in enumerate(names_list):
        ap = float(val_ap_class[i]) if i < len(val_ap_class) else float('nan')
        print(f"  Val  AP@0.5 [{name}]: {ap:.4f}")

    # ── 3. Test set metrics ───────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("[3] Test set metrics  (split='test')  ← USE THESE IN PAPER")
    print("=" * 60)
    test_results = model.val(
        data=DATA_YAML,
        split="test",
        imgsz=640,
        conf=0.25,
        iou=0.5,
        verbose=True,
        save_json=False,
    )

    test_map50   = test_results.box.map50
    test_map5095 = test_results.box.map
    test_p       = test_results.box.mp
    test_r       = test_results.box.mr
    test_ap_class= test_results.box.ap50

    print(f"\n  Test mAP@0.5      : {test_map50:.4f}")
    print(f"  Test mAP@0.5:0.95 : {test_map5095:.4f}")
    print(f"  Test Precision    : {test_p:.4f}")
    print(f"  Test Recall       : {test_r:.4f}")
    for i, name in enumerate(names_list):
        ap = float(test_ap_class[i]) if i < len(test_ap_class) else float('nan')
        print(f"  Test AP@0.5 [{name}]: {ap:.4f}")

    # ── 4. Summary table ──────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("[4] SUMMARY — paste into paper")
    print("=" * 60)

    print("\n--- Dataset statistics ---")
    header = f"  {'Split':<8} {'Images':>7} {'Bio boxes':>10} {'Non-Bio boxes':>14} {'Total boxes':>12}"
    print(header)
    print("  " + "-" * 55)
    totals = {"Images": 0, "Bio": 0, "NonBio": 0, "Total": 0}
    for split_name, stats in split_stats.items():
        imgs = stats["files"]
        bio  = stats["counts"].get("Bio", stats["counts"].get("Bio", 0))
        # handle "Non Bio" key with space
        nonbio = 0
        for k, v in stats["counts"].items():
            if k != "Bio":
                nonbio = v
        tot  = stats["total"]
        print(f"  {split_name:<8} {imgs:>7} {bio:>10} {nonbio:>14} {tot:>12}")
        totals["Images"] += imgs
        totals["Bio"]    += bio
        totals["NonBio"] += nonbio
        totals["Total"]  += tot
    print(f"  {'Total':<8} {totals['Images']:>7} {totals['Bio']:>10} {totals['NonBio']:>14} {totals['Total']:>12}")

    print("\n--- Model performance ---")
    print(f"  {'Metric':<20} {'Validation':>12} {'Test':>12}")
    print("  " + "-" * 46)
    metrics = [
        ("mAP@0.5",      val_map50,   test_map50),
        ("mAP@0.5:0.95", val_map5095, test_map5095),
        ("Precision",    val_p,       test_p),
        ("Recall",       val_r,       test_r),
    ]
    for name, v, t in metrics:
        print(f"  {name:<20} {v:>12.4f} {t:>12.4f}")

    # Per-class AP
    for i, cls_name in enumerate(names_list):
        v_ap = float(val_ap_class[i])  if i < len(val_ap_class)  else float('nan')
        t_ap = float(test_ap_class[i]) if i < len(test_ap_class) else float('nan')
        print(f"  {'AP@0.5 ' + cls_name:<20} {v_ap:>12.4f} {t_ap:>12.4f}")

    print("\nDone. Copy the numbers above into your paper and report.")
    print("=" * 60)


if __name__ == "__main__":
    main()
