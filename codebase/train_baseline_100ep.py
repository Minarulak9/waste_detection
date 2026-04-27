import os
import torch
import importlib
import ultralytics

BASE = '/data/aicoe_gpu/debdutta_pal_n2a/mca_minarul/waste_project'
DATASET = f'{BASE}/wastemanagement-2'

print("="*50)
print("yolo11s BASELINE — 100 Epochs — No Architecture")
print("="*50)

# ── Fix Labels ──
def fix_labels(label_dir):
    converted, dupes = 0, 0
    for f in os.listdir(label_dir):
        fp = f"{label_dir}/{f}"
        with open(fp,'r') as file:
            lines = file.readlines()
        new_lines = []
        for line in lines:
            v = line.strip().split()
            if len(v) > 5:
                cls = v[0]
                coords = list(map(float, v[1:]))
                xs, ys = coords[0::2], coords[1::2]
                cx = (min(xs)+max(xs))/2
                cy = (min(ys)+max(ys))/2
                w  = max(xs)-min(xs)
                h  = max(ys)-min(ys)
                new_lines.append(f"{cls} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}\n")
                converted += 1
            elif len(v) == 5:
                new_lines.append(line)
        unique = list(dict.fromkeys(new_lines))
        dupes += len(new_lines) - len(unique)
        with open(fp,'w') as file:
            file.writelines(unique)
    print(f"  Converted: {converted} | Dupes: {dupes}")

for split in ['train','valid','test']:
    path = f"{DATASET}/{split}/labels"
    if os.path.exists(path):
        print(f"Fixing {split}...")
        fix_labels(path)
print("✅ Labels clean")

# ── Load Fresh yolo11s — No patches ──
importlib.reload(ultralytics)
from ultralytics import YOLO

print("\n🔧 Loading fresh yolo11s — no modifications...")
model = YOLO(f'{BASE}/yolo11s.pt')

# ── Train — identical config to final model ──
print("\n🚀 Starting Training...")
model.train(
    data=f"{DATASET}/data.yaml",
    epochs=100,
    imgsz=640,
    batch=32,
    optimizer='AdamW',
    lr0=0.01,
    lrf=0.001,
    cos_lr=True,
    weight_decay=0.0005,
    warmup_epochs=3,
    patience=15,
    project=f'{BASE}/results',
    name='baseline_yolo11s_100ep_noWIoU',
    exist_ok=True,
    save=True,
    save_period=10,
    verbose=True,
    device=0
)

# ── Evaluate ──
print("\n📊 Evaluating...")
best = YOLO(f'{BASE}/results/baseline_yolo11s_100ep_noWIoU/weights/best.pt')
val = best.val(
    data=f"{DATASET}/data.yaml",
    split='val', imgsz=640, batch=32,
    conf=0.30, iou=0.5, verbose=False
)

print("\n========== yolo11s BASELINE 100ep ==========")
print(f"mAP@0.5:       {val.box.map50:.4f}")
print(f"mAP@0.5:0.95:  {val.box.map:.4f}")
print(f"Precision:     {val.box.mp:.4f}")
print(f"Recall:        {val.box.mr:.4f}")
print("=============================================")
for i, cls_name in enumerate(best.names.values()):
    print(f"  {cls_name:15s} | P: {val.box.p[i]:.4f} | R: {val.box.r[i]:.4f} | AP@0.5: {val.box.ap50[i]:.4f}")

print("\n--- COMPARISON ---")
print(f"yolo11s 100ep baseline  vs  WIoU+BiFPN 100ep")
print(f"This result             vs  mAP@0.5: 0.9439")
print(f"Gap = architecture contribution")
