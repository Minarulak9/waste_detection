# train_dualstream.py
# DualStream BiFPN-YOLO — Train from Scratch

import sys
import os
sys.path.insert(0, './ultralytics_custom')

import torch
from ultralytics import YOLO

print(f"PyTorch:  {torch.__version__}")
# train_dualstream.py
# DualStream BiFPN-YOLO — Train from Scratch

import sys
import os
sys.path.insert(0, './ultralytics_custom')

import torch
from ultralytics import YOLO
from ultralytics_custom.utils.wiou import WIoUBboxLoss
from ultralytics.utils import loss as loss_module

# ── Apply WIoU patch ─────────────────────────────────────
loss_module.BboxLoss = WIoUBboxLoss
print("[WIoU]  Patch applied")

# ── Verify GPU ───────────────────────────────────────────
print(f"[Info]  PyTorch:  {torch.__version__}")
print(f"[Info]  CUDA:     {torch.cuda.is_available()}")
print(f"[Info]  GPU:      {torch.cuda.get_device_name(0)}")

# ── Verify data.yaml ─────────────────────────────────────
data_yaml = '../wastemanagement-2/data.yaml'
assert os.path.exists(data_yaml), f"data.yaml not found: {data_yaml}"
print(f"[Data]  Found: {data_yaml}")

# ── Load model (from scratch — no pretrained weights) ────
model = YOLO('yolo11s.yaml')
total_params = sum(p.numel() for p in model.model.parameters())
print(f"[Model] Loaded yolo11s from scratch")
print(f"[Model] Parameters: {total_params:,}")
print("-" * 60)
print("Starting full training — 200 epochs")
print("-" * 60)

# ── Train ────────────────────────────────────────────────
results = model.train(
    data          = data_yaml,
    epochs        = 200,
    imgsz         = 640,
    batch         = 32,
    optimizer     = 'AdamW',
    lr0           = 0.01,
    lrf           = 0.001,
    cos_lr        = True,
    warmup_epochs = 5,
    patience      = 20,
    project       = 'runs',
    name          = 'dualstream_bifpn_scratch',
    device        = 0,
    exist_ok      = True,
    verbose       = True,
    # Augmentation
    hsv_h         = 0.015,
    hsv_s         = 0.7,
    hsv_v         = 0.4,
    degrees       = 15.0,
    fliplr        = 0.5,
    mosaic        = 1.0,
)

# ── Print Results ─────────────────────────────────────────
print("\n" + "=" * 60)
print("TRAINING COMPLETE")
print("=" * 60)
m = results.results_dict
map50    = m.get('metrics/mAP50(B)',    0)
map5095  = m.get('metrics/mAP50-95(B)', 0)
prec     = m.get('metrics/precision(B)', 0)
rec      = m.get('metrics/recall(B)',    0)
print(f"mAP@0.5:       {map50:.4f}")
print(f"mAP@0.5:0.95:  {map5095:.4f}")
print(f"Precision:     {prec:.4f}")
print(f"Recall:        {rec:.4f}")
print("=" * 60)
print(f"\nPrevious best: 0.944 mAP@0.5")
print(f"New result:    {map50:.4f} mAP@0.5")
if map50 > 0.944:
    print("NEW BEST — architecture integration successful!")
else:
    print("Keep training or check architecture integration.")
print(f"\nModel saved at:")
print(f"  runs/dualstream_bifpn_scratch/weights/best.pt")
