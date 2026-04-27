# train_fresh.py
# Start fresh with stable config for from-scratch training

import sys
import os
sys.path.insert(0, './ultralytics_custom')

import torch
from ultralytics import YOLO
from ultralytics_custom.utils.wiou import WIoUBboxLoss
from ultralytics.utils import loss as loss_module

loss_module.BboxLoss = WIoUBboxLoss
print("[WIoU]  Patch applied")
print(f"[GPU]   {torch.cuda.get_device_name(0)}")

data_yaml = '../wastemanagement-2/data.yaml'
assert os.path.exists(data_yaml)
print(f"[Data]  {data_yaml}")

# Fresh model — random weights
model = YOLO('yolo11s.yaml')
print(f"[Model] yolo11s from scratch — {sum(p.numel() for p in model.model.parameters()):,} params")
print("-" * 60)

results = model.train(
    data          = data_yaml,
    epochs        = 200,
    imgsz         = 640,
    batch         = 32,
    optimizer     = 'AdamW',
    lr0           = 0.001,    # ← lower than before (was 0.01)
    lrf           = 0.0001,   # ← lower final lr
    cos_lr        = True,
    warmup_epochs = 10,       # ← longer warmup (was 5)
    warmup_bias_lr = 0.01,    # ← warmup bias lr
    patience      = 50,       # ← enough patience
    project       = 'runs',
    name          = 'dualstream_fresh',
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

print("\n" + "=" * 60)
print("TRAINING COMPLETE")
print("=" * 60)
m = results.results_dict
map50   = m.get('metrics/mAP50(B)',     0)
map5095 = m.get('metrics/mAP50-95(B)', 0)
prec    = m.get('metrics/precision(B)', 0)
rec     = m.get('metrics/recall(B)',    0)
print(f"mAP@0.5:       {map50:.4f}")
print(f"mAP@0.5:0.95:  {map5095:.4f}")
print(f"Precision:     {prec:.4f}")
print(f"Recall:        {rec:.4f}")
print("=" * 60)
if map50 > 0.944:
    print("NEW BEST — beats 0.944!")
else:
    print(f"Gap from 0.944: {0.944 - map50:.4f}")
print(f"\nModel: runs/dualstream_fresh/weights/best.pt")
