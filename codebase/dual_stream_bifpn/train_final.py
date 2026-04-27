# train_final.py
# FINAL TRAINING RUN — One shot, best possible config

import sys
import os
sys.path.insert(0, './ultralytics_custom')

import torch
from ultralytics import YOLO
from ultralytics_custom.utils.wiou import WIoUBboxLoss
from ultralytics.utils import loss as loss_module

loss_module.BboxLoss = WIoUBboxLoss
print("[WIoU]   Patch applied")
print(f"[GPU]    {torch.cuda.get_device_name(0)}")

data_yaml = '../wastemanagement-2/data.yaml'
assert os.path.exists(data_yaml)
print(f"[Data]   {data_yaml}")

model = YOLO('yolo11s.pt')
total = sum(p.numel() for p in model.model.parameters())
print(f"[Model]  yolo11s pretrained — {total:,} parameters")
print("=" * 60)
print("FINAL TRAINING RUN")
print("yolo11s.pt + WIoU + augmentation + mixup + 150ep cosine")
print("=" * 60)

results = model.train(
    data            = data_yaml,
    epochs          = 150,
    imgsz           = 640,
    batch           = 16,
    optimizer       = 'AdamW',
    lr0             = 0.01,
    lrf             = 0.001,
    cos_lr          = True,
    warmup_epochs   = 5,
    warmup_momentum = 0.8,
    warmup_bias_lr  = 0.1,
    patience        = 30,
    weight_decay    = 0.0005,
    momentum        = 0.937,
    # Proven augmentation
    hsv_h           = 0.015,
    hsv_s           = 0.7,
    hsv_v           = 0.4,
    degrees         = 15.0,
    translate       = 0.1,
    scale           = 0.5,
    fliplr          = 0.5,
    mosaic          = 1.0,
    # New augmentation — not used before
    mixup           = 0.1,
    copy_paste      = 0.1,
    # Save config
    project         = 'runs',
    name            = 'final_best_v2',
    device          = 0,
    exist_ok        = True,
    verbose         = True,
    save_period     = 10,
)

print("\n" + "=" * 60)
print("FINAL TRAINING COMPLETE")
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
print(f"\nComparison:")
print(f"  Verified baseline:  0.9371  mAP@0.5")
print(f"  This run:           {map50:.4f}  mAP@0.5")
print(f"  Difference:         {map50 - 0.9371:+.4f}")
print("=" * 60)

if map50 > 0.9371:
    print("NEW BEST — beats verified baseline!")
    # Auto copy to BEST_MODELS
    import shutil
    src = f'runs/detect/final_best/weights/best.pt'
    dst = f'/data/aicoe_gpu/debdutta_pal_n2a/mca_minarul/waste_project/BEST_MODELS/best_{map50:.4f}_final.pt'
    if os.path.exists(src):
        shutil.copy(src, dst)
        print(f"  Auto-saved to: {dst}")
else:
    print("Use verified baseline: BEST_MODELS/best_0937_verified.pt")

print(f"\nThis run saved at:")
print(f"  runs/detect/final_best/weights/best.pt")
