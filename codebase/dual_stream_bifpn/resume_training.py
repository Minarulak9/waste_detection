# resume_training.py

# Resume from last checkpoint

import sys

import os

sys.path.insert(0, './ultralytics_custom')

import torch

from ultralytics import YOLO

from ultralytics_custom.utils.wiou import WIoUBboxLoss

from ultralytics.utils import loss as loss_module

# Apply WIoU patch

loss_module.BboxLoss = WIoUBboxLoss

print("[WIoU]  Patch applied")

last_pt = 'runs/detect/runs/dualstream_bifpn_scratch/weights/last.pt'

assert os.path.exists(last_pt), f"Not found: {last_pt}"

print(f"[Resume] Loading from: {last_pt}")

model = YOLO(last_pt)

results = model.train(

    resume        = True,

    data          = '../wastemanagement-2/data.yaml',

    epochs        = 200,

    imgsz         = 640,

    batch         = 32,

    optimizer     = 'AdamW',

    lr0           = 0.01,

    lrf           = 0.001,

    cos_lr        = True,

    warmup_epochs = 5,

    patience      = 50,      # increased from 20

    project       = 'runs',

    name          = 'dualstream_bifpn_scratch',

    device        = 0,

    exist_ok      = True,

    verbose       = True,

    hsv_h         = 0.015,

    hsv_s         = 0.7,

    hsv_v         = 0.4,

    degrees       = 15.0,

    fliplr        = 0.5,

    mosaic        = 1.0,

)

print("\n" + "=" * 60)

print("RESUMED TRAINING COMPLETE")

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

print(f"Previous best: 0.944 mAP@0.5")

print(f"New result:    {map50:.4f} mAP@0.5")

if map50 > 0.944:

    print("NEW BEST!")
