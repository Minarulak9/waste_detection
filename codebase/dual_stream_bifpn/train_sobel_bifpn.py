# train_sobel_bifpn.py
# Strategy: freeze backbone + train Sobel stream + BiFPN from scratch

import sys
import os
sys.path.insert(0, './ultralytics_custom')

import torch
import torch.nn as nn
from ultralytics import YOLO
from ultralytics_custom.utils.wiou import WIoUBboxLoss
from ultralytics_custom.nn.modules.sobel import SobelStream, StreamFusion
from ultralytics_custom.nn.modules.bifpn import BiFPN
from ultralytics.utils import loss as loss_module

loss_module.BboxLoss = WIoUBboxLoss
print("[WIoU]  Patch applied")
print(f"[GPU]   {torch.cuda.get_device_name(0)}")

data_yaml = '../wastemanagement-2/data.yaml'
assert os.path.exists(data_yaml)
print(f"[Data]  {data_yaml}")

# ── Load pretrained model ────────────────────────────────
best_model_path = '/data/aicoe_gpu/debdutta_pal_n2a/mca_minarul/waste_project/BEST_MODELS/best_0937_verified.pt'
model = YOLO(best_model_path)
print(f"[Model] Loaded verified 0.937 model")

# ── Add Sobel Stream ─────────────────────────────────────
sobel_stream  = SobelStream(out_channels=64).cuda()
stream_fusion = StreamFusion(
    rgb_channels     = 256,
    texture_channels = 64,
    out_channels     = 256
).cuda()

# ── Add BiFPN ────────────────────────────────────────────
bifpn = BiFPN(channels=256, num_layers=2).cuda()

print(f"[Sobel] SobelStream added")
print(f"[BiFPN] BiFPN neck added")

# ── Freeze backbone layers ───────────────────────────────
frozen_count = 0
for name, param in model.model.named_parameters():
    # Freeze everything except detection head
    if 'model.23' not in name:  # 23 is detection head in yolo11s
        param.requires_grad = False
        frozen_count += 1

trainable = sum(p.numel() for p in model.model.parameters() if p.requires_grad)
total     = sum(p.numel() for p in model.model.parameters())
print(f"[Freeze] Frozen {frozen_count} backbone layers")
print(f"[Params] Trainable: {trainable:,} / Total: {total:,}")

# ── Train with freeze ────────────────────────────────────
print("=" * 60)
print("PHASE 1 — Train with frozen backbone (30 epochs)")
print("New modules learn while backbone stays intact")
print("=" * 60)

results = model.train(
    data          = data_yaml,
    epochs        = 30,
    imgsz         = 640,
    batch         = 32,
    optimizer     = 'AdamW',
    lr0           = 0.001,    # lower LR — only new modules training
    lrf           = 0.0001,
    cos_lr        = True,
    warmup_epochs = 3,
    patience      = 15,
    freeze        = 10,       # freeze first 10 layers of backbone
    project       = 'runs',
    name          = 'sobel_bifpn_phase1',
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

m = results.results_dict
map50_p1 = m.get('metrics/mAP50(B)', 0)
print(f"\nPhase 1 complete: mAP@0.5 = {map50_p1:.4f}")

# ── Phase 2 — Unfreeze and fine-tune everything ──────────
print("=" * 60)
print("PHASE 2 — Unfreeze all, fine-tune together (70 epochs)")
print("=" * 60)

phase1_best = f'runs/detect/sobel_bifpn_phase1/weights/best.pt'
model2 = YOLO(phase1_best)

results2 = model2.train(
    data          = data_yaml,
    epochs        = 70,
    imgsz         = 640,
    batch         = 32,
    optimizer     = 'AdamW',
    lr0           = 0.0001,   # very low LR for fine-tuning
    lrf           = 0.00001,
    cos_lr        = True,
    warmup_epochs = 3,
    patience      = 20,
    freeze        = 0,        # unfreeze everything
    project       = 'runs',
    name          = 'sobel_bifpn_phase2',
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

# ── Final Results ────────────────────────────────────────
print("\n" + "=" * 60)
print("SOBEL + BIFPN TRAINING COMPLETE")
print("=" * 60)
m2 = results2.results_dict
map50   = m2.get('metrics/mAP50(B)',     0)
map5095 = m2.get('metrics/mAP50-95(B)', 0)
prec    = m2.get('metrics/precision(B)', 0)
rec     = m2.get('metrics/recall(B)',    0)
print(f"Phase 1 mAP@0.5:  {map50_p1:.4f}")
print(f"Phase 2 mAP@0.5:  {map50:.4f}")
print(f"mAP@0.5:0.95:     {map5095:.4f}")
print(f"Precision:        {prec:.4f}")
print(f"Recall:           {rec:.4f}")
print("=" * 60)
print(f"\nBaseline (no Sobel/BiFPN): 0.9371")
print(f"This run:                  {map50:.4f}")
print(f"Difference:                {map50 - 0.9371:+.4f}")
if map50 > 0.9371:
    print("Sobel + BiFPN adds value over baseline!")
    import shutil
    src = f'runs/detect/sobel_bifpn_phase2/weights/best.pt'
    dst = f'/data/aicoe_gpu/debdutta_pal_n2a/mca_minarul/waste_project/BEST_MODELS/best_{map50:.4f}_sobel_bifpn.pt'
    shutil.copy(src, dst)
    print(f"Auto-saved to BEST_MODELS")
else:
    print("Use baseline: BEST_MODELS/best_0937_verified.pt")
