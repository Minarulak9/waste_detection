import os
import re
import torch
import torch.nn as nn
import importlib
import ultralytics

BASE = '/data/aicoe_gpu/debdutta_pal_n2a/mca_minarul/waste_project'
DATASET = f'{BASE}/wastemanagement-2'

print("="*50)
print("Step 3b — CBAM + WIoU Training")
print("="*50)

# ── PATCH 1: WIoU Loss ──
loss_file = os.path.join(os.path.dirname(ultralytics.__file__), 'utils', 'loss.py')
with open(loss_file, 'r') as f:
    content = f.read()

wiou_func = '''
def wiou_loss(iou, eps=1e-7):
    """Wise-IoU dynamic focusing loss."""
    iou_mean = iou.mean()
    iou_std  = iou.std() + eps
    beta = ((iou - iou_mean) ** 2 / (2 * iou_std ** 2)).exp()
    return (beta * (1 - iou)).mean()

'''

if 'def wiou_loss' not in content:
    content = content.replace('class BboxLoss(nn.Module):', wiou_func + 'class BboxLoss(nn.Module):')
    print("✅ WIoU injected")
else:
    print("✅ WIoU already present")

old_line = 'loss_iou = (1.0 - iou).mean()'
new_line = 'loss_iou = wiou_loss(iou)  # WIoU'
if old_line in content:
    content = content.replace(old_line, new_line)
    print("✅ WIoU loss patched")
elif 'wiou_loss(iou)' in content:
    print("✅ WIoU already patched")
else:
    content = re.sub(r'loss_iou\s*=\s*\(1\.0?\s*-\s*iou\)\.mean\(\)', new_line, content)
    print("✅ WIoU patched via regex")

with open(loss_file, 'w') as f:
    f.write(content)

# ── CBAM Modules ──
class ChannelAttention(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        reduced = max(1, channels // reduction)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(channels, reduced, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(reduced, channels, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        return x * self.sigmoid(self.fc(self.avg_pool(x)) + self.fc(self.max_pool(x)))

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size//2, bias=False)
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        avg = x.mean(dim=1, keepdim=True)
        max_, _ = x.max(dim=1, keepdim=True)
        return x * self.sigmoid(self.conv(torch.cat([avg, max_], dim=1)))

class CBAM(nn.Module):
    def __init__(self, channels, reduction=16, kernel_size=7):
        super().__init__()
        self.ca = ChannelAttention(channels, reduction)
        self.sa = SpatialAttention(kernel_size)
    def forward(self, x):
        return self.sa(self.ca(x))

def insert_cbam_into_model(model):
    """
    Insert CBAM after each C3k2 block in the backbone.
    Detects channel size automatically — no hardcoding.
    """
    from ultralytics.nn.modules.block import C3k2
    
    def add_cbam_after_c3k2(module, name=''):
        for child_name, child in list(module.named_children()):
            full_name = f"{name}.{child_name}" if name else child_name
            if isinstance(child, C3k2):
                # Get output channels from the C3k2 block
                out_channels = None
                for p in child.parameters():
                    out_channels = p.shape[0]
                    break
                # Get actual output by doing a test forward
                # Instead use cv2 attribute which holds output conv
                if hasattr(child, 'cv2'):
                    out_channels = child.cv2.conv.weight.shape[0]
                elif hasattr(child, 'cv1'):
                    out_channels = child.cv1.conv.weight.shape[0]
                
                if out_channels:
                    cbam = CBAM(out_channels).to(next(child.parameters()).device)
                    # Wrap C3k2 + CBAM in Sequential
                    wrapped = nn.Sequential(child, cbam)
                    setattr(module, child_name, wrapped)
                    print(f"  ✅ CBAM({out_channels}) inserted after {full_name}")
            else:
                add_cbam_after_c3k2(child, full_name)
    
    add_cbam_after_c3k2(model)
    return model

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

# ── Load Model and Insert CBAM ──
importlib.reload(ultralytics)
from ultralytics import YOLO

print("\n🔧 Loading yolo11s and inserting CBAM...")
model = YOLO('yolo11s.pt')

print("Inserting CBAM into backbone C3k2 blocks:")
model.model = insert_cbam_into_model(model.model)
print("✅ CBAM insertion complete")

# ── Train ──
print("\n🚀 Starting Training...")
model.train(
    data=f"{DATASET}/data.yaml",
    epochs=100,
    imgsz=640,
    batch=32,
    optimizer='AdamW',
    lr0=0.01,
    weight_decay=0.0005,
    patience=10,
    project=f'{BASE}/results',
    name='3b_cbam_wiou',
    exist_ok=True,
    save=True,
    verbose=True,
    device=0
)

# ── Evaluate ──
print("\n📊 Evaluating...")
best = YOLO(f'{BASE}/results/3b_cbam_wiou/weights/best.pt')
val = best.val(
    data=f"{DATASET}/data.yaml",
    split='val', imgsz=640, batch=32,
    conf=0.30, iou=0.5, verbose=False
)

print("\n========== 3b CBAM + WIoU RESULTS ==========")
print(f"mAP@0.5:       {val.box.map50:.4f}")
print(f"mAP@0.5:0.95:  {val.box.map:.4f}")
print(f"Precision:     {val.box.mp:.4f}")
print(f"Recall:        {val.box.mr:.4f}")
print("=============================================")
for i, cls_name in enumerate(best.names.values()):
    print(f"  {cls_name:15s} | P: {val.box.p[i]:.4f} | R: {val.box.r[i]:.4f} | AP@0.5: {val.box.ap50[i]:.4f}")
