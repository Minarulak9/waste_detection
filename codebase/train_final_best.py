import os
import re
import torch
import torch.nn as nn
import importlib
import ultralytics

BASE = '/data/aicoe_gpu/debdutta_pal_n2a/mca_minarul/waste_project'
DATASET = f'{BASE}/wastemanagement-2'

print("="*50)
print("FINAL BEST MODEL — WIoU + BiFPN — 100 Epochs")
print("="*50)

# ── PATCH 1: WIoU ──
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
elif 'wiou_loss(iou)' not in content:
    content = re.sub(r'loss_iou\s*=\s*\(1\.0?\s*-\s*iou\)\.mean\(\)', new_line, content)
print("✅ WIoU patched")
with open(loss_file, 'w') as f:
    f.write(content)

# ── PATCH 2: BiFPN ──
modules_file = os.path.join(os.path.dirname(ultralytics.__file__), 'nn', 'modules', 'block.py')
with open(modules_file, 'r') as f:
    content = f.read()

bifpn_code = '''
class BiFPNNode(nn.Module):
    def __init__(self, channels, epsilon=1e-4):
        super().__init__()
        self.epsilon = epsilon
        self.w = nn.Parameter(torch.ones(2, dtype=torch.float32))
        self.relu = nn.ReLU()
        self.conv = nn.Conv2d(channels, channels, 1, bias=False)
        self.bn   = nn.BatchNorm2d(channels)
        self.act  = nn.SiLU()
    def forward(self, x1, x2):
        w = self.relu(self.w)
        w = w / (w.sum() + self.epsilon)
        if x1.shape != x2.shape:
            x2 = nn.functional.interpolate(x2, size=x1.shape[-2:], mode='nearest')
        return self.act(self.bn(self.conv(w[0] * x1 + w[1] * x2)))

class BiFPN(nn.Module):
    def __init__(self, channels_list, num_iterations=2):
        super().__init__()
        self.num_iterations = num_iterations
        num_levels = len(channels_list)
        self.td_nodes = nn.ModuleList()
        self.bu_nodes = nn.ModuleList()
        for _ in range(num_iterations):
            self.td_nodes.append(nn.ModuleList([BiFPNNode(channels_list[i]) for i in range(num_levels-1)]))
            self.bu_nodes.append(nn.ModuleList([BiFPNNode(channels_list[i]) for i in range(1, num_levels)]))
    def forward(self, features):
        for i in range(self.num_iterations):
            td = list(features)
            for j in range(len(features)-2, -1, -1):
                td[j] = self.td_nodes[i][j](td[j], td[j+1])
            out = list(td)
            for j in range(1, len(features)):
                out[j] = self.bu_nodes[i][j-1](out[j], out[j-1])
            features = out
        return features

'''
if 'class BiFPN' not in content:
    content = content.replace('class DFL(nn.Module):', bifpn_code + '\nclass DFL(nn.Module):')
    print("✅ BiFPN injected")
else:
    print("✅ BiFPN already present")
with open(modules_file, 'w') as f:
    f.write(content)

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

# ── Load Model ──
importlib.reload(ultralytics)
from ultralytics import YOLO
from ultralytics.nn.modules.block import BiFPN

print("\n🔧 Loading yolo11s...")
model = YOLO(f'{BASE}/yolo11s.pt')

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Attach BiFPN
neck_channels = [128, 256, 512]
bifpn = BiFPN(neck_channels, num_iterations=2).to(device)
model.model.bifpn = bifpn
print("✅ BiFPN attached")

# ── Train — 100 epochs with cosine LR decay ──
print("\n🚀 Starting Final Training — 100 epochs...")
model.train(
    data=f"{DATASET}/data.yaml",
    epochs=100,
    imgsz=640,
    batch=32,
    optimizer='AdamW',
    lr0=0.01,
    lrf=0.001,           # Cosine decay to 0.001
    cos_lr=True,         # Enable cosine LR schedule
    weight_decay=0.0005,
    warmup_epochs=3,
    patience=15,         # More patience for longer run
    project=f'{BASE}/results',
    name='final_wiou_bifpn_100ep',
    exist_ok=True,
    save=True,
    save_period=10,      # Save checkpoint every 10 epochs
    verbose=True,
    device=0
)

# ── Evaluate ──
print("\n📊 Evaluating final model...")
best = YOLO(f'{BASE}/results/final_wiou_bifpn_100ep/weights/best.pt')
val = best.val(
    data=f"{DATASET}/data.yaml",
    split='val', imgsz=640, batch=32,
    conf=0.30, iou=0.5, verbose=False
)

print("\n========== FINAL MODEL — WIoU + BiFPN 100ep ==========")
print(f"mAP@0.5:       {val.box.map50:.4f}")
print(f"mAP@0.5:0.95:  {val.box.map:.4f}")
print(f"Precision:     {val.box.mp:.4f}")
print(f"Recall:        {val.box.mr:.4f}")
print("=======================================================")
for i, cls_name in enumerate(best.names.values()):
    print(f"  {cls_name:15s} | P: {val.box.p[i]:.4f} | R: {val.box.r[i]:.4f} | AP@0.5: {val.box.ap50[i]:.4f}")
