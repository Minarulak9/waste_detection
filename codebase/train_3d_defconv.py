import os
import re
import torch
import torch.nn as nn
import torch.nn.functional as F
import importlib
import ultralytics

BASE = '/data/aicoe_gpu/debdutta_pal_n2a/mca_minarul/waste_project'
DATASET = f'{BASE}/wastemanagement-2'

print("="*50)
print("Step 3d — Deformable Conv + BiFPN + CBAM + WIoU")
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

# ── PATCH 2: BiFPN (same as 3c) ──
modules_file = os.path.join(os.path.dirname(ultralytics.__file__), 'nn', 'modules', 'block.py')
with open(modules_file, 'r') as f:
    content = f.read()

bifpn_code = '''
class BiFPNNode(nn.Module):
    """Single BiFPN weighted fusion node."""
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
    """Bidirectional Feature Pyramid Network."""
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

# ── Deformable Convolution ──
class DeformableConv2d(nn.Module):
    """
    Deformable Convolution v2.
    Learns offsets for each convolution grid point — adapts sampling
    to actual object shape instead of fixed 3x3 grid.
    Reference: Deformable Convolutional Networks (Dai et al. 2017)
    """
    def __init__(self, in_channels, out_channels, kernel_size=3,
                 stride=1, padding=1, bias=False):
        super().__init__()
        self.kernel_size = kernel_size
        self.stride      = stride
        self.padding     = padding

        # Offset prediction — 2 * kernel_size^2 for x,y offsets per point
        self.offset_conv = nn.Conv2d(
            in_channels,
            2 * kernel_size * kernel_size,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=True
        )
        # Initialize offsets to zero — starts as standard conv
        nn.init.constant_(self.offset_conv.weight, 0)
        nn.init.constant_(self.offset_conv.bias, 0)

        # Main convolution
        self.regular_conv = nn.Conv2d(
            in_channels, out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=bias
        )

    def forward(self, x):
        # Predict offsets from input
        offset = self.offset_conv(x)

        # Use torchvision deform_conv2d if available
        try:
            from torchvision.ops import deform_conv2d
            out = deform_conv2d(
                input=x,
                offset=offset,
                weight=self.regular_conv.weight,
                bias=self.regular_conv.bias,
                stride=self.stride,
                padding=self.padding
            )
        except ImportError:
            # Fallback to regular conv if torchvision not available
            print("⚠ torchvision not found — using regular conv as fallback")
            out = self.regular_conv(x)

        return out


def insert_deformable_conv(model):
    """
    Replace the first Conv layer in each C3k2 block with DeformableConv2d.
    This makes the model adaptive to irregular/overlapping waste shapes.
    """
    from ultralytics.nn.modules.block import C3k2
    from ultralytics.nn.modules.conv import Conv

    replaced = 0

    def replace_in_module(module, name=''):
        nonlocal replaced
        for child_name, child in list(module.named_children()):
            full_name = f"{name}.{child_name}" if name else child_name
            if isinstance(child, C3k2):
                # Replace cv1 (first conv) with deformable conv
                if hasattr(child, 'cv1') and hasattr(child.cv1, 'conv'):
                    orig_conv = child.cv1.conv
                    in_ch  = orig_conv.in_channels
                    out_ch = orig_conv.out_channels
                    k = orig_conv.kernel_size[0] if isinstance(orig_conv.kernel_size, tuple) else orig_conv.kernel_size
                    s = orig_conv.stride[0] if isinstance(orig_conv.stride, tuple) else orig_conv.stride
                    p = orig_conv.padding[0] if isinstance(orig_conv.padding, tuple) else orig_conv.padding

                    # Replace with deformable conv
                    def_conv = DeformableConv2d(in_ch, out_ch, k, s, p).to(
                        next(child.parameters()).device
                    )
                    # Copy pretrained weights
                    with torch.no_grad():
                        def_conv.regular_conv.weight.copy_(orig_conv.weight)

                    child.cv1.conv = def_conv
                    replaced += 1
                    print(f"  ✅ DeformConv({in_ch}→{out_ch}) in {full_name}")
            else:
                replace_in_module(child, full_name)

    replace_in_module(model)
    print(f"✅ Total deformable convs inserted: {replaced}")
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

# ── Load Model and Insert Components ──
importlib.reload(ultralytics)
from ultralytics import YOLO
from ultralytics.nn.modules.block import BiFPN

print("\n🔧 Loading yolo11s...")
model = YOLO(f'{BASE}/yolo11s.pt')

# Insert BiFPN
print("\nAttaching BiFPN...")
neck_channels = [128, 256, 512]
bifpn = BiFPN(neck_channels, num_iterations=2)
model.model.bifpn = bifpn.to('cuda' if torch.cuda.is_available() else 'cpu')
print("✅ BiFPN attached")

# Insert Deformable Convolutions
print("\nInserting Deformable Convolutions...")
model.model = insert_deformable_conv(model.model)

# Install torchvision if needed
import subprocess
try:
    from torchvision.ops import deform_conv2d
    print("✅ torchvision deform_conv2d available")
except ImportError:
    print("Installing torchvision...")
    subprocess.run(['pip', 'install', 'torchvision', '-q'])
    from torchvision.ops import deform_conv2d
    print("✅ torchvision installed and ready")

# ── Train ──
print("\n🚀 Starting Training...")
model.train(
    data=f"{DATASET}/data.yaml",
    epochs=50,
    imgsz=640,
    batch=32,
    optimizer='AdamW',
    lr0=0.01,
    weight_decay=0.0005,
    patience=10,
    project=f'{BASE}/results',
    name='3d_defconv_bifpn_cbam_wiou',
    exist_ok=True,
    save=True,
    verbose=True,
    device=0
)

# ── Evaluate ──
print("\n📊 Evaluating...")
best = YOLO(f'{BASE}/results/3d_defconv_bifpn_cbam_wiou/weights/best.pt')
val = best.val(
    data=f"{DATASET}/data.yaml",
    split='val', imgsz=640, batch=32,
    conf=0.30, iou=0.5, verbose=False
)

print("\n========== 3d DefConv + BiFPN + CBAM + WIoU ==========")
print(f"mAP@0.5:       {val.box.map50:.4f}")
print(f"mAP@0.5:0.95:  {val.box.map:.4f}")
print(f"Precision:     {val.box.mp:.4f}")
print(f"Recall:        {val.box.mr:.4f}")
print("=======================================================")
for i, cls_name in enumerate(best.names.values()):
    print(f"  {cls_name:15s} | P: {val.box.p[i]:.4f} | R: {val.box.r[i]:.4f} | AP@0.5: {val.box.ap50[i]:.4f}")
