import ultralytics
import os

loss_file = os.path.join(os.path.dirname(ultralytics.__file__), 'utils', 'loss.py')

with open(loss_file, 'r') as f:
    content = f.read()

if 'wiou_loss' in content:
    print("✅ WIoU IS in loss.py")
    # Show the patched line
    for i, line in enumerate(content.split('\n')):
        if 'wiou_loss' in line:
            print(f"  Line {i}: {line.strip()}")
else:
    print("❌ WIoU NOT in loss.py — patch not surviving")
