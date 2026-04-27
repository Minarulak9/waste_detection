import ultralytics
import os
import re

loss_file = os.path.join(os.path.dirname(ultralytics.__file__), 'utils', 'loss.py')

with open(loss_file, 'r') as f:
    content = f.read()

# Remove wiou_loss function
content = re.sub(
    r'\ndef wiou_loss\(iou.*?return \(beta \* \(1 - iou\)\)\.mean\(\)\n\n',
    '\n',
    content,
    flags=re.DOTALL
)

# Restore original loss line
content = content.replace(
    'loss_iou = wiou_loss(iou)  # WIoU',
    'loss_iou = (1.0 - iou).mean()'
)

with open(loss_file, 'w') as f:
    f.write(content)

# Verify
if 'wiou_loss' not in open(loss_file).read():
    print("✅ WIoU removed — loss.py restored to default")
else:
    print("⚠️ WIoU still present — check manually")
