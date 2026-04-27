import ultralytics
import os

# Fix 1 — Add CBAM to tasks.py globals
tasks_file = os.path.join(os.path.dirname(ultralytics.__file__), 'nn', 'tasks.py')

with open(tasks_file, 'r') as f:
    content = f.read()

# Add CBAM import at the top of tasks.py
cbam_import = "from ultralytics.nn.modules.block import CBAM, ChannelAttention, SpatialAttention\n"

if 'from ultralytics.nn.modules.block import CBAM' not in content:
    # Add after existing block imports
    content = content.replace(
        'from ultralytics.nn.modules import (',
        cbam_import + 'from ultralytics.nn.modules import ('
    )
    print("✅ CBAM imported in tasks.py")
else:
    print("✅ CBAM already imported in tasks.py")

with open(tasks_file, 'w') as f:
    f.write(content)

# Fix 2 — Verify CBAM is in block.py
modules_file = os.path.join(os.path.dirname(ultralytics.__file__), 'nn', 'modules', 'block.py')
with open(modules_file, 'r') as f:
    content = f.read()

if 'class CBAM' in content:
    print("✅ CBAM class confirmed in block.py")
else:
    print("❌ CBAM missing from block.py — re-run train script patches")

# Fix 3 — Verify tasks.py uses globals() correctly
# CBAM needs to be in the globals dict that parse_model uses
tasks_file_content = open(tasks_file).read()
if 'CBAM' in tasks_file_content:
    print("✅ CBAM visible in tasks.py")
else:
    print("❌ CBAM still not in tasks.py")

print("\n✅ Fix complete — now run train_3b_cbam.py")
