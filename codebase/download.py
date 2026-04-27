from roboflow import Roboflow
import os

rf = Roboflow(api_key="YrVcZyd6gmK5afUKesPm")
project = rf.workspace("minaruls-workspace-2ptiz").project("wastemanagement-3iq6q-hl0ll")

# Download augmented dataset — version 2
dataset = project.version(2).download("yolov8")

dataset_path = dataset.location
print(f"\n✅ Dataset ready at: {dataset_path}")

# Verify splits
for split in ['train', 'valid', 'test']:
    path = f"{dataset_path}/{split}/images"
    if os.path.exists(path):
        print(f"{split}: {len(os.listdir(path))} images")
