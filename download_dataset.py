from roboflow import Roboflow

rf = Roboflow(api_key="YrVcZyd6gmK5afUKesPm")
project = rf.workspace("minaruls-workspace-2ptiz").project("wastemanagement-3iq6q-hl0ll")
version = project.version(2)

dataset = version.download("yolov8")

print("Dataset downloaded at:", dataset.location)