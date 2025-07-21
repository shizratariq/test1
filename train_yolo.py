from ultralytics import YOLO

# Load a pre-trained segmentation model (nano/small/medium)
model = YOLO("yolov8n-seg.pt")  # or yolov8s-seg.pt, yolov8m-seg.pt, etc.

# Start training
model.train(
    data="dataset/dataset.yaml",
    epochs=100,
    imgsz=640,
    batch=8,
    name="probe_segmentation",
    task="segment"
)
