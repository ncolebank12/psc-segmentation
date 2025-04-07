from ultralytics import YOLO

model = YOLO("yolo11m-seg.pt")

results = model.train(data="data.yaml", epochs=500, imgsz=1280, save_period=50, name="m_1280_augmented", batch=2)