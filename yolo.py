from ultralytics import YOLO

model = YOLO("models/best.pt")

model.predict("input_videos/short.mp4", save=True)
