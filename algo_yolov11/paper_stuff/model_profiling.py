from ultralytics import YOLO

# Load a model
model = YOLO("/Users/kenton/HOME/coding/python/publish_the_paper/runs/detect/train50/weights/best.pt")  # load a pretrained model (recommended for training)

num_layers = sum(1 for _ in model.model.modules())
print(f"Number of layers: {num_layers}")