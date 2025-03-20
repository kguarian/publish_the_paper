from ultralytics import YOLO

# Load a model
model = YOLO("yolo11n.pt")  # load a pretrained model (recommended for training)

results = model.train(data="dataset.yaml", epochs=1, device="mps")
if results is None:
   exit(0)
