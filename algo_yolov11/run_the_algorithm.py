# %% [markdown]
# initialize yolov11 model.

# %%
from ultralytics import YOLO
import warnings

warnings.filterwarnings("ignore")

# %%
# Load a model
# load a pretrained model (recommended for training)
model = YOLO("yolo11n.pt")


# %%
results = model.train(
    data="/Users/kenton/HOME/coding/python/publish_the_paper/yolov11/dataset_tinkering/med_training_data_pn/dataset.yaml",
    epochs=10,
    device="mps",
    workers=8,
    batch=8,
    imgsz=416,
)

if results is None:
    exit(0)

# %%
model.val()

path = "/Users/kenton/HOME/coding/python/publish_the_paper/create_ml_image_detection_databases/test_data/"


# %%

# Perform inference on test images
for i in range(10):  # Loop through 10 test images
    img = path + f"sig_{i}.png"

    # Make predictions
    results = model(img)

    # Retrieve and print predictions for the image
    print(results[0].boxes.xyxy)  # Bounding boxes (x1, y1, x2, y2, confidence, class)
    print(results[0].boxes.cls)  # Class IDs
    print(results[0].boxes.conf)  # Confidence scores

    # Save or display results
    # results.save()  # Save annotated image to the default output directory
    # results.show()  # Display the image with bounding boxes (optional)

model.export(format="onnx", dynamic=True)