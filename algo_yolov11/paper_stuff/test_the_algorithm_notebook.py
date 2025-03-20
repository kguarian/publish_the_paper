# %%
import os
import random
from math import ceil, sqrt
from ultralytics import YOLO
from PIL import Image, ImageDraw, ImageFont



# %%
# Path to the test directory
test_dir = "/Users/kenton/HOME/coding/python/publish_the_paper/yolov11/dataset_tinkering/training_data_pn/images/test"
output_collage = "collage_with_boxes.png"

# Load the model
model = YOLO(
    "/Users/kenton/HOME/coding/python/publish_the_paper/runs/detect/train50/weights/best.pt"
)

# Get all image files in the directory
all_images = [
    os.path.join(test_dir, f)
    for f in os.listdir(test_dir)
    if f.lower().endswith((".png", ".jpg", ".jpeg"))
]

# Randomly select 100 images (or fewer if there are not enough images)
random_images = random.sample(all_images, min(len(all_images), 100))

# Optional: Load a font for better text rendering
try:
    font = ImageFont.truetype("arial.ttf", size=16)  # Use a font installed on your system
except IOError:
    font = ImageFont.load_default()


# %%
# Annotate each selected image
annotated_images = []
for image_path in random_images:
    # Predict results for the image
    results = model.predict(source=image_path, conf=0.8)

    # Load the image using PIL
    image = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(image)

    # Process model predictions
    boxes = []
    for r in results:
        for i in range(len(r.boxes.data)):
            boxes.append(r.boxes.data[i].tolist())

    box = boxes[0]
    x1, y1, x2, y2, conf, cls = box
    boxes_with_burst = [box for box in boxes if box[5] == 1]
    sorted = sorted(boxes_with_burst, key=lambda x: x[0])

    # Resize the image to a fixed size for the collage
    resized_image = image.resize((200, 200))  # Adjust size as needed
    annotated_images.append(resized_image)

# Determine collage dimensions (square layout)
collage_size = ceil(sqrt(len(annotated_images)))  # Closest square size
collage_width = collage_size * 200
collage_height = collage_size * 200

# Create the blank collage canvas
collage = Image.new("RGB", (collage_width, collage_height), color="white")

# Paste each image into the collage
for i, annotated_image in enumerate(annotated_images):
    row = i // collage_size
    col = i % collage_size
    x_offset = col * 200
    y_offset = row * 200
    collage.paste(annotated_image, (x_offset, y_offset))

# Save the collage
collage.save(output_collage)
print(f"Collage saved to {output_collage}")
collage.show()

# %%
print("hi")


