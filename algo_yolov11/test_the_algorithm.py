# /Users/kenton/HOME/coding/python/publish_the_paper/runs/detect/train50/weights/last.pt
# /Users/kenton/HOME/coding/python/publish_the_paper/runs/detect/train50/weights/best.pt

import os
import random
from math import ceil, sqrt
from ultralytics import YOLO
from PIL import Image, ImageDraw, ImageFont

# Path to the test directory
test_dir = "/Users/kenton/HOME/coding/python/publish_the_paper/yolov11/dataset_tinkering/training_data_pn/images/test"
output_collage = "collage_with_boxes_and_borders_2.png"

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

# Annotate each selected image
annotated_images = []
for image_path in random_images:
    # Predict results for the image
    results = model.predict(source=image_path, conf=0.7)

    # Load the image using PIL
    image = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(image)

    # Process model predictions
    for r in results:
        for box in r.boxes.data:
            # Extract bounding box and class information
            x1, y1, x2, y2, confidence, class_id = box.tolist()
            class_name = model.names[int(class_id)]  # Get class name using model's class names
            if class_id==1:
                print(f"box bounds: {x1}, {x2}")

            # Draw the bounding box
            draw.rectangle([x1, y1, x2, y2], outline="red", width=3)

            # Create a label
            label = f"{class_name} ({confidence:.2f})"

            # Draw label inside the bounding box
            text_bbox = draw.textbbox((x1, y1), label, font=font)
            text_width = text_bbox[2] - text_bbox[0]
            text_height = text_bbox[3] - text_bbox[1]

            # Position text inside the bounding box, adjusted to fit
            label_x = max(x1, 0) + 2
            label_y = max(y1, 0) + 2

            # Draw label background and text
            draw.rectangle(
                [label_x, label_y, label_x + text_width, label_y + text_height],
                fill="red",
            )
            draw.text((label_x, label_y), label, fill="white", font=font)

    # Add a black border around the image
    border_size = 5
    bordered_image = Image.new(
        "RGB",
        (image.width + 2 * border_size, image.height + 2 * border_size),
        color="black",
    )
    bordered_image.paste(image, (border_size, border_size))
    

    # Ensure the image is resized to 910x100
    resized_image = bordered_image.resize((910, 100))  # Natural resolution
    annotated_images.append(resized_image)

# Determine collage dimensions
collage_width = 910  # Each image's width
collage_images_per_row = 10  # Number of images per row
collage_rows = ceil(len(annotated_images) / collage_images_per_row)
collage_height = collage_rows * 100  # 100 pixels per image height

# Create the blank collage canvas
collage = Image.new("RGB", (collage_width * collage_images_per_row, collage_height), color="white")

# Paste each image into the collage
for i, annotated_image in enumerate(annotated_images):
    row = i // collage_images_per_row
    col = i % collage_images_per_row
    x_offset = col * 910
    y_offset = row * 100
    collage.paste(annotated_image, (x_offset, y_offset))

# Save the collage
collage.save(output_collage)
print(f"Collage saved to {output_collage}")
collage.show()