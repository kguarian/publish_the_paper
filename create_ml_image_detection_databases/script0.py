# %% [markdown]
# initialize yolov11 model.

# %%
from ultralytics import YOLO



# %%
# Load a model
# load a pretrained model (recommended for training)
model = YOLO("yolo11n.pt") 


# %%
results = model.train(data="dataset.yaml", epochs=1, device="mps")
if results is None:
   exit(0)

# %%
path = "/Users/kenton/HOME/coding/python/publish_the_paper/create_ml_image_detection_databases/test_data/"


# %%
for i in range(0, 10):
    img = path + "sig_" + str(i) + ".png"
    results = model(img)
    
    print(results.xyxy[0]) # print img1 predictions (pixels)
    results.save() # or .show(), .crop(), .pandas(), etc.


