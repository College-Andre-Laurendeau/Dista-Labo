# Model
import torch

model = torch.hub.load('ultralytics/yolov5', 'yolov5m6', pretrained=True)


# Images
dir = 'https://github.com/ultralytics/yolov5/raw/master/data/images/'
imgs = [dir + f for f in ('zidane.jpg', 'bus.jpg')]  # batched list of images

# Inference
results = model(imgs)

# Results
results.print()
results.show()  # or .show()