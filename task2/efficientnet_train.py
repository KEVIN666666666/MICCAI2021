import tensorflow as tf
from tqdm import tqdm
from efficientnet_model import efficientnet_b7 as create_model
import os


img_size = {"B0": 224,
            "B1": 240,
            "B2": 260,
            "B3": 300,
            "B4": 380,
            "B5": 456,
            "B6": 528,
            "B7": 600}

num_model = "B7"
im_height = im_width = img_size[num_model]
num_classes = 3
freeze_layers = True
initial_lr = 0.01

model = create_model(num_classes=num_classes)

# load weights
pre_weights_path = './efficientnetb7.h5'
# assert os.path.exists(pre_weights_path), "cannot find {}".format(pre_weights_path)
if os.path.exists(pre_weights_path):
    model.load_weights(pre_weights_path, by_name=True, skip_mismatch=True)

    # freeze bottom layers
    if freeze_layers:
        unfreeze_layers = ["top_conv", "top_bn", "predictions"]
        for layer in model.layers:
            if layer.name not in unfreeze_layers:
                layer.trainable = False
            else:
                print("training {}".format(layer.name))

model.summary()

# TODO: read dataset

# TODO: training

print("Finish.")
