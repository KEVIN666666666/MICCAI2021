import tensorflow as tf
from tqdm import tqdm
from efficientnet_model import efficientnet_b7 as create_model
from dataset import get_dataset, split_dataset
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
num_classes = 2
freeze_layers = True
initial_lr = 1e-4  # 0.0001

model = create_model(num_classes=num_classes)

# load weights
# TODO: Download pretrained model
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

# read dataset
IMAGE_SIZE = im_height
PATH = "../pre_data_V1/"
label_file = "./task_2.txt"
check = False

data, label = get_dataset(PATH, IMAGE_SIZE, label_file, check)

RATIO = 0.1
SEED = 0
train_X, train_Y, test_X, test_Y = split_dataset(data, label, RATIO, SEED)

# TODO: training

print("Finish.")
