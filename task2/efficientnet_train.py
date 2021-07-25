import numpy as np
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

# TODO(Kuo): training
loss_object = tf.keras.losses.MeanSquaredError()
optimizer = tf.keras.optimizers.Adam(learning_rate=initial_lr)

train_loss = tf.keras.metrics.Mean(name='train_loss')
train_accuracy = tf.keras.metrics.Accuracy(name='train_accuracy')

val_loss = tf.keras.metrics.Mean(name='val_loss')
val_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='val_accuracy')


@tf.function
def train_step(train_images, train_labels):
    with tf.GradientTape() as tape:
        output = model(train_images, training=True)
        loss = loss_object(train_labels, output)
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    train_loss(loss)
    train_accuracy(train_labels, output)


@tf.function
def val_step(val_images, val_labels):
    output = model(val_images, training=False)
    loss = loss_object(val_labels, output)

    val_loss(loss)
    val_accuracy(val_labels, output)


best_val_acc = 0.
epochs = 30
for epoch in range(epochs):
    train_loss.reset_states()  # clear history info
    train_accuracy.reset_states()  # clear history info
    val_loss.reset_states()  # clear history info
    val_accuracy.reset_states()  # clear history info

    # train
    train_bar = tqdm(zip(train_X, train_Y))
    for images, labels in train_bar:
        images = np.expand_dims(images, axis=0)
        labels = np.expand_dims(labels, axis=0)
        train_step(images, labels)

        # print train process
        train_bar.desc = "train epoch[{}/{}] loss:{:.3f}, acc:{:.3f}".format(epoch + 1,
                                                                             epochs,
                                                                             train_loss.result(),
                                                                             train_accuracy.result())
    val_bar = tqdm(zip(test_X, test_Y))
    for images, labels in val_bar:
        images = np.expand_dims(images, axis=0)
        labels = np.expand_dims(labels, axis=0)
        val_step(images, labels)

        # print val process
        val_bar.desc = "valid epoch[{}/{}] loss:{:.3f}, acc:{:.3f}".format(epoch + 1,
                                                                           epochs,
                                                                           val_loss.result(),
                                                                           val_accuracy.result())

    # writing training loss and acc
    print("train loss", train_loss.result(), epoch)
    print("train accuracy", train_accuracy.result(), epoch)

    # writing validation loss and acc
    print("validation loss", val_loss.result(), epoch)
    print("validation accuracy", val_accuracy.result(), epoch)

    # only save best weights
    if val_accuracy.result() > best_val_acc:
        best_val_acc = val_accuracy.result()
        save_name = "./save_weights/efficientnet.ckpt"
        model.save_weights(save_name, save_format="tf")


print("Finish.")
