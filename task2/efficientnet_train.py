import numpy as np
import tensorflow as tf
from tqdm import tqdm
from efficientnet_model import efficientnet_b7 as create_model
from dataset import get_dataset, split_dataset
from tensorflow.keras.losses import Loss
from matplotlib import pyplot as plt
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
freeze_layers = False  # freeze pretrained convolution layers
initial_lr = 1e-4  # 0.0001

model = create_model(num_classes=num_classes)

# load weights
# TODO: Download pretrained model
# Download from https://storage.googleapis.com/keras-applications/efficientnetb7.h5
pre_weights_path = '../../efficientnetb7.h5'
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
train_dataset = tf.data.Dataset.from_tensor_slices((train_X, train_Y))
test_dataset = tf.data.Dataset.from_tensor_slices((test_X, test_Y))


# TODO: Standardization, Augmentation
def normalization(image, label):
    # normalization
    return image / 255, label


def augment(image, label):
    # data augmentation
    return image, label


BATCH_SIZE = 1
train_dataset = train_dataset.map(normalization).map(augment).batch(BATCH_SIZE)
test_dataset = test_dataset.map(normalization).batch(BATCH_SIZE)


# model compile
# Custom loss function
class AverageEuclideanDistance(Loss):
    def call(self, y_true, y_pred):
        # print(y_true, y_pred)
        y_pred = tf.convert_to_tensor(y_pred)
        y_true = tf.cast(y_true, y_pred.dtype)
        return tf.reduce_mean(tf.sqrt(tf.square(y_pred - y_true)), axis=-1)


def score(loss):
    return 1 / (loss + 0.1)


def mean(l: list):
    return sum(l) / len(l)


loss_object = AverageEuclideanDistance()
optimizer = tf.keras.optimizers.Adam(learning_rate=initial_lr)

train_loss = tf.keras.metrics.Mean(name='train_loss')
# train_accuracy = tf.keras.metrics.MeanSquaredError(name='train_accuracy')

val_loss = tf.keras.metrics.Mean(name='val_loss')
# val_accuracy = tf.keras.metrics.MeanSquaredError(name='val_accuracy')


@tf.function
def train_step(train_images, train_labels):
    with tf.GradientTape() as tape:
        output = model(train_images, training=True)
        loss = loss_object(train_labels, output)
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    train_loss(loss)
    # train_accuracy(train_labels, output)
    return output, loss


@tf.function
def val_step(val_images, val_labels):
    output = model(val_images, training=False)
    loss = loss_object(val_labels, output)

    val_loss(loss)
    # val_accuracy(val_labels, output)
    return output, loss


def visualize_batch(images, labels, output, epoch, index, prefix):
    image = images[0].numpy()
    point = labels[0]
    shape = image.shape
    point = (int(point[0] * shape[0]), int(point[1] * shape[1]))
    image[point[0] - 5:point[0] + 5, point[1] - 5:point[1] + 5, :] = [1, 1, 1]
    pred = output[0]
    pred = (int(pred[0] * shape[0]), int(pred[1] * shape[1]))
    image[pred[0] - 5:pred[0] + 5, pred[1] - 5:pred[1] + 5, :] = [0, 0, 1]
    plt.imshow(image)
    # plt.show()
    plt.savefig(f"{prefix}_epoch{epoch}_index{index}.png")


best_score = 0
epochs = 100
for epoch in range(epochs):
    train_loss.reset_states()  # clear history info
    # train_accuracy.reset_states()  # clear history info
    val_loss.reset_states()  # clear history info
    # val_accuracy.reset_states()  # clear history info

    # train
    train_bar = tqdm(train_dataset)
    train_scores = []
    for index, (images, labels) in enumerate(train_bar):
        output, loss = train_step(images, labels)
        train_scores.append(score(loss))
        # Save and Visualize the batch
        if index % 10 == 0:
            save_dir_prefix = "./train_image/train_image"
            visualize_batch(images, labels, output, epoch, index, save_dir_prefix)

        # print train process
        train_bar.desc = "train epoch[{}/{}] AED loss:{:.3f}, score:{:.3f}".format(epoch + 1,
                                                                                   epochs,
                                                                                   train_loss.result(),
                                                                                   mean(train_scores))

    val_bar = tqdm(test_dataset)
    validation_images = []
    validation_labels = []
    validation_predictions = []
    validation_scores = []
    for images, labels in val_bar:
        # Save the validation image, label and prediction
        output, loss = val_step(images, labels)
        validation_scores.append(score(loss))
        validation_images.append(images)
        validation_labels.append(labels)
        validation_predictions.append(output)

        # print val process
        val_bar.desc = "valid epoch[{}/{}] AED loss:{:.3f}, score:{:.3f}".format(epoch + 1,
                                                                                 epochs,
                                                                                 val_loss.result(),
                                                                                 mean(validation_scores))

    # # writing training loss and acc
    # print("train loss", train_loss.result(), epoch)
    # print("train accuracy", train_accuracy.result(), epoch)
    #
    # # writing validation loss and acc
    # print("validation loss", val_loss.result(), epoch)
    # print("validation accuracy", val_accuracy.result(), epoch)

    # only save best weights
    # Loss and Accuracy Evaluate
    if mean(validation_scores) > best_score:
        best_score = mean(validation_scores)
        save_name = "./save_weights/efficientnet.ckpt"
        model.save_weights(save_name, save_format="tf")

        # visualize the result
        prefix = "./validation_image/validation_image"
        for index, (images, labels, output) in enumerate(
                zip(validation_images, validation_labels, validation_predictions)):
            visualize_batch(images, labels, output, epoch, index, prefix)

print("Finish.")
