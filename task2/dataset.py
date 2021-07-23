from matplotlib import pyplot as plt
from PIL import Image
import numpy as np
import os
import pandas as pd


def split_dataset(data, label, ratio, seed):
    total_data = len(data)
    assert (len(data) == len(label))
    np.random.seed(seed)
    selected_index = np.random.randint(0, total_data, int(total_data * ratio))

    # TODO: split dataset base on the ratio

    return


def get_dataset(PATH, IMAGE_SIZE, label_file, check=False):
    """
    return data in RGB mode, as well as its label for task 2
    the label is returned in a normalized ratio of the image
    :param PATH: image path, only accept jpg image
    :param IMAGE_SIZE: the desired output size
    :param label_file: the label file path, named task_2.txt
    :param check: check the label after normalized in resized image
    :return: numpy format image and label
    """

    image_names = os.listdir(PATH)
    data_frame = pd.read_csv(label_file, sep="\t").to_numpy()

    data = []
    for image_name in image_names:
        if not image_name.endswith(".jpg"):
            continue  # not a jpg file
        image = Image.open(PATH + image_name).copy()  # RGB mode
        index = int(image_name.removesuffix(".jpg"))
        assert(data_frame[index - 1][0] == index)
        shape = image.size
        # index, column, row
        point = (data_frame[index - 1][2] / shape[0], data_frame[index - 1][1] / shape[1])  # row, column
        image = image.resize((IMAGE_SIZE, IMAGE_SIZE))
        data.append((index, np.array(image), point))

    np_data = np.zeros((len(data), IMAGE_SIZE, IMAGE_SIZE, 3), dtype=np.uint8)
    np_label = np.zeros((len(data), 2))
    for index, image, point in data:
        np_data[index - 1, :, :, :] = image
        np_label[index - 1, :] = point

    if check:
        for image, point in zip(np_data, np_label):
            shape = image.shape
            point = (int(point[0] * shape[0]), int(point[1] * shape[1]))
            image[point[0]-5:point[0]+5, point[1]-5:point[1]+5, :] = [255, 255, 255]
            plt.imshow(image)
            plt.show()

    return np_data, np_label


if __name__ == "__main__":
    IMAGE_SIZE = 1200
    PATH = "../pre_data_V1/"
    label_file = "./task_2.txt"
    check = False

    data, label = get_dataset(PATH, IMAGE_SIZE, label_file, check)

    RATIO = 0.1
    SEED = 0
    split_dataset(data, label, RATIO, SEED)

    print("Finish.")
