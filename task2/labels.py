import pandas as pd
import cv2


data_frame = pd.read_excel("../fovea_localization_training_GT.xlsx")
original_data = data_frame.to_numpy()

cropped_information = pd.read_excel("../pre_data_V1/boxlist.xlsx", header=None)
cropped_information = cropped_information.to_numpy()

# original_data = [index, column, row]
# cropped_information = [row, column]
original_data[:, 1] = original_data[:, 1] - cropped_information[:, 1]
original_data[:, 2] = original_data[:, 2] - cropped_information[:, 0]


img = cv2.imread("../pre_data_V1/1.jpg")
point = (int(original_data[0, 1]), int(original_data[0, 2]))  # column, row
cv2.rectangle(img, pt1=(point[0] - 5, point[1] - 5), pt2=(point[0] + 5, point[1] + 5), color=(0, 255, 0))
cv2.imshow("1", img)
cv2.waitKey()
cv2.destroyAllWindows()

with open("task_2.txt", "w", encoding="utf-8") as file:
    file.write("index" + "\t" + "column" + "\t" + "row" + "\n")
    for index, val in enumerate(original_data):
        file.write(str(int(val[0])) + "\t" + str(int(val[1])) + "\t" + str(int(val[2])) + "\n")

print("Finish")
