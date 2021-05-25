import os
import cv2

# generates label txt files in yolo style
# labels the images and makes the bounding box the entire image

''' Generate label txt files in yolo style.
    [class, x_center, y_center, w, h].
    Takes in a structured folder heirarchy where all images in the 
        same folder belong to the same class.
    Params:
    Returns:
        None
    Author: 
        Jay
    Date: 
        2021/05/19
    Suggestions: (Natasha)
        1. Wrap everything up in a run function
        2. Add __main__ function and call run
        3. Add argparse for the path
        4. Add timer function
'''

root = r'../data/ChineseFoodNet/dataset_release/release_data/val/'

floder_list = []
for i in os.listdir(root):
    floder_list.append(i)

for roots, dir, files in os.walk(root):
    for file in files:
        if file.endswith((".jpg", ".jpeg")):
            file = os.path.join(roots, file)
            img = cv2.imread(file)
            img_w, img_h = img.shape[:2]

            file_pathname, ext = os.path.splitext(file)
            txt_file = file_pathname + ".txt"
            floder_name = os.path.basename(roots)
            floder_index = floder_list.index(floder_name)
            with open(txt_file, "w") as f:
                f.write(str(floder_index) + " " + str(0.5) + " " + str(0.5) + " " + str(1) + " " + str(1))