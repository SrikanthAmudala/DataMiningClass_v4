"""
Data pre processing
https://vision.eng.au.dk/plant-seedlings-dataset/
"""
import os
import cv2
import numpy as np
# base_url = "/home/k_mathin/PycharmProjects/DataMiningClass/datasets/classification/imagesfromwild"
base_url = "/home/k_mathin/PycharmProjects/DataMiningClass/datasets/classification/weedplant_full_dataset"
dir_list = os.listdir(base_url)


weed_dataset = []
targets = []
for weedtype in dir_list:
    weed_cat = os.listdir(base_url+"/"+weedtype)
    temp = []
    targets.append(weedtype)
    for i in weed_cat:
        # temp.append(base_url+"/"+weedtype+"/"+i)
        img = cv2.imread(base_url + "/" + weedtype + "/" + i, 0)
        img_resize = cv2.resize(img, (60,60))
        temp.append(img_resize)
    weed_dataset.append(np.asarray(temp))


weed_dataset =np.asarray(weed_dataset)
weed_dataset_reshape = weed_dataset.reshape(8,12,-1)

dataset = []
for i, t in zip(weed_dataset_reshape, targets):
    for j in i:
        j = list(j)
        j.append(t)
        dataset.append(j)

# for i, t in zip(weed_dataset, targets):
#     for j in i:
#         j = list(j.reshape(-1))
#         j.append(t)
#         dataset.append(j)