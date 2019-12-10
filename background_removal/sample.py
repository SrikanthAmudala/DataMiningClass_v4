from sklearn.mixture import BayesianGaussianMixture, GaussianMixture
from sklearn.model_selection import train_test_split
import sklearn

import pandas

import cv2

bg_input_path = "/home/k_mathin/PycharmProjects/DataMiningClass_v4/background_removal/imgs/bg.jpeg"
foreground_input_path = "/home/k_mathin/PycharmProjects/DataMiningClass_v4/background_removal/imgs/foreground.jpeg"
bg = cv2.imread(bg_input_path, 0)
fg = cv2.imread(foreground_input_path, 0)


bg_o_shape = bg.shape
fg_o_shape = fg.shape
k = 5
bg_new_data = bg.reshape(-1,1)
fg_new_data = fg.reshape(-1,1)

bg_vgmm = BayesianGaussianMixture(n_components=k)
fg_vgmm = BayesianGaussianMixture(n_components=k)


# vgmm = GaussianMixture(n_components=k)
bg_vgmm = bg_vgmm.fit(bg_new_data)
fg_vgmm = fg_vgmm.fit(fg_new_data)
bg_cluater = bg_vgmm.predict(bg_new_data)
fg_cluater = fg_vgmm.predict(fg_new_data)

# Reshape the input data to the orignal shape
bg_img_cluater = bg_cluater.reshape(bg_o_shape[0], bg_o_shape[1])
fg_img_cluater = fg_cluater.reshape(fg_o_shape[0], fg_o_shape[1])
from matplotlib import pyplot
pyplot.subplot(2,1,1)
pyplot.imshow(bg_img_cluater)
pyplot.subplot(2,1,2)
pyplot.imshow(fg_img_cluater)
pyplot.show()





