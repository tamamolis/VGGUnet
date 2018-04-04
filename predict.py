from __future__ import absolute_import
from __future__ import print_function
import os
import matplotlib.pyplot as plt
import keras.models as models

import cv2
import numpy as np
from scipy import misc
from keras.preprocessing.image import array_to_img

from PIL import Image

np.random.seed(0o7)
data_shape = 512*512
batch_size = 6
DataPath = 'data/'

Building = [0, 0, 255]
Grass = [0, 255, 0]
Development = [255, 255, 0]  # стройка
Concrete = [255, 255, 255]  # бетон
Roads = [0, 255, 255]
NotAirplanes = [252, 40, 252]
Unlabelled = [255, 0, 0]

label_colours = np.array([Unlabelled, Grass, Roads, Concrete, Development, NotAirplanes, Building])
#  legend_list = [[0, 0, 255], [0, 255, 0], [255, 255, 0], [255, 255, 255], [0, 255, 255], [255, 0, 255], [255, 0, 0]]


def visualize(temp, plot=True):
    r = temp.copy()
    g = temp.copy()
    b = temp.copy()
    for l in range(0, 7):
        r[temp == l] = label_colours[l, 0]
        g[temp == l] = label_colours[l, 1]
        b[temp == l] = label_colours[l, 2]

    rgb = np.zeros((temp.shape[0], temp.shape[1], 3))
    rgb[:, :, 0] = (r/255.0)
    rgb[:, :, 1] = (g/255.0)
    rgb[:, :, 2] = (b/255.0)
    if plot:
        plt.imshow(rgb)
    else:
        return rgb


if __name__ == '__main__':

    with open('segNet_basic_model.json') as model_file:
        segnet_basic = models.model_from_json(model_file.read())

    segnet_basic.load_weights("weights/weights_best.hdf5")
    segnet_basic.compile(loss="categorical_crossentropy", optimizer='adadelta', metrics=["accuracy"])

    test_data = np.load(DataPath + 'test_data.npy')

    gt = []
    with open(DataPath + 'test.txt') as f:
        txt = f.readlines()
        txt = [line.split(' ') for line in txt]

    for i in range(len(txt)):
        gt.append(cv2.imread('/Users/kate/PycharmProjects/make_data/' + txt[i][0][11:]))

    for i in range(test_data.shape[0]):
        output = segnet_basic.predict(test_data[i:i+1])

        pred = np.argmax(np.squeeze(output[0]), axis=1).reshape((360, 480))

        rgb = visualize(pred, False)

        label = array_to_img(gt[i])
        new_rgb = array_to_img(rgb)

        print(np.shape(new_rgb), np.shape(label))
        vis = np.concatenate((new_rgb, label), axis=1)
        vis = array_to_img(vis)
        vis.save('imgs_results/res4/img_label' + str(i) + str(i) + '.png')
