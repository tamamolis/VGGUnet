import LoadBatches
import glob
from VGGSegnet import VGGUnet, set_keras_backend
import numpy as np
import cv2
import os

Building = [0, 0, 255]
Grass = [0, 255, 0]
Development = [255, 255, 0]  # стройка
Concrete = [255, 255, 255]  # бетон
Roads = [0, 255, 255]
NotAirplanes = [252, 40, 252]
Unlabelled = [255, 0, 0]

colors = np.array([Building, Grass, Development, Concrete, Roads, NotAirplanes, Unlabelled])

n_classes = 7
images_path = 'data/test/'
input_width = 416
input_height = 608
save_weights_path = 'weights/VGGSegNet/'
output_path = 'imgs_results/res5/'
images_path = 'data/res/'
input_width = 416
input_height = 608
save_weights_path = 'weights/VGGSegNet/'
output_path = 'imgs_results/res7/'
DataPath = 'data/'


def visualize(temp):
    r = temp.copy()
    g = temp.copy()
    b = temp.copy()
    for l in range(0, 7):
        r[temp == l] = colors[l, 0]
        g[temp == l] = colors[l, 1]
        b[temp == l] = colors[l, 2]

    rgb = np.zeros((temp.shape[0], temp.shape[1], 3))
    rgb[:, :, 0] = (r / 255.0)
    rgb[:, :, 1] = (g / 255.0)
    rgb[:, :, 2] = (b / 255.0)
    return rgb


if __name__ == '__main__':
    set_keras_backend("theano")
    m, output_width, output_height = VGGUnet(7, vgg_level=3)
    m.load_weights(save_weights_path + "weights.best.hdf5")
    m.compile(loss='categorical_crossentropy',
              optimizer='adadelta',
              metrics=['accuracy'])

    images = glob.glob(images_path + "*.png")
    images.sort()
    gt = []
    with open(DataPath + 'test.txt') as f:
        txt = f.readlines()
        txt = [line.split('\n') for line in txt]
    print(txt)

    for i in range(len(txt)):
        print(os.getcwd() + '/data/test/' + txt[i][0])
        gt.append(cv2.imread(os.getcwd() + '/data/test/' + txt[i][0]))
    # gt = []
    # with open(DataPath + 'test.txt') as f:
    #     txt = f.readlines()
    #     txt = [line.split('\n') for line in txt]
    # print(txt)
    #
    # for i in range(len(txt)):
    #     print(os.getcwd() + '/data/test/' + txt[i][0])
    #     gt.append(cv2.imread(os.getcwd() + '/data/test/' + txt[i][0]))
    i = 0
    for imgName in images:

        outName = imgName.replace(images_path, output_path)
        X = LoadBatches.getImageArr(imgName, input_height, input_width)
        pr = m.predict(np.array([X]))[0]
        pr = pr.reshape((output_height, output_width, n_classes)).argmax(axis=2)
        seg_img = np.zeros((output_height, output_width, 3))
        for c in range(n_classes):
            seg_img[:, :, 0] += ((pr[:, :] == c) * (colors[c][0])).astype('uint8')
            seg_img[:, :, 1] += ((pr[:, :] == c) * (colors[c][1])).astype('uint8')
            seg_img[:, :, 2] += ((pr[:, :] == c) * (colors[c][2])).astype('uint8')
        cv2.imwrite(outName, seg_img)
        cv2.imwrite(outName + str(i) + '.png', gt[i])
        # cv2.imwrite(outName + str(i) + '.png', gt[i])
        i += 1