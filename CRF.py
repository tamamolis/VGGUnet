import numpy as np
import pydensecrf.densecrf as dcrf
from skimage.io import imread, imsave
from pydensecrf.utils import unary_from_labels
from skimage.color import gray2rgb
from skimage.color import rgb2gray
import cv2

Building = [0, 0, 255]
Grass = [0, 255, 0]
Development = [255, 255, 0]  # стройка
Concrete = [255, 255, 255]  # бетон
Roads = [0, 255, 255]
NotAirplanes = [252, 40, 252]
Unlabelled = [255, 0, 0]

legend_list = [Building, Grass, Development, Concrete, Roads, NotAirplanes, Unlabelled]

height = 416
width = 608


def visualize(temp):
    color = np.array([Building, Grass, Development, Concrete, Roads, NotAirplanes, Unlabelled])
    r = temp.copy()
    g = temp.copy()
    b = temp.copy()
    for l in range(0, 7):
        r[temp == l] = color[l, 0]
        g[temp == l] = color[l, 1]
        b[temp == l] = color[l, 2]

    rgb = np.zeros((temp.shape[0], temp.shape[1], 3))
    rgb[:, :, 0] = (r / 255.0)
    rgb[:, :, 1] = (g / 255.0)
    rgb[:, :, 2] = (b / 255.0)
    return rgb


def euclidean_metric(input):
    #  сумма модулей разности
    total = 100000
    index = -1

    for legend in legend_list:
        new_total = abs(input[0] - legend[0]) + abs(input[1] - legend[1]) + abs(input[2] - legend[2])
        if new_total < total:
            total = new_total
            index = legend_list.index(legend)
    return index


def table(img):

    h = np.shape(img)[0]
    w = np.shape(img)[1]
    new_array = np.zeros((h, w))
    for i in range(h):
        for j in range(w):
            # print(img[i][j])
            buf = euclidean_metric(img[i][j])
            # print(buf)
            new_array[i][j] = buf
    return new_array


def crf(original_image, annotated_image, output_image, use_2d=True):
    # Converting annotated image to RGB if it is Gray scale

    if len(annotated_image.shape) < 3:
        annotated_image = gray2rgb(annotated_image)

    annotated_label = table(annotated_image)
    print('table done')

    print(np.shape(annotated_image))
    colors, labels = np.unique(annotated_label, return_inverse=True)

    colors = np.array([0, 1, 2, 3, 4, 5, 6])
    print(colors)
    colorize = np.empty((len(colors), 3), np.uint8)
    colorize[:,0] = (colors & 0x0000FF)
    colorize[:,1] = (colors & 0x00FF00) >> 8
    colorize[:,2] = (colors & 0xFF0000) >> 16

    n_labels = len(set(labels.flat))

    f = open('labels.txt', 'w')
    for lines in labels:
        f.write(str(lines))
    f.close()

    labels = np.array(labels)
    if use_2d:
        d = dcrf.DenseCRF2D(original_image.shape[1], original_image.shape[0], n_labels)

        # get unary potentials (neg log probability)
        print('labels: ', labels, len(labels), np.shape(labels))
        print('n_lables: ', n_labels)
        U = unary_from_labels(labels, n_labels, gt_prob=0.7, zero_unsure=False)
        d.setUnaryEnergy(U)

        # This adds the color-independent term, features are the locations only.
        d.addPairwiseGaussian(sxy=(3, 3), compat=3, kernel=dcrf.DIAG_KERNEL,
                              normalization=dcrf.NORMALIZE_SYMMETRIC)

        # This adds the color-dependent term, i.e. features are (x,y,r,g,b).
        d.addPairwiseBilateral(sxy=(80, 80), srgb=(13, 13, 13), rgbim=original_image,
                               compat=10,
                               kernel=dcrf.DIAG_KERNEL,
                               normalization=dcrf.NORMALIZE_SYMMETRIC)

    Q = d.inference(5)
    MAP = np.argmax(Q, axis=0)
    # print(MAP)
    # MAP = colorize[MAP, :]
    # imsave(output_image, MAP.reshape(original_image.shape))

    rgb = visualize(MAP.reshape(np.shape(original_image)[0], np.shape(original_image)[1]))
    imsave(output_image, rgb)


if __name__ == '__main__':

    orig = 'img_res/Сочи.jpg'
    seg = 'img_res/Сочи_сег.png'

    image = imread(orig)
    seg_image = imread(seg)

    crf(image, seg_image, "img_res/crf_res_Сочи.jpg")