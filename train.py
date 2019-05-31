import LoadBatches
from VGGSegnetModel import VGGUnet
from VGGSegnetModel import set_keras_backend
from keras.callbacks import ModelCheckpoint
from keras import backend as K
import numpy as np
# import tensorflow as tf
import theano.tensor as T

train_images_path = "/Users/kate/PycharmProjects/make_data/VGG_Unet_UNITE_COLORS/bw_aug/train/"
train_segs_path = "/Users/kate/PycharmProjects/make_data/VGG_Unet_UNITE_COLORS/bw_aug/trainmask/"
train_batch_size = 6
n_classes = 5  # было 7

input_height = 416
input_width = 608
save_weights_path = "/Users/kate/PycharmProjects/VGGUnet/weights/"
epochs = 30
load_weights = ""
checkpoint_filepath = ""  # "weights/VGGUnet.weights.best.hdf5"

val_images_path = "/Users/kate/PycharmProjects/make_data/VGG_Unet_UNITE_COLORS/bw_aug/val/"
val_segs_path = "/Users/kate/PycharmProjects/make_data/VGG_Unet_UNITE_COLORS/bw_aug/valmask/"
val_batch_size = 6


def dice_coef(y_true, y_pred):
    smooth = 1e-7
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)


def dice_coef_multilabel(y_true, y_pred):
    dice = [0, 0, 0, 0, 0]
    for index in range(n_classes):
        dice -= dice_coef(y_true[:, index, :], y_pred[:, index, :])
    return dice

gamma = 2.0
alpha = 0.25

def focal_loss(y_true, y_pred):
    epsilon = K.epsilon()
    y_pred = K.clip(y_pred, epsilon, 1.0 - epsilon)
    cross_entropy = -y_true * K.log(y_pred)
    weight = alpha * y_true * K.pow((1 - y_pred), gamma)
    loss = weight * cross_entropy
    loss = K.sum(loss, axis=1)
    return loss

def categorical_focal_loss(y_true, y_pred):

    focal = [0, 0, 0, 0, 0]
    for index in range(n_classes):
        focal -= (focal_loss(y_true[:, index, :], y_pred[:, index, :]))

    return focal


if __name__ == '__main__':

    set_keras_backend("theano")
    m, output_width, output_height = VGGUnet(n_classes, vgg_level=3)

    m.compile(loss=categorical_focal_loss, # 2.0 и 0.25 было
              optimizer='adam',
              metrics=['accuracy'])

    # m.compile(loss=dice_coef_multilabel,
    #           optimizer='adadelta',
    #           metrics=['accuracy'])

    # m.compile(loss='categorical_crossentropy',
    #           optimizer='adadelta',
    #           metrics=['accuracy'])

    if len(load_weights) > 0:
        m.load_weights(load_weights)

    print("Model output shape", m.output_shape)

    G = LoadBatches.imageSegmentationGenerator(train_images_path, train_segs_path, train_batch_size, n_classes,
                                               input_height, input_width, output_height, output_width)
    G2 = LoadBatches.imageSegmentationGenerator(val_images_path, val_segs_path, val_batch_size, n_classes, input_height,
                                                input_width, output_height, output_width)

    print()
    # checkpoint = ModelCheckpoint(checkpoint_filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
    # callbacks_list = [checkpoint]

    for ep in range(epochs):
        m.fit_generator(G, steps_per_epoch=int(924 / train_batch_size), validation_data=G2,  # callbacks=callbacks_list,
                        validation_steps=int(116 / val_batch_size), epochs=epochs)
        m.save_weights(save_weights_path + "." + str(ep))
        m.save(save_weights_path + ".model." + str(ep))
