from __future__ import absolute_import
from __future__ import print_function
import keras.models as models
from keras.callbacks import ModelCheckpoint

import numpy as np

np.random.seed(0o7)

data_shape = 360*480

if __name__ == '__main__':

    train_data = np.load('./data/train_data.npy')
    train_label = np.load('./data/train_label.npy')

    val_data = np.load('./data/val_data.npy')
    val_label = np.load('./data/val_label.npy')

    print(np.shape(train_label), np.shape(val_label))

    with open('segNet_basic_model.json') as model_file:
        segnet_basic = models.model_from_json(model_file.read())

    segnet_basic.compile(loss="categorical_crossentropy", optimizer='adadelta', metrics=["accuracy"])
    filepath = "weights.best.hdf5"

    checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
    callbacks_list = [checkpoint]

    nb_epoch = 100
    batch_size = 6

    history = segnet_basic.fit(train_data, train_label, callbacks=callbacks_list, batch_size=batch_size,
                               nb_epoch=nb_epoch, verbose=1, class_weight='auto',
                               validation_data=(val_data, val_label), shuffle=True)

    segnet_basic.save_weights('weights/model_weight_{}.hdf5'.format(nb_epoch))