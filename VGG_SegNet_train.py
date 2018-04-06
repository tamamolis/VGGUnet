import LoadBatches
from VGGSegnet import VGGUnet
from VGGSegnet import set_keras_backend

train_images_path = "/Users/kate/PycharmProjects/make_data/VGG_SegNet/train/"
train_segs_path = "/Users/kate/PycharmProjects/make_data/VGG_SegNet/trainmask/"
train_batch_size = 6
n_classes = 7
input_height = 416
input_width = 608
save_weights_path = "/Users/kate/PycharmProjects/Segnet/weights/"
epochs = 100
# load_weights = "VGGSegNet.weights.best.hdf5"

val_images_path = "/Users/kate/PycharmProjects/make_data/VGG_SegNet/val/"
val_segs_path = "/Users/kate/PycharmProjects/make_data/VGG_SegNet/valmask/"
val_batch_size = 6

if __name__ == '__main__':
    set_keras_backend("theano")
    m, output_width, output_height = VGGUnet(7, vgg_level=3)

    m.compile(loss='categorical_crossentropy',
              optimizer='adadelta',
              metrics=['accuracy'])

    # if len(load_weights) > 0:
    #     m.load_weights(load_weights)

    print("Model output shape", m.output_shape)

    G = LoadBatches.imageSegmentationGenerator(train_images_path, train_segs_path, train_batch_size, n_classes,
                                               input_height, input_width, output_height, output_width)
    G2 = LoadBatches.imageSegmentationGenerator(val_images_path, val_segs_path, val_batch_size, n_classes, input_height,
                                                input_width, output_height, output_width)

    for ep in range(epochs):
        m.fit_generator(G, 512, validation_data=G2, validation_steps=200, epochs=1)
        m.save_weights(save_weights_path + "." + str(ep))
        m.save(save_weights_path + ".model." + str(ep))
