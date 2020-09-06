from models.vgg16 import VGG16
from keras.layers import TimeDistributed, concatenate, Flatten, Dense, Dropout, Conv3D, MaxPool3D
from keras.regularizers import l2
from keras.models import Model, Sequential


def multi_vgg(name, input_shape, time, classes, dropout, l2_reg, weights, include_top):
    model = VGG16(
        name=name,
        input_shape=input_shape,
        classes=classes,
        dropout=dropout,
        l2_reg=l2_reg,
        weights=weights,
        include_top=False
    )

    distributed_shape = (time,) + input_shape

    time_model = Sequential()
    time_model.add(TimeDistributed((model), input_shape=distributed_shape))

    # Freeze all layers
    for layer in time_model.layers:
        layer.trainable = False

    return time_model


def STVGG16(name, temporal_lenght, time, classes, dropout, l2_reg, spatial_weights, temporal_weights):

    spatial_model = multi_vgg(
        name='spatial',
        input_shape=(224, 224, 3),
        time=time,
        classes=classes,
        dropout=dropout,
        l2_reg=l2_reg,
        weights=spatial_weights,
        include_top=False
    )

    temporal_model = multi_vgg(
        name='temporal',
        input_shape=(224, 224, 2 * temporal_lenght),
        time=time,
        classes=classes,
        dropout=dropout,
        l2_reg=l2_reg,
        weights=temporal_weights,
        include_top=False
    )

    model = concatenate([spatial_model.output, temporal_model.output])

    # Classification block
    fc = Conv3D(name='conv_fusion', kernel_size=(3, 3, 3), strides=1, filters=512)(model)
    fc = MaxPool3D(name='3d_pooling', strides=1, pool_size=2)(fc)
    fc = Flatten(name='flatten')(fc)
    fc = Dense(4096, activation='relu', name='fc1', kernel_regularizer=l2(l2_reg))(fc)
    fc = Dropout(rate=dropout, name='dropout1')(fc)
    fc = Dense(4096, activation='relu', name='fc2', kernel_regularizer=l2(l2_reg))(fc)
    fc = Dropout(rate=dropout, name='dropout2')(fc)
    fc = Dense(classes, activation='softmax', name='predictions')(fc)

    return Model(name=name, inputs=[spatial_model.input, temporal_model.input], outputs=fc)


if __name__ == '__main__':
    model = STVGG16(
        name='st_model',
        temporal_lenght=10,
        time=5,
        classes=101,
        dropout=0.5,
        l2_reg=1e-5,
        spatial_weights='imagenet',
        temporal_weights='imagenet'
    )

    model.summary()

    print([layer.trainable for layer in model.layers])
