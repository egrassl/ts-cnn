"""
Adaptation of the VGG16 model for training with regularizations.
The original model can be found at:
https://github.com/keras-team/keras-applications/blob/master/keras_applications/vgg16.py
"""


from keras import layers, regularizers, models, utils, backend
import keras_applications
import os


def VGG16(name, input_shape, classes: int, dropout: float, l2_reg: float, weights, include_top=False, pooling=None):

    if not (weights in {'imagenet', None} or os.path.exists(weights)):
        raise ValueError('The `weights` argument should be either '
                         '`None` (random initialization), `imagenet` '
                         '(pre-training on ImageNet), '
                         'or the path to the weights file to be loaded.')

    # Returns error if wrong input shape is provided
    if input_shape[0] != 224 or input_shape[1] != 224:
        raise ValueError('Input shave received was %dx%dx%d, but it should be 224x224x%d' %
                         (input_shape[0], input_shape[1], input_shape[2], input_shape[2]))

    # ==== Defines model ====
    img_input = layers.Input(shape=input_shape)
    # Block 1
    x = layers.Conv2D(64, (3, 3),
                      activation='relu',
                      padding='same',
                      name='block1_conv1',
                      kernel_regularizer=regularizers.l2(l2_reg))(img_input)
    x = layers.Conv2D(64, (3, 3),
                      activation='relu',
                      padding='same',
                      name='block1_conv2',
                      kernel_regularizer=regularizers.l2(l2_reg))(x)
    x = layers.MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool')(x)

    # Block 2
    x = layers.Conv2D(128, (3, 3),
                      activation='relu',
                      padding='same',
                      name='block2_conv1',
                      kernel_regularizer=regularizers.l2(l2_reg))(x)
    x = layers.Conv2D(128, (3, 3),
                      activation='relu',
                      padding='same',
                      name='block2_conv2',
                      kernel_regularizer=regularizers.l2(l2_reg))(x)
    x = layers.MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool')(x)

    # Block 3
    x = layers.Conv2D(256, (3, 3),
                      activation='relu',
                      padding='same',
                      name='block3_conv1',
                      kernel_regularizer=regularizers.l2(l2_reg))(x)
    x = layers.Conv2D(256, (3, 3),
                      activation='relu',
                      padding='same',
                      name='block3_conv2',
                      kernel_regularizer=regularizers.l2(l2_reg))(x)
    x = layers.Conv2D(256, (3, 3),
                      activation='relu',
                      padding='same',
                      name='block3_conv3',
                      kernel_regularizer=regularizers.l2(l2_reg))(x)
    x = layers.MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool')(x)

    # Block 4
    x = layers.Conv2D(512, (3, 3),
                      activation='relu',
                      padding='same',
                      name='block4_conv1',
                      kernel_regularizer=regularizers.l2(l2_reg))(x)
    x = layers.Conv2D(512, (3, 3),
                      activation='relu',
                      padding='same',
                      name='block4_conv2',
                      kernel_regularizer=regularizers.l2(l2_reg))(x)
    x = layers.Conv2D(512, (3, 3),
                      activation='relu',
                      padding='same',
                      name='block4_conv3',
                      kernel_regularizer=regularizers.l2(l2_reg))(x)
    x = layers.MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool')(x)

    # Block 5
    x = layers.Conv2D(512, (3, 3),
                      activation='relu',
                      padding='same',
                      name='block5_conv1',
                      kernel_regularizer=regularizers.l2(l2_reg))(x)
    x = layers.Conv2D(512, (3, 3),
                      activation='relu',
                      padding='same',
                      name='block5_conv2',
                      kernel_regularizer=regularizers.l2(l2_reg))(x)
    x = layers.Conv2D(512, (3, 3),
                      activation='relu',
                      padding='same',
                      name='block5_conv3',
                      kernel_regularizer=regularizers.l2(l2_reg))(x)
    x = layers.MaxPooling2D((2, 2), strides=(2, 2), name='block5_pool')(x)

    if include_top:
        # Classification block
        x = layers.Flatten(name='flatten')(x)
        x = layers.Dense(4096, activation='relu', name='fc1', kernel_regularizer=regularizers.l2(l2_reg))(x)
        x = layers.Dropout(rate=dropout, name='dropout1')(x)
        x = layers.Dense(4096, activation='relu', name='fc2', kernel_regularizer=regularizers.l2(l2_reg))(x)
        x = layers.Dropout(rate=dropout, name='dropout2')(x)
        x = layers.Dense(classes, activation='softmax', name='predictions')(x)
    else:
        if pooling == 'avg':
            x = layers.GlobalAveragePooling2D()(x)
        elif pooling == 'max':
            x = layers.GlobalMaxPooling2D()(x)

    inputs = img_input
    # Create model.
    model = models.Model(inputs, x, name=name)

    # Load weights.
    if weights == 'imagenet':
        weights_path = utils.get_file(
            'vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5',
            keras_applications.vgg16.WEIGHTS_PATH_NO_TOP,
            cache_subdir='models',
            file_hash='6d6bbae143d832006294945121d1f1fc')
        model.load_weights(weights_path, by_name=True)
        if backend.backend() == 'theano':
            utils.convert_all_kernels_in_model(model)
    elif weights is not None:
        model.load_weights(weights, by_name=True)

    return model


if __name__ == '__main__':

    print('====== Spatial model ======')
    print()
    print()

    model = VGG16(
        'spatial',
        input_shape=(224, 224, 3),
        classes=101,
        dropout=0.9,
        l2_reg=1e-5,
        weights='imagenet',
        include_top=True
    )

    model.summary()

    # wait for key
    input()

    print('====== Temporal model ======')
    print()
    print()

    model = VGG16(
        'spatial',
        input_shape=(224, 224, 20),
        classes=101,
        dropout=0.9,
        l2_reg=1e-5,
        weights=None,
        include_top=True
    )

    model.summary()
