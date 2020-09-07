"""
Adaptation of the VGG16 model for training with regularizations.
The original model can be found at:
https://github.com/keras-team/keras-applications/blob/master/keras_applications/vgg16.py
"""


from keras import layers, regularizers, models, utils, backend
import keras_applications
import os
import h5py
import numpy as np
import tensorflow.keras.backend as K
from keras.engine.saving import load_attributes_from_hdf5_group


def VGG16(name, input_shape, classes: int, dropout: float, l2_reg: float, weights, train_only_last=False, include_top=False, pooling=None):

    if not (weights in {'imagenet', None} or os.path.exists(weights)):
        raise ValueError('The `weights` argument should be either '
                         '`None` (random initialization), `imagenet` '
                         '(pre-training on ImageNet), '
                         'or the path to the weights file to be loaded.')

    # Returns error if wrong input shape is provided
    if input_shape[0] != 224 or input_shape[1] != 224:
        raise ValueError('Input shave received was %dx%dx%d, but it should be 224x224x%d' %
                         (input_shape[0], input_shape[1], input_shape[2], input_shape[2]))

    trainable = not train_only_last

    # ==== Defines model ====
    img_input = layers.Input(shape=input_shape)
    # Block 1
    x = layers.Conv2D(64, (3, 3),
                      activation='relu',
                      padding='same',
                      name='block1_conv1',
                      kernel_regularizer=regularizers.l2(l2_reg),
                      trainable=trainable)(img_input)
    x = layers.Conv2D(64, (3, 3),
                      activation='relu',
                      padding='same',
                      name='block1_conv2',
                      kernel_regularizer=regularizers.l2(l2_reg),
                      trainable=trainable)(x)
    x = layers.MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool',
                            trainable=trainable)(x)

    # Block 2
    x = layers.Conv2D(128, (3, 3),
                      activation='relu',
                      padding='same',
                      name='block2_conv1',
                      kernel_regularizer=regularizers.l2(l2_reg),
                      trainable=trainable)(x)
    x = layers.Conv2D(128, (3, 3),
                      activation='relu',
                      padding='same',
                      name='block2_conv2',
                      kernel_regularizer=regularizers.l2(l2_reg),
                      trainable=trainable)(x)
    x = layers.MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool')(x)

    # Block 3
    x = layers.Conv2D(256, (3, 3),
                      activation='relu',
                      padding='same',
                      name='block3_conv1',
                      kernel_regularizer=regularizers.l2(l2_reg),
                      trainable=trainable)(x)
    x = layers.Conv2D(256, (3, 3),
                      activation='relu',
                      padding='same',
                      name='block3_conv2',
                      kernel_regularizer=regularizers.l2(l2_reg),
                      trainable=trainable)(x)
    x = layers.Conv2D(256, (3, 3),
                      activation='relu',
                      padding='same',
                      name='block3_conv3',
                      kernel_regularizer=regularizers.l2(l2_reg),
                      trainable=trainable)(x)
    x = layers.MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool', trainable=trainable)(x)

    # Block 4
    x = layers.Conv2D(512, (3, 3),
                      activation='relu',
                      padding='same',
                      name='block4_conv1',
                      kernel_regularizer=regularizers.l2(l2_reg),
                      trainable=trainable)(x)
    x = layers.Conv2D(512, (3, 3),
                      activation='relu',
                      padding='same',
                      name='block4_conv2',
                      kernel_regularizer=regularizers.l2(l2_reg),
                      trainable=trainable)(x)
    x = layers.Conv2D(512, (3, 3),
                      activation='relu',
                      padding='same',
                      name='block4_conv3',
                      kernel_regularizer=regularizers.l2(l2_reg),
                      trainable=trainable)(x)
    x = layers.MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool', trainable=trainable)(x)

    # Block 5
    x = layers.Conv2D(512, (3, 3),
                      activation='relu',
                      padding='same',
                      name='block5_conv1',
                      kernel_regularizer=regularizers.l2(l2_reg),
                      trainable=trainable)(x)
    x = layers.Conv2D(512, (3, 3),
                      activation='relu',
                      padding='same',
                      name='block5_conv2',
                      kernel_regularizer=regularizers.l2(l2_reg),
                      trainable=trainable)(x)
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

        if input_shape[2] == 3 or input_shape[2] == 1:
            model.load_weights(weights_path, by_name=True)
            if backend.backend() == 'theano':
                utils.convert_all_kernels_in_model(model)
        else:
            channels = input_shape[2]
            weight_values_ = get_named_layer_weights_from_h5py(weights_path)
            symbolic_weights_ = get_symbolic_filtered_layer_weights_from_model(model)[:len(weight_values_)]

            weight_values_[0] = ("conv1_cross_modality",
                                 [cross_modality_init(kernel=weight_values_[0][1][0], in_channels=channels),
                                  weight_values_[0][1][1]
                                  # 0 = first layer , 1 = weight_value , 0 = kernel
                                  # VGG16 has no bias
                                  ]
                                 )

            load_layer_weights(weight_values=weight_values_, symbolic_weights=symbolic_weights_)

        # Change layers names
        for layer in model.layers:
            layer.name = layer.name + '_' + name

    elif weights is not None:
        # Change layers names
        for layer in model.layers:
            layer.name = layer.name + '_' + name

        model.load_weights(weights, by_name=True)

    return model


'''
The code for cross modality weights used was develop in: 
https://github.com/mohammed-elkomy/two-stream-action-recognition/blob/master/models/motion_models.py

This project reuses these functions to allow VGG16 to use pre-trained weights from imagenet in 
(224, 224, 20) channel input.
'''
def is_same_shape(shape1, shape2):
    """Checks if two structures[could be list or single value for example] have the same shape"""
    if len(shape1) != len(shape2):
        return False
    else:
        for i in range(len(shape1)):
            if shape1[i] != shape2[i]:
                return False

        return True


# This piece of code is inspired by keras source
def compare_layers_weights(first_model_layers, second_model_layers):
    """Compare layers weights: I use them to test the pre trained models are loaded correctly"""
    for i in range(len(first_model_layers)):
        weights1 = first_model_layers[i].get_weights()
        weights2 = second_model_layers[i].get_weights()
        if len(weights1) == len(weights2):
            if not all([is_same_shape(weights2[w].shape, weights1[w].shape) and np.allclose(weights2[w], weights1[w]) for w in range(len(weights1))]):
                print(first_model_layers[i].name, "!=", second_model_layers[i].name)
        else:
            print(first_model_layers[i].name, "!=", second_model_layers[i].name)


# This piece of code is inspired by keras source
def get_symbolic_filtered_layer_weights_from_model(model):
    """For the given model get the symbolic(tensors) weights"""
    symbolic_weights = []
    for layer in model.layers:
        if layer.weights:
            symbolic_weights.append(layer.weights)
    return symbolic_weights  # now you can load those weights with tensorflow feed


# This piece of code is inspired by keras source
def get_named_layer_weights_from_h5py(h5py_file):
    """decodes h5py for a given model downloaded by keras and gets layer weight name to value mapping"""
    with h5py.File(h5py_file) as h5py_stream:
        layer_names = load_attributes_from_hdf5_group(h5py_stream, 'layer_names')

        weights_values = []
        for name in layer_names:
            layer = h5py_stream[name]
            weight_names = load_attributes_from_hdf5_group(layer, 'weight_names')
            if weight_names:
                weight_values = [np.asarray(layer[weight_name]) for weight_name in weight_names]
                weights_values.append((name, weight_values))
    return weights_values


# This piece of code is inspired by keras source
def load_layer_weights(weight_values, symbolic_weights):
    """loads weight_values which is a list ot tuples from get_named_layer_weights_from_h5py()
        into symbolic_weights obtained from get_symbolic_filtered_layer_weights_from_model()
    """
    if len(weight_values) != len(symbolic_weights):  # they must have the same length of layers
        raise ValueError('number of weights aren\'t equal', len(weight_values), len(symbolic_weights))
    else:  # similar to keras source code :D .. load_weights_from_hdf5_group
        print("length of layers to load", len(weight_values))
        weight_value_tuples = []

        # load layer by layer weights
        for i in range(len(weight_values)):  # list(layers) i.e. list of lists(weights)
            assert len(symbolic_weights[i]) == len(weight_values[i][1])
            # symbolic_weights[i] : list of symbolic names for layer i
            # symbolic_weights[i] : list of weight ndarrays for layer i
            weight_value_tuples += zip(symbolic_weights[i], weight_values[i][1])  # both are lists with equal lengths (name,value) mapping

        K.batch_set_value(weight_value_tuples)  # loaded a batch to be efficient


def cross_modality_init(in_channels, kernel):
    """
        Takes a weight computed for RGB and produces a new wight to be used by motion streams which need about 20 channels !
        kernel is (x, y, 3, 64)
    """
    # if in_channels == 3:  # no reason for cross modality
    #   return kernel
    print("cross modality kernel", kernel.shape)
    avg_kernel = np.mean(kernel, axis=2)  # mean (x, y, 64)
    weight_init = np.expand_dims(avg_kernel, axis=2)  # mean (x, y, 1, 64)
    return np.tile(weight_init, (1, 1, in_channels, 1))  # mean (x, y, in_channels, 64)


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
        include_top=True,
        train_only_last=True
    )


    model.summary()

    print([layer.trainable for layer in model.layers])

    # wait for key
    input()

    print('====== Temporal model ======')
    print()
    print()

    model = VGG16(
        'temporal',
        input_shape=(224, 224, 20),
        classes=101,
        dropout=0.9,
        l2_reg=1e-5,
        weights='imagenet',
        include_top=True
    )

    model.summary()

    print([layer.trainable for layer in model.layers])
