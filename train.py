import argparse
import yaml
import os
from utils.file_management import create_dir, create_dirs


# Creates arg parsers
parser = argparse.ArgumentParser()

parser.add_argument('name', type=str, metavar='model_name', help='Name that will be used in this model')
parser.add_argument('type', type=str, metavar='stream_type', choices=['s', 't', 'st'], help='CNN stream type')
parser.add_argument('config', type=str, metavar='config_file', help='Config file path')
parser.add_argument('--gpu', type=int, metavar='gpu_id', help='Specifies a GPU to run the application')

args = parser.parse_args()

# Specifies a GPU to use
if args.gpu is not None:
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

# Gets parameters
with open(args.config) as c_file:
    configs = yaml.load(c_file, Loader=yaml.FullLoader)

    # keras imports
    from models.vgg16 import VGG16
    from models.stvgg16 import STVGG16
    from keras_extensions import custom_crops
    from keras import callbacks, optimizers, preprocessing
    from keras_extensions.FlowDataGenerator import MotionFlowDataGenerator
    from keras_extensions.SpatialDataGenerator import SpatialDataGenerator

    data_aug = preprocessing.image.ImageDataGenerator(
        zoom_range=.3,
        horizontal_flip=True,
        rotation_range=25,
        width_shift_range=.25,
        height_shift_range=.25,
        channel_shift_range=.35,
        brightness_range=[.5, 1.5]
    )

    # Loads cnn model
    if args.type == 's':
        input_shape = (224, 224, 3)

        model = VGG16(
            'spatial',
            input_shape=input_shape,
            classes=configs['n_classes'],
            dropout=configs['dropout'],
            l2_reg=configs['l2_reg'],
            weights=configs['weights'],
            train_only_last=configs['train_only_last'],
            include_top=True
        )

        train_set = SpatialDataGenerator(
            src=configs['dataset'],
            annotations=configs['annotations'],
            classes_info=configs['classes'],
            nb_frames=10,
            batch_size=configs['batch_size'],
            input_shape=(224, 224, 3),
            split='train',
            augmentation=data_aug
        )

        val_set = SpatialDataGenerator(
            src=configs['dataset'],
            annotations=configs['annotations'],
            classes_info=configs['classes'],
            nb_frames=10,
            batch_size=configs['batch_size'],
            input_shape=(224, 224, 3),
            split='val'
        )

    elif args.type == 't':
        input_shape = (224, 224, 20)

        model = VGG16(
            'temporal',
            input_shape=input_shape,
            classes=configs['n_classes'],
            dropout=configs['dropout'],
            l2_reg=configs['l2_reg'],
            weights=configs['weights'],
            train_only_last=configs['train_only_last'],
            include_top=True
        )

        train_set = MotionFlowDataGenerator(
            src=configs['dataset'],
            annotations=configs['annotations'],
            classes_info=configs['classes'],
            nb_frames=10,
            batch_size=configs['batch_size'],
            input_shape=(224, 224, 3),
            split='train',
            augmentation=data_aug
        )

        val_set = MotionFlowDataGenerator(
            src=configs['dataset'],
            annotations=configs['annotations'],
            classes_info=configs['classes'],
            nb_frames=10,
            batch_size=configs['batch_size'],
            input_shape=(224, 224, 3),
            split='val'
        )

    else:
        model = STVGG16(
            name='st_model',
            temporal_lenght=10,
            time=5,
            classes=configs['n_classes'],
            dropout=configs['dropout'],
            l2_reg=configs['l2_reg'],
            spatial_weights=configs['spatial_weights'],
            temporal_weights=configs['temporal_weights']
        )

model.summary()

input('Press enter to continue...')

create_dir(os.path.join(os.getcwd(), 'chkp'))

callbacks = [
    callbacks.ModelCheckpoint(os.path.join('chkp', '%s_best.h5' % args.name), save_best_only=True),
    callbacks.CSVLogger(filename=os.path.join('chkp', '%s.hist' % args.name), separator=','),
    callbacks.EarlyStopping(monitor='val_loss', patience=100),
    callbacks.ReduceLROnPlateau(monitor='val_loss', patience=50, verbose=1, min_lr=configs['learn_rate'] * 0.1)
]

optimizer = optimizers.SGD(lr=configs['learn_rate'], momentum=configs['momentum'])
model.compile(
    optimizer=optimizer,
    loss='categorical_crossentropy',
    #metrics=['accuracy', 'top_k_categorical_accuracy'],
    metrics=['accuracy']
)

# Train model
history = model.fit_generator(
    train_set,
    validation_data=val_set,
    verbose=1,
    epochs=configs['epochs'],
    callbacks=callbacks,
    use_multiprocessing=True,
    workers=5
)



