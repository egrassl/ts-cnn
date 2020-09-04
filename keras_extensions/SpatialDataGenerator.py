import keras
import os
from skimage.io import imread
from skimage.transform import resize
from keras_extensions import CustomDataGenerator
from models.vgg16 import VGG16


class SpatialDataGenerator(CustomDataGenerator.CustomDataGenerator):

    def get_data(self, sample_name, transform):
        s_path = os.path.join(self.src, 'spatial', sample_name + '.jpg')
        img = imread(s_path)
        img = keras.preprocessing.image.img_to_array(img)
        img = self.augmentation.apply_transform(img, transform) if transform is not None else img
        img = resize(img, (self.input_shape[0], self.input_shape[1]))
        return img * (1.0 / 255.0)


if __name__ == '__main__':

    train_datagen = keras.preprocessing.image.ImageDataGenerator(
        zoom_range=.3,
        horizontal_flip=True,
        rotation_range=25,
        width_shift_range=.25,
        height_shift_range=.25,
        channel_shift_range=.35,
        brightness_range=[.5, 1.5],
        rescale=1.0 / 255.0
    )

    train_set = SpatialDataGenerator(
        r'D:\Mestrado\databases\UCF101\test',
        r'D:\Mestrado\databases\UCF101\test\frame_info.csv',
        r'D:\Mestrado\databases\UCF101\classes_index.csv',
        10,
        16,
        (224, 224, 3),
        split='train',
        augmentation=train_datagen
    )

    val_set = SpatialDataGenerator(
        r'D:\Mestrado\databases\UCF101\test',
        r'D:\Mestrado\databases\UCF101\test\frame_info.csv',
        r'D:\Mestrado\databases\UCF101\classes_index.csv',
        10,
        16,
        (224, 224, 3),
        split='val',
        augmentation=train_datagen
    )

    model = VGG16(
        'spatial',
        input_shape=(224, 224, 3),
        classes=3,
        dropout=0.9,
        l2_reg=1e-5,
        weights='imagenet',
        include_top=True
    )

    model.summary()

    learning_rate = 1e-5
    print('Learning rate: %.10f' % learning_rate)
    model.compile(
        keras.optimizers.Adam(learning_rate=learning_rate),
        keras.losses.CategoricalCrossentropy(),
        metrics=['acc']
    )

    history = model.fit_generator(
        train_set,
        validation_data=val_set,
        verbose=1,
        epochs=200,
        use_multiprocessing=True,
        workers=8
    )
