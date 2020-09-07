import keras
import numpy as np
import os
from skimage.io import imread
from skimage.transform import resize
from keras_extensions import CustomDataGenerator
from models.vgg16 import VGG16


class MotionFlowDataGenerator(CustomDataGenerator.CustomDataGenerator):

    def __load_flow(self, img_path, transform):
        img = imread(img_path)
        img = keras.preprocessing.image.img_to_array(img)
        img = self.augmentation.apply_transform(img, transform) if transform is not None else img
        img = img - np.mean(img)
        img = resize(img, (self.input_shape[0], self.input_shape[1]))
        return img * (1.0/255.0)

    def get_data(self, sample_name, transform):
        # Create stacked flow image
        image = np.empty((224, 224, 20))
        channel_count = 0

        u_path = os.path.join(self.src, 'temporal', 'u')
        v_path = os.path.join(self.src, 'temporal', 'v')

        for i in range(0, 10):
            u_img = None
            v_img = None

            # Get horizontal and vertical frames
            u_img_path = os.path.join(u_path, sample_name + '_u%s.jpg' % str(i).zfill(3))
            u_img = self.__load_flow(u_img_path, transform)

            v_img_path = os.path.join(v_path, sample_name + '_v%s.jpg' % str(i).zfill(3))
            v_img = self.__load_flow(v_img_path, transform)

            # Stack frames
            image[:, :, channel_count] = u_img[:, :, 0]
            channel_count += 1
            image[:, :, channel_count] = v_img[:, :, 0]
            channel_count += 1

        return image


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

    train_set = MotionFlowDataGenerator(
        r'/home/coala/mestrado/datasets/UCF101/split01/',
        r'/home/coala/mestrado/datasets/UCF101/split01/frame_info.csv',
        r'/home/coala/mestrado/datasets/UCF101/classes_index.csv',
        10,
        16,
        (224, 224, 20),
        split='train',
        augmentation=train_datagen
    )

    val_set = MotionFlowDataGenerator(
        r'/home/coala/mestrado/datasets/UCF101/split01/',
        r'/home/coala/mestrado/datasets/UCF101/split01/frame_info.csv',
        r'/home/coala/mestrado/datasets/UCF101/classes_index.csv',
        10,
        16,
        (224, 224, 20),
        split='val',
        augmentation=train_datagen
    )

    model = VGG16(
        'temporal',
        input_shape=(224, 224, 20),
        classes=101,
        dropout=0.9,
        l2_reg=1e-5,
        weights='imagenet',
        include_top=True,
        train_only_last=False
    )

    model.summary()

    learning_rate = 1e-3
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
