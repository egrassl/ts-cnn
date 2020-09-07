from models.stvgg16 import STVGG16
from keras_extensions.CustomDataGenerator import CustomDataGenerator
from skimage.io import imread
from skimage.transform import resize
import keras
import numpy as np
import os
from keras.utils import to_categorical



class SpatioTemporalDataGenerator(CustomDataGenerator):

    def __load_spatial(self, sample_name, transform):
        s_path = os.path.join(self.src, 'spatial', sample_name + '.jpg')
        img = imread(s_path)
        img = keras.preprocessing.image.img_to_array(img)
        img = self.augmentation.apply_transform(img, transform) if transform is not None else img
        img = resize(img, (self.input_shape[0], self.input_shape[1]))
        return img * (1.0 / 255.0)

    def __load_flow(self, img_path, transform):
        img = imread(img_path)
        img = keras.preprocessing.image.img_to_array(img)
        img = self.augmentation.apply_transform(img, transform) if transform is not None else img
        img = img - np.mean(img)
        img = resize(img, (self.input_shape[0], self.input_shape[1]))
        return img * (1.0/255.0)

    def __get_flow(self, sample_name, transform):
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

    def get_data(self, sample_name, transform):

        sample_name = sample_name[:-4]

        spatial_frames = []
        motion_flow = []

        for i in range(0, 5):
            frame_name = sample_name + '_%s' % str(i).zfill(3)

            spatial = self.__load_spatial(frame_name, transform)
            motion = self.__get_flow(frame_name, transform)

            spatial_frames.append(spatial)
            motion_flow.append(motion)

        return spatial_frames, motion_flow

    def __getitem__(self, index):
        '''
        Gets the batch for training
        '''

        # Gets samples indexes
        indexes = [i for i in range(index * self.batch_size, (index + 1) * self.batch_size)]

        spatial_result = []
        temporal_result = []
        labels = []

        for i in indexes:
            data, label = self.get_sample(i)

            labels.append(label)
            spatial_result.append(data[0])
            temporal_result.append(data[1])

        # transform result in batch like ndarray and makes y hot encoded
        x = [np.array(spatial_result), np.array(temporal_result)]
        y = to_categorical([l for l in labels], num_classes=len(self.classes))
        assert y.shape[1] == len(self.classes)

        return x, y


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

    train_set = SpatioTemporalDataGenerator(
        r'/home/coala/mestrado/datasets/UCF101/split01/',
        r'/home/coala/mestrado/datasets/UCF101/split01/video_info.csv',
        r'/home/coala/mestrado/datasets/UCF101/classes_index.csv',
        10,
        8,
        (224, 224, 20),
        split='train',
        augmentation=train_datagen
    )

    val_set = SpatioTemporalDataGenerator(
        r'/home/coala/mestrado/datasets/UCF101/split01/',
        r'/home/coala/mestrado/datasets/UCF101/split01/video_info.csv',
        r'/home/coala/mestrado/datasets/UCF101/classes_index.csv',
        10,
        8,
        (224, 224, 20),
        split='val',
        augmentation=train_datagen
    )

    model = STVGG16(
        name='st_model',
        temporal_lenght=10,
        time=5,
        classes=101,
        dropout=0.5,
        l2_reg=1e-5,
        #spatial_weights='imagenet',
        #temporal_weights='imagenet',
        spatial_weights=None,
        temporal_weights=None
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
        workers=4
    )
