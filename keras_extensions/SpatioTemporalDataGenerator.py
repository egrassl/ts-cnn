from keras_extensions.CustomDataGenerator import CustomDataGenerator
from skimage.io import imread
from skimage.transform import resize
import keras
import numpy as np
import os


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
            frame_name = os.path.join(self.src, 'spatial', sample_name + '_%d' % str(i).zfill(3))

            spatial = self.__load_spatial(frame_name, transform)
            motion = self.__get_flow(frame_name, transform)

            spatial_frames.append(spatial)
            motion_flow.append(motion)

        return [spatial_frames, motion_flow]
