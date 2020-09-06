from keras.utils import Sequence, to_categorical
import pandas as pd
import numpy as np


class CustomDataGenerator(Sequence):

    def __init__(self,
                 src,
                 annotations,
                 classes_info,
                 nb_frames,
                 batch_size,
                 input_shape,
                 split='train',
                 augmentation=None):

        # Dataset variables
        self.src = src
        self.nb_frame = nb_frames
        self.batch_size = batch_size
        self.input_shape = input_shape
        self.split = split
        self.augmentation = augmentation

        # Process classes info csv file
        classes_file = pd.read_csv(classes_info)
        self.classes = {}
        for i in range(0, len(classes_file['classes'])):
            self.classes[classes_file['classes'][i]] = i

        # Process annotations in csv file
        ann_file = pd.read_csv(annotations)
        self.annotations = []
        # ['sample', 'class', 'split']
        for i in range(0, len(ann_file['sample'])):
            sample_name = ann_file['sample'][i]
            class_name = ann_file['class'][i]
            split = ann_file['split'][i]

            # Only adds file if it is for the same split
            if split == self.split:
                self.annotations.append({
                    'sample': sample_name,
                    'class': class_name,
                    'split': split
                })

        self.n_samples = len(self.annotations)

        print('Found %d classes and %d images for %s split' % (len(self.classes), self.n_samples, self.split))

        self.on_epoch_end()

    def __len__(self):
        '''
        Returns the total of batches for the training
        '''
        return self.n_samples // self.batch_size

    def on_epoch_end(self):
        np.random.shuffle(self.annotations)

    def __getitem__(self, index):
        '''
        Gets the batch for training
        '''

        # Gets samples indexes
        indexes = [i for i in range(index * self.batch_size, (index + 1) * self.batch_size)]
        result = [self.__get_sample(i) for i in indexes]

        # transform result in batch like ndarray and makes y hot encoded
        x = np.array([r[0] for r in result])
        y = to_categorical([r[1] for r in result], num_classes=len(self.classes))
        assert y.shape[0] == len(self.classes)

        return x, y

    def __get_sample(self, index):

        sample_name = self.annotations[index]['sample']
        class_name = self.annotations[index]['class']

        transform = self.augmentation.get_random_transform((self.input_shape[0], self.input_shape[1])) if self.augmentation is not None else None

        sample = self.get_data(sample_name, transform)

        return sample, self.classes[class_name]

    def get_data(self, sample_name, transform):

        raise Exception('The base class must implement __get_data method!!')


if __name__ == '__main__':
    dt = CustomDataGenerator(
        r'D:\Mestrado\databases\UCF101\test',
        r'',
        r'D:\Mestrado\databases\UCF101\classes_index.csv',
        10,
        4,
        (224, 224, 3),
        split='train'
    )