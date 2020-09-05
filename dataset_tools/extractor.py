import pandas as pd
import os
import cv2
import random
from utils import file_management as fm, video_processing as vp


class Extractor(object):

    def __init__(self, src, dst, annotations, nb_frames: int, chunks: int, split_func, spatial: bool, temporal: bool, verbose: bool):

        # Sets attributes
        self.src = src
        self.dst = dst
        self.nb_frames = nb_frames
        self.nb_range = int(nb_frames / 2)
        self.chunks = chunks
        self.spatial = spatial
        self.temporal = temporal
        self.verbose = verbose

        # Reads annotation info
        self.annotations = {}
        annotation_file = pd.read_csv(annotations)
        for i in range(0, len(annotation_file['sample'])):
            sample_name = annotation_file['sample'][i]
            class_name = annotation_file['class'][i]
            split = split_func(sample_name)

            self.annotations[sample_name] = {'class': class_name, 'split': split}

        # Setup CSV classes files
        self.video_data = pd.DataFrame(columns=['sample', 'class', 'split'])
        self.frame_data = pd.DataFrame(columns=['sample', 'class', 'split'])
        self.video_file = os.path.join(dst, 'video_info.csv')
        self.frame_file = os.path.join(dst, 'frame_info.csv')
        self.sample_counter = 0
        self.frame_counter = 0

        # Set data modalities directories and creates them
        self.path_spatial = os.path.join(dst, 'spatial')
        self.path_u = os.path.join(dst, 'temporal', 'u')
        self.path_v = os.path.join(dst, 'temporal', 'v')

        fm.create_dirs([
            self.path_spatial,
            os.path.join(dst, 'temporal'),
            self.path_u,
            self.path_v
        ], verbose=True)

    def __get_indexes(self, n_frames, offset=.2):
        '''
        Gets the T distributed indexes over the video frames. It returns less indexes if could not get self.chunks indexes

        :param n_frames: number of frames in the video
        :param offset: offset on the left of the video (to avoid the starting frames, for example)
        :return: a list of frame indexes and its quantity
        '''
        offset = int(n_frames * offset)
        n = n_frames - offset

        # Try to find T chunks, but a smaller chunk quantity will be extracted if video is too short
        i = 0
        indexes = [n]
        while indexes[-1] + self.nb_range >= n:
            step = float((n - 1) / (self.chunks + 1 - i))
            indexes = [round(step * i) for i in range(1, self.chunks + 1 - i)]
            i += 1

        if self.verbose:
            print('%d chunks will be extracted' % len(indexes))

        return [i + offset for i in indexes]

    def __extract_spatial(self, video, video_name, indexes):

        frame_counter = 0
        for i in indexes:
            frame_path = os.path.join(self.path_spatial, video_name[:-4] + '_%s.jpg' % str(frame_counter).zfill(3))

            if os.path.isfile(frame_path):
                print('File already exists, thus extraction was skipped: %s' % frame_path)

            else:
                cv2.imwrite(frame_path, vp.frame_resize(image=video[i], width=256))

                if self.verbose:
                    print('File created: %s' % frame_path)

            frame_counter += 1

    def __extract_temporal(self, video, video_name, indexes):

        frame_counter = 0
        for i in indexes:
            frame_name = video_name[:-4] + '_%s' % str(frame_counter).zfill(3)

            flow_counter = 0
            for j in range(i - self.nb_range, i + self.nb_range):

                u_frame_path = os.path.join(os.path.join(self.path_u, frame_name + '_u%s.jpg' % str(flow_counter).zfill(3)))
                v_frame_path = os.path.join(os.path.join(self.path_v, frame_name + '_v%s.jpg' % str(flow_counter).zfill(3)))

                # Calculates motion flow only if file does not exists
                if os.path.isfile(u_frame_path) and os.path.isfile(v_frame_path):
                    print('File already exists, thus extraction was skipped: %s' % u_frame_path)
                    print('File already exists, thus extraction was skipped: %s' % v_frame_path)

                else:
                    # Get motion flow
                    flow = vp.calculate_flow(vp.frame_resize(video[j]), vp.frame_resize(video[j + 1]), flow_type=2)

                    cv2.imwrite(u_frame_path, flow[:, :, 0])
                    cv2.imwrite(v_frame_path, flow[:, :, 1])

                    if self.verbose:
                        print('File created: %s' % u_frame_path)
                        print('File created: %s' % v_frame_path)

                flow_counter += 1
            frame_counter += 1

    def __process_sample(self, sample_path):

        # Solves for sample name
        sample_src, sample_name = os.path.split(sample_path)
        video = vp.read_video(src=sample_path)
        sample_class = self.annotations[sample_name]['class']
        split = self.annotations[sample_name]['split']

        indexes = self.__get_indexes(n_frames=len(video), offset=0)

        # Adds sample definitions to csv ['sample', 'class', 'split']
        self.video_data.loc[self.sample_counter] = [sample_name, sample_class, split]
        self.sample_counter += 1

        for i in range(0, self.chunks):
            self.frame_counter += 1
            self.frame_data.loc[self.frame_counter] = [
                sample_name[:-4] + '_%s' % str(i).zfill(3),
                sample_class,
                split
            ]

        # Extracts and saves spatial frames
        if self.spatial:
            self.__extract_spatial(video=video, video_name=sample_name, indexes=indexes)

        # Extracts and saves temporal frames
        if self.temporal:
            self.__extract_temporal(video=video, video_name=sample_name, indexes=indexes)

    def extract(self):
        '''
        Iterates though all samples of the src dataset and extract the dataset to dst
        '''

        fm.sample_iterator(path=self.src,
                           process_sample=self.__process_sample,
                           extension='.avi')

        # Writes dataset csv file
        self.video_data.to_csv(self.video_file)
        self.frame_data.to_csv(self.frame_file)

        print('\nSamples extraction was finished with success!!')


if __name__ == '__main__':

    from dataset_tools.ucf_definitions import UcfDefinitions as ucf

    extractor = Extractor(
        src=r'/home/coala/mestrado/datasets/UCF101/raw/',
        dst=r'/home/coala/mestrado/datasets/UCF101/split01/',
        annotations=r'/home/coala/mestrado/datasets/UCF101/samples_annotations.csv',
        nb_frames=10,
        chunks=5,
        split_func=ucf.get_split,
        spatial=True,
        temporal=True,
        verbose=True
    )

    extractor.extract()
