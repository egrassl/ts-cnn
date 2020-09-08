import os


# 25 classes
class NtuDefinitions(object):

    classes = {
        'A001': 'drink_water',
        'A002': 'eat_meal',
        'A003': 'brush_teeth',
        'A008': 'sit_down',
        'A009': 'stand_up',
        'A011': 'reading',
        'A012': 'writing',
        'A027': 'jump_up',
        'A028': 'phone_call',
        'A032': 'taking_a_selfie',
        'A037': 'wipe_face',
        'A041': 'sneeze',
        'A043': 'falling_down',
        'A044': 'headache',
        'A045': 'chest_pain',
        'A046': 'back_pain',
        'A047': 'neck_pain',
        'A048': 'nausea',
        'A069': 'thumb_up',
        'A070': 'thumb_down',
        'A074': 'counting_money',
        'A085': 'apply_cream_on_face',
        'A103': 'yawn',
        'A104': 'stretch_oneself',
        'A105': 'blow_nose'
    }

    # 106 subjects total
    # 53 training subjects
    train_subjects = ['S001', 'S002', 'S004', 'S005', 'S008', 'S009', 'S013', 'S014', 'S015', 'S016', 'S017', 'S018',
                      'S019', 'S025', 'S027', 'S028', 'S031', 'S034', 'S035', 'S038', 'S045', 'S046', 'S047', 'S049',
                      'S050', 'S052', 'S053', 'S054', 'S055', 'S056', 'S057', 'S058', 'S059', 'S070', 'S074', 'S078',
                      'S080', 'S081', 'S082', 'S083', 'S084', 'S085', 'S086', 'S089', 'S091', 'S092', 'S093', 'S094',
                      'S095', 'S097', 'S098', 'S100', 'S103']

    train_cameras = ['C001']

    @staticmethod
    def get_cs_split(video_name):
        '''
        Returns if a video belongs to the train, validation os test dataset given its name by the cross-subject method
        :param video_name: video file name
        :return: 'train', 'val' or 'test'
        '''
        subject = video_name[0:4]

        if subject in NtuDefinitions.train_subjects:
            return 'train'
        else:
            return 'val'
    @staticmethod
    def get_cv_split(video_name):
        '''
        Returns if a video belongs to the train, validation os test dataset given its name by the cross-view method
        :param video_name: video file name
        :return: 'train', 'val' or 'test'
        '''
        camera = video_name[4:8]

        if camera in NtuDefinitions.train_cameras:
            return 'train'
        else:
            return 'val'


if __name__ == '__main__':
    split = NtuDefinitions.get_cs_split('S001C001P001R001A047_rgb')
    print(split)
