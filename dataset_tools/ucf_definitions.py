import os


class UcfDefinitions(object):

    ucf_splits = None

    @staticmethod
    def __parse_defs():
        # train = open(os.path.join(os.getcwd(), 'dataset_tools', 'ucftrainlist.txt'), 'r')
        train = open(os.path.join(os.getcwd(), 'ucftrainlist.txt'), 'r')
        ucf_splits = {}

        for line in train:
            video_name = line.split(' ')[0]
            _, video_name = os.path.split(video_name)
            video_name = video_name.strip()
            ucf_splits[video_name] = 'train'

        # val = open(os.path.join(os.getcwd(), 'dataset_tools', 'ucftestlist.txt'), 'r')
        val = open(os.path.join(os.getcwd(), 'ucftestlist.txt'), 'r')

        for line in val:
            video_name = line.split(' ')[0]
            _, video_name = os.path.split(video_name)
            video_name = video_name.strip()
            ucf_splits[video_name] = 'val'

        return ucf_splits

    @staticmethod
    def get_split(video_name):

        if UcfDefinitions.ucf_splits is None:
            UcfDefinitions.ucf_splits = UcfDefinitions.__parse_defs()

        return UcfDefinitions.ucf_splits[video_name]


if __name__ == '__main__':
    split = UcfDefinitions.get_split('v_ApplyEyeMakeup_g01_c01.avi')
    print(split)
