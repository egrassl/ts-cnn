import glob
import os
import pandas as pd

src = r'D:\Mestrado\databases\UCF101\raw'
# Dst path for config files
dst = r'D:\Mestrado\databases\UCF101'

classes = glob.glob(os.path.join(src, '*'))
classes.sort()

df_classes = pd.DataFrame(columns=['classes'])
df_samples = pd.DataFrame(columns=['sample', 'class'])

sample_counter = 0
for i in range(0, len(classes)):
    _, class_name = os.path.split(classes[i])

    videos = glob.glob(os.path.join(classes[i], '*.avi'))
    for video in videos:

        sample_counter += 1

        _, video_name = os.path.split(video)

        df_samples.loc[sample_counter] = [video_name, class_name]

    df_classes.loc[i+1] = [class_name]


# saves files
df_classes.to_csv(os.path.join(dst, 'classes_index.csv'))
df_samples.to_csv(os.path.join(dst, 'samples_annotations.csv'))
