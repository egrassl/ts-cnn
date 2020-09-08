import glob
import os
import pandas as pd
from dataset_tools.ntu_home_definitions import NtuDefinitions


src = r'D:\Mestrado\databases\NTU-HOME\raw'
# Dst path for config files
dst = r'D:\Mestrado\databases\NTU-HOME'

df_classes = pd.DataFrame(columns=['classes', 'classes_name'])
df_samples = pd.DataFrame(columns=['sample', 'class'])

samples = glob.glob(os.path.join(src, '*.avi'))

class_counter = 0
for class_id in NtuDefinitions.classes.keys():
    class_counter += 1
    df_classes.loc[class_counter] = [class_id, NtuDefinitions.classes[class_id]]


for i in range(0, len(samples)):

    _, video_name = os.path.split(samples[i])

    class_id = video_name[16:20]

    if class_id not in NtuDefinitions.classes.keys():
        raise ValueError('Class id %s does not exist in NTU dataset' % class_id)

    df_samples.loc[i+1] = [video_name, class_id]

# saves files
df_classes.to_csv(os.path.join(dst, 'classes_index.csv'))
df_samples.to_csv(os.path.join(dst, 'samples_annotations.csv'))