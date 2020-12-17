import os
import pandas as pd
from io import StringIO

audio_dir = 'AudioWAV'
file_names = os.listdir(audio_dir)
name_df = pd.DataFrame(file_names, columns=['audio_file_name'])
name_df['emotion'] = name_df['audio_file_name'].map(lambda x: x.split('_')[2])
emotions = name_df['emotion'].unique()

for e in emotions:
    name_df[e] = -1
    name_df.loc[name_df['emotion'] == e, e] = 1

info_df = pd.read_csv('recovery_info_1s48_hi.csv')
target_columns = ['img_file_name']+emotions.tolist()
label_df = info_df.merge(name_df, on='audio_file_name')[target_columns]
csv_buffer = StringIO(newline='\n')
label_df.to_csv(csv_buffer, index=False, header=None, sep=' ')

with open('list_attr_celeba.txt', ' w') as f:
    f.write(str(len(label_df)))
    f.write('\n')
    f.write(' '.join(emotions))
    f.write('\n')
    f.write(csv_buffer.getvalue().replace('\r\n', '\n'))