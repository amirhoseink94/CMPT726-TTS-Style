import os
import random
import shutil
import pandas as pd


def shuffle_files(input_dir, output_dir, prefix=''):
    files = os.listdir(input_dir)
    if not os.path.isdir(output_dir):
        os.mkdir(output_dir)
    numbers = list(range(1, len(files)+1))
    random.shuffle(numbers)
    match_list = []
    for i in range(0,len(files)):
        input_filename = files[i]
        name, file_extension = os.path.splitext(input_filename)
        output_filename = '%s_file_%s' % (prefix, numbers[i]) + file_extension
        src_path = os.path.join(input_dir, input_filename)
        dst_path = os.path.join(output_dir, output_filename)
        shutil.copyfile(src_path, dst_path)
        match_list.append([input_filename, output_filename])
    csv_path = os.path.join(output_dir, 'matching_info.csv')
    pd.DataFrame(match_list, columns=['name_before', 'name_after']).to_csv(csv_path,index=False)


if __name__ == '__main__':
    shuffle_files('happy_before_shuffle', 'happy_group', prefix='happy')
    shuffle_files('sad_before_shuffle', 'sad_group', prefix='sad')