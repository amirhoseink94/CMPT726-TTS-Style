#!/usr/bin/env python
# coding: utf-8
import os
import librosa
import soundfile
import numpy
import pandas as pd
from tqdm import tqdm
from skimage import io as skio


def scale_minmax(X, min=0.0, max=255.0):
    X_std = (X - X.min()) / (X.max() - X.min())
    X_scaled = X_std * (max - min) + min
    return X_scaled, X.max(), X.min()


def reverse_scale_minmax(X, old_max, old_min):
    return (X/255)*(old_max-old_min)+old_min


def audio2img(y, sr, hop_length, n_mels=128):
    # 1.melspectrogram
    mels = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels, n_fft=hop_length*2, hop_length=hop_length)
    # 2.log-transform.add small number to avoid log(0)
    mels = numpy.log(mels + 1e-9)
    # 3.standardize into 0 - 255 scale
    img, old_max, old_min = scale_minmax(mels, 0, 255)
    img = img.astype(numpy.uint8)
    return img, old_max, old_min


def img2audio(img,old_max,old_min,hop_length, sr=22050):
    # from 0 - 255 scale to orinally scale
    r1 = reverse_scale_minmax(img,old_max,old_min)
    # inverse log tranformation
    r2 = numpy.exp(r1)
    # melspectrogram to audio
    r3 = librosa.feature.inverse.mel_to_audio(r1, sr=sr, n_fft=hop_length*2, hop_length=hop_length)
    return r3


def cut_by_second(y,sr):
    n_pieces = len(y)//sr + 1
    result_list = []
    for i in range(1,n_pieces+1):
        start = (i-1)*sr
        if i!= n_pieces:
            end = i*sr
            y_piece = y[start:end]
            result_list.append([y_piece, i])
        else:
            y_piece = y[start:]
            if len(y_piece)<sr/2:
                pass
            else:
                pad_len = sr - len(y_piece)
                y_piece = numpy.append(y_piece, [0]*pad_len)
                result_list.append([y_piece, i])
    return result_list


def cut_without_padding(y,sr):
    n_pieces = len(y)//sr 
    if n_pieces == 0:
        return []
    result_list = []
    for i in range(1,n_pieces+1):
        start = (i-1)*sr
        end = i*sr
        y_piece = y[start:end]
        result_list.append([y_piece, i])
    return result_list


def audio_file_2_img_file(filename, mel_hop_length=512, sr=22050, input_dir='AudioWAV', output_dir='img_output', second_weight=1.48):
    input_path = os.path.join(input_dir, filename)
    y, sr = librosa.load(input_path, sr=sr)
    cutted_list = cut_without_padding(y, int(sr*second_weight))
    info_list = []
    if len(cutted_list)>0:
        for row in cutted_list:
            audio = row[0]
            piece_number = row[1]
            img, old_max, old_min = audio2img(audio, sr, mel_hop_length, n_mels=128)
            file_prefix = filename.split('.')[0]
            output_file = '%s__%s.png' % (file_prefix, piece_number)
            output_path = os.path.join(output_dir, output_file)
            skio.imsave(output_path, img)
            info_list.append([output_file, filename, piece_number, sr, mel_hop_length, old_max, old_min])
    return info_list


def img_file_2_audio_file(audio_file_name, info_df, img_dir='img_output', result_dir='reverse_audio'):
    sub_df = info_df[info_df['audio_file_name'] == audio_file_name].sort_values(by='audio_piece_number')
    audio = 0.0
    for i,row in sub_df.iterrows():
        img_path = os.path.join(img_dir, row['img_file_name'])
        img = skimage.io.imread(img_path)
        audio_piece = img2audio(img, row['old_max'], row['old_min'], row['mel_hop_length'])
        audio = numpy.concatenate((audio, audio_piece), axis=None)
    sr = sub_df['sample_rate'].values[0]
    audio_path = os.path.join(result_dir,audio_file_name)
    soundfile.write(audio_path, audio, sr)


def high_emo_filter(files):
    return [f for f in files if f[-6:-4] == 'HI']


if __name__ == '__main__':
    files = os.listdir('AudioWAV')
    files = high_emo_filter(files)
    info_list = []
    for i in tqdm(files):
        file_info_list = audio_file_2_img_file(i, output_dir='img_output_hi')
        info_list += file_info_list
    info_columns = ['img_file_name', 'audio_file_name', 'audio_piece_number', 'sample_rate', 'mel_hop_length', 'old_max', 'old_min']
    info_df = pd.DataFrame(info_list, columns=info_columns)
    info_df = pd.DataFrame(info_list,
                           columns=['img_file_name', 'audio_file_name', 'audio_piece_number', 'sample_rate', 'mel_hop_length', 'old_max', 'old_min'])
    info_df.to_csv('recovery_info_1s48_hi.csv')






