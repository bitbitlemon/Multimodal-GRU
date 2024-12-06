import librosa
import numpy as np
import os
import pickle

# 定义提取MFCC特征并保存为pickle文件的函数
def extract_and_save_mfcc_features(file_dir, output_file, voice_len=32000, sample_rate=16000):
    # 初始化MFCC特征字典
    mfcc_dict = {}

    # 提取MFCC特征并添加到字典
    for persondir in os.listdir(file_dir):
        if persondir != '.' and persondir != '..':
            emotion_dir = os.path.join(file_dir, persondir)
            for ed in os.listdir(emotion_dir):
                if ed != '.' and ed != '..':
                    files_dir = os.path.join(emotion_dir, ed)
                    for fileName in os.listdir(files_dir):
                        if fileName.endswith('.wav'):
                            audio_path = os.path.join(files_dir, fileName)
                            print(f"Processing file: {audio_path}")
                            y, sr = librosa.load(audio_path, sr=sample_rate)
                            normalized_y = y[:voice_len] if len(y) > voice_len else np.pad(y, (0, voice_len - len(y)), 'constant')
                            mfcc_data = librosa.feature.mfcc(y=normalized_y, sr=sample_rate, n_mfcc=13)
                            mfcc_dict[fileName] = mfcc_data

    # 保存MFCC特征到pickle文件
    with open(output_file, 'wb') as f:
        pickle.dump(mfcc_dict, f)
    print(f"MFCC features have been saved successfully to {output_file} as a pickle file.")

# 使用示例
# 假设您的音频文件存储在'./CASIA data'目录下
file_dir = './CASIA data'
output_file = 'audio_mfcc_features.pkl'
extract_and_save_mfcc_features(file_dir, output_file)