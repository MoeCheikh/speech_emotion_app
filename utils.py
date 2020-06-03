import librosa
import soundfile
import os, glob, pickle
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
import os

emotions={
  '01':'Neutral',
  '02':'Calm',
  '03':'Happy',
  '04':'Sad',
  '05':'Angry',
  '06':'Fearful',
  '07':'Disgusted',
  '08':'Surprised'
}

def load_data(sound_directory):
    x,y=[],[]
    for file in glob.glob(sound_directory):
        file_name=os.path.basename(file)
        emotion=emotions[file_name.split("-")[2]]
        feature=extract_feature(file)
        x.append(feature)
        y.append(emotion)

    return np.array(x),y

def extract_feature(file_name, mfcc=True, chroma=True, mel=True):
    with soundfile.SoundFile(file_name) as sound_file:
        X = sound_file.read(dtype="float32")
        sample_rate=sound_file.samplerate
        try:
            if chroma:
                stft=np.abs(librosa.stft(X))
            result=np.array([])
        except Exception as e:
            print(e)
            pass
        try:
            if mfcc:
                mfccs=np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=40).T, axis=0)
                result=np.hstack((result, mfccs))
        except Exception as e:
            print(e)
            pass
        try:
            if chroma:
                chroma=np.mean(librosa.feature.chroma_stft(S=stft, sr=sample_rate).T,axis=0)
                result=np.hstack((result, chroma))
        except Exception as e:
            print(e)
            pass
        try:
            if mel:
                mel=np.mean(librosa.feature.melspectrogram(X, sr=sample_rate).T,axis=0)
                result=np.hstack((result, mel))
        except Exception as e:
            print(e)
            pass
        return result

def convert(audio_path):
    file_split_list = audio_path.split("/")
    filename = file_split_list[-1].split(".")[0]
    new_filename = f"{filename}_converted.wav"
    file_split_list[-1] = new_filename
    seperator = "/"
    target_path = seperator.join(file_split_list)
    if not audio_path.endswith(".wav"):
        print("Invalid File: Must be in .wav format")
        return
    else:
        os.system(f"ffmpeg -i {audio_path} -ac 1 -ar 16000 {target_path}")
        os.remove(audio_path)
        return target_path
