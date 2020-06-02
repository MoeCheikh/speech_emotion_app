import librosa
import soundfile
import os, glob, pickle
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
import os

def load_data():
    x,y=[],[]
    for file in glob.glob("/home/moe/Desktop/projects/sound_freq/speech_emotion_app/speech_data/Actor_*/*.wav"):
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

def convert_audio(audio_path, target_path, remove=True):
    """This function sets the audio `audio_path` to:
        - 16000Hz Sampling rate
        - one audio channel ( mono )
            Params:
                audio_path (str): the path of audio wav file you want to convert
                target_path (str): target path to save your new converted wav file
                remove (bool): whether to remove the old file after converting
        Note that this function requires ffmpeg installed in your system."""

    os.system(f"ffmpeg -i {audio_path} -ac 1 -ar 16000 {target_path}")
    # os.system(f"ffmpeg -i {audio_path} -ac 1 {target_path}")
    if remove:
        os.remove(audio_path)


def convert_audios(path, target_path, remove=True):
    """Converts a path of wav files to:
        - 16000Hz Sampling rate
        - one audio channel ( mono )
        and then put them into a new folder called `target_path`
            Params:
                audio_path (str): the path of audio wav file you want to convert
                target_path (str): target path to save your new converted wav file
                remove (bool): whether to remove the old file after converting
        Note that this function requires ffmpeg installed in your system."""

    for dirpath, dirnames, filenames in os.walk(path):
        for dirname in dirnames:
            dirname = os.path.join(dirpath, dirname)
            target_dir = dirname.replace(path, target_path)
            if not os.path.isdir(target_dir):
                os.mkdir(target_dir)

    for dirpath, _, filenames in os.walk(path):
        for filename in filenames:
            file = os.path.join(dirpath, filename)
            if file.endswith(".wav"):
                # it is a wav file
                target_file = file.replace(path, target_path)
                convert_audio(file, target_file, remove=remove)
