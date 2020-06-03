import librosa
import soundfile
import os, glob, pickle
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
import os

#ALL emotions in RAVDESS dataset
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

def load_data(sound_directory, emotions_to_observe):

    """
    This function will loop through every file inside the RAVDESS directory and will
    load the data in and extract featurse IFF the emotion of the file is within The
    emotions_to_observe list
    """
    x,y=[],[]
    for file in glob.glob(sound_directory):
        file_name=os.path.basename(file)
        emotion=emotions[file_name.split("-")[2]]
        if emotion not in emotions_to_observe:
            continue
        feature=extract_feature(file)
        x.append(feature)
        y.append(emotion)

    return np.array(x),y

def extract_feature(file_name):

    """
    This function will extract features from the wav file in terms of mfcc, chroma, and melspec,
    mfcc: Mel Frequency Cepstral Coefficient, represents the short-term power spectrum of a sound
    chroma: Pertains to the 12 different pitch classes
    mel: Mel Spectrogram Frequency
    """

    with soundfile.SoundFile(file_name) as sound_file:
        X = sound_file.read(dtype="float32")
        sample_rate=sound_file.samplerate
        try:
            stft=np.abs(librosa.stft(X))
        except Exception as e:
            print(e)
            pass
        result=np.array([])
        try:
            mfccs=np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=40).T, axis=0)
            result=np.hstack((result, mfccs))
        except Exception as e:
            print(e)
            pass
        try:
            chroma=np.mean(librosa.feature.chroma_stft(S=stft, sr=sample_rate).T,axis=0)
            result=np.hstack((result, chroma))
        except Exception as e:
            print(e)
            pass
        try:
            mel=np.mean(librosa.feature.melspectrogram(X, sr=sample_rate).T,axis=0)
            result=np.hstack((result, mel))
        except Exception as e:
            print(e)
            pass
        return result

def convert(audio_path):
    """
    This function will convert any .wav files into .wav files compatible
    with the model. -ac audio channels 1 (monochannel), -ar audio frequency 16000hz
    """
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
