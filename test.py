import os
import pickle
import numpy as np
from utils import convert_audio, convert_audios, extract_feature

def convert():
    audio_path = "/home/moe/Desktop/testing.wav"
    target_path = "/home/moe/Desktop/testing3.wav"
    if os.path.isdir(audio_path):
        if not os.path.isdir(target_path):
            os.makedirs(target_path)
            convert_audios(audio_path, target_path)
    elif os.path.isfile(audio_path) and audio_path.endswith(".wav"):
        if not target_path.endswith(".wav"):
            target_path += ".wav"
        convert_audio(audio_path, target_path)
    else:
        raise TypeError("The audio_path file you specified isn't appropriate for this operation")

if __name__ == "__main__":
    # convert()
    model = pickle.load(open("/home/moe/Desktop/model.model", "rb"))
    x_test =extract_feature("/home/moe/Desktop/testing3.wav")
    y_pred=model.predict(np.array([x_test]))
    print(y_pred)
