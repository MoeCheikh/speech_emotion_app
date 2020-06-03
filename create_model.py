import librosa
import soundfile
import os, glob, pickle
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
from utils import load_data

#DataFlair - Emotions in the RAVDESS dataset

#DataFlair - Load the data and extract features for each sound file

x,y=load_data("/home/moe/Desktop/projects/sound_freq/speech_emotion_app/speech_data/Actor_*/*.wav")
model=MLPClassifier(alpha=0.01, batch_size=256, epsilon=1e-08, hidden_layer_sizes=(300,), learning_rate='adaptive', max_iter=500)
model.fit(x,y)
pickle.dump(model, open("model.model", "wb"))

# x_test =extract_feature("/home/moe/Desktop/testing.wav")
# y_pred=model.predict(np.array([x_test]))
