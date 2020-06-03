from flask import Flask, render_template, request
from werkzeug.utils import secure_filename
from utils import extract_feature,convert
import pickle
import os
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
app = Flask(__name__)

app.config['UPLOAD_FOLDER'] = "./videos"

@app.route('/')
def home():
    return render_template('index.html',value="")


## Route for
@app.route('/results', methods = ['GET', 'POST'])
def results():
    if not os.path.isdir("./videos"):
        os.mkdir("videos")
    if request.method == 'POST':
        try:
          f = request.files['file']
          filename = secure_filename(f.filename)
          f.save(os.path.join(app.config["UPLOAD_FOLDER"], filename))
        except:
          return render_template('index.html', value="")

    wav_file = os.listdir("./videos")[0]
    wav_file = f"{os.getcwd()}/videos/{wav_file}"
    wav_file = convert(wav_file)
    model = pickle.load(open(f"{os.getcwd()}/model.model", "rb"))
    x_test =extract_feature(wav_file)
    y_pred=model.predict(np.array([x_test]))
    os.remove(wav_file)
    return render_template('index.html', value=y_pred[0])
    print(y_pred)
