# import matplotlib as plt
# import numpy as np
# import os
# import librosa
# import random

# from fastapi import FastAPI
# app = FastAPI()

# @app.get("/ser")
# def ser():
    # # # Load pickle file

    # # # Load in audio file and trim blank space
    # # y, sr = librosa.load('/path/to/wav')

    # # yt, _ = librosa.effects.trim(y)

    # # # Converting the sound clips into a melspectogram with librosa
    # # # A mel spectrogram is a spectrogram where the frequencies are converted to the mel scale
    # # audio_spectogram = librosa.feature.melspectrogram(
    # #     y=yt, sr=sr, n_fft=1024, hop_length=100)

    # # # Convert a power spectrogram (amplitude squared) to decibel (dB) units with power_to_db
    # # audio_spectogram = librosa.power_to_db(audio_spectogram, ref=np.max)

    # # librosa.display.specshow(
    # #     audio_spectogram, y_axis='mel', fmax=20000, x_axis='time')

    # # p = os.path.join('/path/to/spectrogram/jpg')
    # # plt.savefig(p)

    # # # Get model prediction

    # # # Conver prediction to JSON

    # tempRet = ['ANGRY', 'CALM', 'DISGUST', 'FEARFUL', 'HAPPY', 'NEUTRAL', 'SAD', 'SURPRISED']
    # return random.choice(tempRet)

import matplotlib as plt
import numpy as np
import os
import librosa
import librosa.display

from fastai import *
from fastai.vision.all import *
from fastai.vision.data import ImageDataLoaders
from fastai.tabular.all import *
from fastai.text.all import *
from fastai.vision.widgets import *

from flask import Flask, render_template, request
from werkzeug import secure_filename

app = Flask('__name__')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/upload')
def upload():
    return render_template('upload.html')

@app.route('/postWav', methods=['POST', 'GET'])
def postWav():
    if request.method == 'POST':
        f = request.files['file']
        f.save(secure_filename(f.filename))
        return 'file uploaded successfully'

@app.route('/endpoint')
def endpoint():
    # Load pickle file
    model = load_learner('./speech02.pkl')

    # # Load in audio file and trim blank space
    y, sr = librosa.load('./test.wav')

    # yt,  = librosa.effects.trim(y)

    # # Converting the sound clips into a melspectogram with librosa
    # # A mel spectrogram is a spectrogram where the frequencies are converted to the mel scale
    audio_spectogram = librosa.feature.melspectrogram(
        y=y, sr=sr, n_fft=1024, hop_length=100)

    # # Convert a power spectrogram (amplitude squared) to decibel (dB) units with power_to_db
    audio_spectogram = librosa.power_to_db(audio_spectogram, ref=np.max)

    librosa.display.specshow(audio_spectogram, y_axis='mel', fmax=20000, xaxis='time')

    p = os.path.join('../out.jpg')
    plt.savefig(p)

    # # Get model prediction
    e, _, _ = model.predict("../out.jpg")

    return e.upper()

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5000)
