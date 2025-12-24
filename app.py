import streamlit as st
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import joblib
import time

MODEL_PATH = "model/emotion_model.pkl"

model, label_encoder = joblib.load(MODEL_PATH)

def extract_mfcc(file, n_mfcc=40):
    audio, sr = librosa.load(file, sr=22050, mono=True)
    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=n_mfcc)
    return np.mean(mfcc.T, axis=0), audio, sr

st.title("Real-Time Audio Emotion Recognition")

uploaded_file = st.file_uploader("Upload a WAV audio file", type=["wav"])

if uploaded_file is not None:
    start_time = time.time()

    features, audio, sr = extract_mfcc(uploaded_file)
    features = features.reshape(1, -1)

    prediction = model.predict(features)
    emotion = label_encoder.inverse_transform(prediction)[0]

    processing_time = (time.time() - start_time) * 1000

    st.subheader("Predicted Emotion")
    st.success(emotion)

    st.subheader("Audio Waveform")
    fig, ax = plt.subplots()
    librosa.display.waveshow(audio, sr=sr, ax=ax)
    st.pyplot(fig)

    st.subheader("Processing Time")
    st.write(f"{processing_time:.2f} ms")
