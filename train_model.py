import os
import numpy as np
import librosa
import joblib
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score

DATASET_PATH = "dataset/"
MODEL_PATH = "model/emotion_model.pkl"

def extract_mfcc(file_path, n_mfcc=40):
    audio, sr = librosa.load(file_path, sr=22050, mono=True)
    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=n_mfcc)
    return np.mean(mfcc.T, axis=0)

X = []
y = []

for emotion in os.listdir(DATASET_PATH):
    emotion_folder = os.path.join(DATASET_PATH, emotion)
    if not os.path.isdir(emotion_folder):
        continue
    for file in os.listdir(emotion_folder):
        file_path = os.path.join(emotion_folder, file)
        mfcc_features = extract_mfcc(file_path)
        X.append(mfcc_features)
        y.append(emotion)

X = np.array(X)
y = np.array(y)

label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, random_state=42
)

model = SVC(kernel="rbf", probability=True)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))

joblib.dump((model, label_encoder), MODEL_PATH)
print("Model saved successfully.")
