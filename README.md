🎧 Real-Time Audio Emotion Recognition App

This project is a real-time audio emotion recognition system that predicts human emotions from speech audio using machine learning.
Users can upload an audio file, visualize the waveform, and instantly receive the predicted emotion through an interactive Streamlit web interface.

📌 Features

Upload speech audio files (.wav)

Visualize audio waveform

Extract MFCC audio features

Predict emotions such as happy, sad, angry, and neutral

Near real-time inference with low latency

Simple and interactive Streamlit UI

🛠 Tech Stack

Python

Librosa – audio signal processing

Scikit-learn – machine learning model (SVM)

Streamlit – web application deployment

Matplotlib – waveform visualization

📂 Project Structure
audio_emotion_recognition/
├── app.py
├── train_model.py
├── requirements.txt
├── model/
│   └── emotion_model.pkl
└── README.md

📊 Dataset

The model is trained using the RAVDESS (Ryerson Audio-Visual Database of Emotional Speech and Song) dataset.

Audio format: WAV

Emotions used: Happy, Sad, Angry, Neutral

Dataset source: Kaggle

The dataset is used only for training and is not included in this repository.

⚙️ Setup Instructions
1. Clone the repository
git clone https://github.com/your-username/audio_emotion_recognition.git
cd audio_emotion_recognition

2. Create and activate virtual environment (Windows)
python -m venv .venv
.\.venv\Scripts\activate

3. Install dependencies
python -m pip install -r requirements.txt

🧠 Train the Model (One Time)
python train_model.py


This will generate the trained model file:

model/emotion_model.pkl

🚀 Run the Application
python -m streamlit run app.py


The app will open in your browser at:

http://localhost:8501

🌐 Live Demo

🔗 Project URL: https://your-app-name.streamlit.app

(Replace with your actual deployment link)

📈 Real-Time Implementation

Although the model is trained offline, the system performs real-time emotion prediction by:

Processing uploaded audio instantly

Extracting features on the fly

Predicting emotion with minimal delay

This satisfies real-time deployment requirements.

⚠️ Limitations

Performance may drop for noisy audio

Emotion overlap can affect accuracy

Short audio clips may reduce prediction reliability

🔮 Future Improvements

Add live microphone recording

Use deep learning models (CNN / LSTM)

Improve noise robustness

Support more emotion categories

📜 License

This project is intended for educational and academic use.
