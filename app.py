import streamlit as st
import numpy as np
import librosa
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import GRU, Bidirectional

# 👇 Define the custom attention layer if used in training
class Attention(tf.keras.layers.Layer):
    def __init__(self):
        super(Attention, self).__init__()

    def call(self, inputs):
        query = inputs
        attention_weights = tf.nn.softmax(query, axis=1)
        context_vector = attention_weights * query
        return tf.reduce_sum(context_vector, axis=1)

# 👇 Load the trained model
model = load_model(
    "emotion_gru_attention_model.h5",
    custom_objects={"GRU": GRU, "Bidirectional": Bidirectional, "Attention": Attention},
    compile=False
)

# 👇 Load the label encoder
import pickle
with open("label_encoder.pkl", "rb") as f:
    label_encoder = pickle.load(f)

# 👇 Feature extraction from audio
def preprocess_audio(file, n_mfcc=40, max_pad_len=174):
    y, sr = librosa.load(file, sr=None)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    if mfcc.shape[1] < max_pad_len:
        pad_width = max_pad_len - mfcc.shape[1]
        mfcc = np.pad(mfcc, pad_width=((0, 0), (0, pad_width)), mode='constant')
    else:
        mfcc = mfcc[:, :max_pad_len]
    mfcc = np.transpose(mfcc[np.newaxis, ...], (0, 2, 1))
    return mfcc

# 👇 Streamlit UI
st.set_page_config(page_title="Speech Emotion Classifier")
st.title("🎙️ Speech Emotion Recognition")

uploaded_file = st.file_uploader("Upload a `.wav` file", type=["wav"])

if uploaded_file is not None:
    st.audio(uploaded_file, format="audio/wav")
    features = preprocess_audio(uploaded_file)
    prediction = model.predict(features)
    pred_label = label_encoder.inverse_transform([np.argmax(prediction)])
    st.success(f"🧠 Predicted Emotion: **{pred_label[0]}**")
