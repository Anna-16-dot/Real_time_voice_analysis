import warnings
warnings.filterwarnings("ignore")

import os
os.environ["SDL_AUDIODRIVER"] = "dummy"  # For SDL-based libraries
os.environ["PYGAME_HIDE_SUPPORT_PROMPT"] = "1"  # Optional: Hide pygame welcome message

import streamlit as st
import numpy as np
import torch
from transformers import AutoFeatureExtractor, AutoModelForAudioClassification
import matplotlib.pyplot as plt
from scipy.signal import butter, lfilter
import pyaudio
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Audio stream parameters
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000
CHUNK = 1024

# Streamlit page config
st.set_page_config(page_title="Real-Time Gender Detection", page_icon="🎤", layout="centered", initial_sidebar_state="collapsed")

# Custom CSS for theme styling
st.markdown("""
    <style>
        .stApp {
            background-color: #e6f7ff;
            color: #333333;
        }
        .stButton > button {
            background-color: #1976d2;
            color: #ffffff;
            border-radius: 8px;
            padding: 0.5rem 1rem;
            margin: 0.5rem;
        }
        .stButton > button:hover {
            background-color: #005bb5;
        }
        .stTitle, .stHeader, .stSubheader {
            color: #1976d2;
        }
    </style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_model():
    model_path = "alefiury/wav2vec2-large-xlsr-53-gender-recognition-librispeech"
    feature_extractor = AutoFeatureExtractor.from_pretrained(model_path)
    model = AutoModelForAudioClassification.from_pretrained(model_path)
    model.eval()
    return feature_extractor, model

placeholder = st.empty()
placeholder.text("Loading model...")
feature_extractor, model = load_model()
placeholder.text("Model loaded!")    

st.title("🎤 Real Time Fourier Transform Series with Gender Detection")
st.subheader("Experience the power of audio processing and gender recognition")
st.write("Click 'Start' to detect gender in real-time and view waveforms.")
placeholder.empty()

if 'listening' not in st.session_state:
    st.session_state['listening'] = False
if 'prediction' not in st.session_state:
    st.session_state['prediction'] = ""
if 'audio_data' not in st.session_state:
    st.session_state['audio_data'] = np.array([])

# Slider for low pass filter cutoff frequency
cutoff = st.slider(
    "Low Pass Filter Cutoff Frequency (Hz)", 
    min_value=500, max_value=8000, value=4000, step=100,
    help="Adjust the cutoff frequency for the low pass filter."
)

# Low Pass Filter Implementation
def low_pass_filter(data, cutoff=cutoff, fs=16000, order=5):
    nyquist = 0.5 * fs
    normal_cutoff = cutoff / nyquist
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    filtered_data = lfilter(b, a, data)
    return filtered_data


def plot_waveforms(raw_waveform, transformed_waveform, filtered_waveform):
    fig, axs = plt.subplots(3, 1, figsize=(10, 8), facecolor='#e6f7ff')
    axs[0].plot(raw_waveform, color='blue')
    axs[0].set_title("Raw Waveform", fontsize=18, color='#1976d2')
    axs[0].set_ylim([-1, 1])

    axs[1].plot(transformed_waveform, color='blue')
    axs[1].set_title("Transformed Waveform", fontsize=18, color='#1976d2')
    axs[1].set_ylim([-1, 1])

    axs[2].plot(filtered_waveform, color='blue')
    axs[2].set_title(f"Low Pass Filtered Waveform (Cutoff: {cutoff} Hz)", fontsize=18, color='#1976d2')
    axs[2].set_ylim([-1, 1])

    plt.tight_layout()
    st.pyplot(fig)


def start_listening():
    try:
        audio = pyaudio.PyAudio()
        stream = audio.open(format=FORMAT, channels=CHANNELS, rate=RATE, input=True, frames_per_buffer=CHUNK)
        st.session_state['listening'] = True
        st.session_state['audio_data'] = np.array([], dtype=np.float32)

        while st.session_state['listening']:
            raw_waveform = np.array([], dtype=np.float32)
            for _ in range(int(RATE / CHUNK * 1.5)):
                data = stream.read(CHUNK, exception_on_overflow=False)
                chunk_data = np.frombuffer(data, dtype=np.int16).astype(np.float32) / 32768.0
                raw_waveform = np.concatenate((raw_waveform, chunk_data))

            st.session_state['audio_data'] = np.concatenate((st.session_state['audio_data'], raw_waveform))

            if np.max(np.abs(raw_waveform)) > 0.02:
                inputs = feature_extractor(raw_waveform, sampling_rate=RATE, return_tensors="pt", padding=True)
                transformed_waveform = inputs['input_values'][0].numpy()
                filtered_waveform = low_pass_filter(raw_waveform, cutoff=cutoff)

                with torch.no_grad():
                    logits = model(**inputs).logits
                    predicted_ids = torch.argmax(logits, dim=-1)
                    predicted_label = model.config.id2label[predicted_ids.item()]
                    st.session_state['prediction'] = predicted_label
                    st.write(f"Detected Gender: {predicted_label}")

                plot_waveforms(raw_waveform, transformed_waveform, filtered_waveform)
            else:
                st.write("No significant sound detected. Try increasing the input volume or speaking closer to the microphone.")

    except Exception as e:
        logging.error(f"An error occurred: {e}")
        st.error(f"An error occurred: {e}")


def stop_listening():
    st.session_state['listening'] = False

col1, col2 = st.columns(2)
with col1: 
    if st.button("Start Listening 🎧"):
        start_listening()
with col2:
    if st.button("Stop Listening 🛑"):
        stop_listening()

if not st.session_state['listening'] and len(st.session_state['audio_data']) > 0:
    st.write("Listening stopped. ")
