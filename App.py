"""
Voice Analysis Pro - Professional Voice & Singing Analysis Tool
===============================================================
A comprehensive single-file Streamlit app for analyzing vocal performance.
"""

import streamlit as st
import numpy as np
import pandas as pd
import librosa
import soundfile as sf
from scipy import signal
import json, io, os, warnings
from pathlib import Path
from typing import Dict, Tuple

warnings.filterwarnings("ignore")

# Plotly
import plotly.graph_objects as go

try:
    import torch, torchcrepe
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch, torchcrepe = None, None

try:
    from streamlit_webrtc import webrtc_streamer
    WEBRTC_AVAILABLE = True
except ImportError:
    WEBRTC_AVAILABLE = False

SAMPLE_RATE = 22050
HOP_LENGTH = 512
MIN_FREQUENCY = 65
MAX_FREQUENCY = 2093
VOICING_THRESHOLD = 0.5

VOICE_TYPES = {
    'Bass': {'min': 40, 'max': 64, 'tessitura': (45, 57)},
    'Baritone': {'min': 45, 'max': 69, 'tessitura': (50, 62)},
    'Tenor': {'min': 48, 'max': 72, 'tessitura': (55, 67)},
    'Alto': {'min': 53, 'max': 77, 'tessitura': (60, 72)},
    'Mezzo-Soprano': {'min': 57, 'max': 81, 'tessitura': (64, 76)},
    'Soprano': {'min': 60, 'max': 84, 'tessitura': (67, 79)},
}

CATALOG_FILE = "song_catalog.csv"


# --- Utility functions ---

def hz_to_midi(hz):
    if hz <= 0: return 0
    return 12 * np.log2(hz / 440.0) + 69

def midi_to_hz(midi): return 440.0 * (2.0 ** ((midi - 69) / 12.0))

def midi_to_note_name(midi):
    if midi <= 0: return "N/A"
    names = ['C','C#','D','D#','E','F','F#','G','G#','A','A#','B']
    note = names[int(round(midi)) % 12]
    octave = int(round(midi)) // 12 - 1
    return f"{note}{octave}"

def smooth_pitch(pitch, conf, window=5, thr=0.5):
    p = pitch.copy()
    p[conf < thr] = 0
    return signal.medfilt(p, kernel_size=window)


# --- Main analysis (simplified identical to original) ---

def analyze_audio(y, sr):
    if sr != SAMPLE_RATE:
        y = librosa.resample(y, orig_sr=sr, target_sr=SAMPLE_RATE)
    if len(y.shape) > 1:
        y = librosa.to_mono(y)
    y = librosa.util.normalize(y)

    try:
        if TORCH_AVAILABLE:
            y16 = librosa.resample(y, orig_sr=SAMPLE_RATE, target_sr=16000)
            audio_tensor = torch.from_numpy(y16).float().unsqueeze(0)
            with torch.no_grad():
                pitch, conf = torchcrepe.predict(audio_tensor, 16000, 160,
                    fmin=MIN_FREQUENCY, fmax=MAX_FREQUENCY, model='tiny', device='cpu', return_periodicity=True)
            pitch, conf = pitch.squeeze().numpy(), conf.squeeze().numpy()
        else:
            pitch = librosa.yin(y, MIN_FREQUENCY, MAX_FREQUENCY, sr=SAMPLE_RATE)
            conf = librosa.feature.rms(y=y)[0]
    except Exception:
        pitch = librosa.yin(y, MIN_FREQUENCY, MAX_FREQUENCY, sr=SAMPLE_RATE)
        conf = librosa.feature.rms(y=y)[0]

    sm = smooth_pitch(pitch, conf)
    voiced = sm > 0
    vp = sm[voiced]
    if len(vp) == 0:
        return {'error': 'No voice detected'}

    midis = np.array([hz_to_midi(f) for f in vp])
    res = {
        'min_note': midi_to_note_name(np.min(midis)),
        'max_note': midi_to_note_name(np.max(midis)),
        'tessitura_low': midi_to_note_name(np.percentile(midis, 25)),
        'tessitura_high': midi_to_note_name(np.percentile(midis, 75)),
        'pitch_contour': {'times': np.arange(len(sm)) * HOP_LENGTH / SAMPLE_RATE, 'pitch_hz': sm.tolist()},
        'voice_type': 'Auto'
    }
    return res


# --- Streamlit UI ---

def main():
    st.set_page_config(page_title="Voice Analysis Pro", page_icon="🎤", layout="wide")
    st.title("🎤 Voice Analysis Pro")
    st.caption("Professional vocal analysis • 100% local")

    uploaded = st.file_uploader("Upload your voice recording", type=["wav", "mp3", "flac", "ogg"])
    if uploaded:
        y, sr = librosa.load(uploaded, sr=None)
        st.audio(uploaded)
        if st.button("Analyze Voice"):
            with st.spinner("Analyzing..."):
                r = analyze_audio(y, sr)
            if 'error' in r:
                st.error(r['error'])
            else:
                st.success("Analysis complete!")
                c1, c2, c3 = st.columns(3)
                c1.metric("Lowest Note", r['min_note'])
                c2.metric("Highest Note", r['max_note'])
                c3.metric("Tessitura", f"{r['tessitura_low']}–{r['tessitura_high']}")
                st.line_chart(pd.DataFrame({'Pitch (Hz)': r['pitch_contour']['pitch_hz']}))

if __name__ == "__main__":
    main()
