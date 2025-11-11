"""
Voice Analysis Pro - Professional Voice & Singing Analysis Tool
===============================================================
A comprehensive single-file Streamlit app for analyzing vocal performance.

Features:
- Live microphone recording with WebRTC
- Audio file upload support
- Pitch detection and vocal range analysis
- Voice type classification
- Timbre analysis
- Local song identification
- Personalized song recommendations
- Interactive visualizations
- Privacy-focused (all processing local)

Author: Claude
Version: 1.0
Python: 3.10+
"""

import streamlit as st
import numpy as np
import pandas as pd
import librosa
import soundfile as sf
from scipy import signal, stats
from scipy.spatial.distance import cosine
import json
import io
import os
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

# Plotly for interactive charts
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

# Optional dependencies with graceful fallbacks
try:
    import torch
    import torchcrepe
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None
    torchcrepe = None

try:
    from streamlit_webrtc import webrtc_streamer, WebRtcMode, RTCConfiguration
    import av
    WEBRTC_AVAILABLE = True
except ImportError:
    WEBRTC_AVAILABLE = False

try:
    from pydub import AudioSegment
    PYDUB_AVAILABLE = True
except ImportError:
    PYDUB_AVAILABLE = False


# ============================================================================
# CONFIGURATION & CONSTANTS
# ============================================================================

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
    'Soprano': {'min': 60, 'max': 84, 'tessitura': (67, 79)}
}

CATALOG_FILE = "song_catalog.csv"


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def hz_to_midi(hz: float) -> float:
    if hz <= 0:
        return 0
    return 12 * np.log2(hz / 440.0) + 69


def midi_to_hz(midi: float) -> float:
    return 440.0 * (2.0 ** ((midi - 69) / 12.0))


def midi_to_note_name(midi: float) -> str:
    if midi <= 0:
        return "N/A"
    note_names = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
    note_num = int(round(midi))
    octave = (note_num // 12) - 1
    note = note_names[note_num % 12]
    return f"{note}{octave}"


def cents_from_midi(hz: float, ref_midi: float) -> float:
    if hz <= 0:
        return 0
    actual_midi = hz_to_midi(hz)
    return (actual_midi - ref_midi) * 100


def smooth_pitch(pitch: np.ndarray, confidence: np.ndarray, window_size: int = 5, conf_threshold: float = 0.5) -> np.ndarray:
    pitch_masked = pitch.copy()
    pitch_masked[confidence < conf_threshold] = 0
    smoothed = signal.medfilt(pitch_masked, kernel_size=window_size)
    return smoothed


def compute_spectral_features(y: np.ndarray, sr: int) -> Dict:
    D = np.abs(librosa.stft(y, hop_length=HOP_LENGTH))
    centroid = librosa.feature.spectral_centroid(y=y, sr=sr, hop_length=HOP_LENGTH)[0]
    rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr, hop_length=HOP_LENGTH)[0]
    freqs = librosa.fft_frequencies(sr=sr)
    low_band = (freqs < 300)
    mid_band = (freqs >= 300) & (freqs < 3000)
    high_band = (freqs >= 3000)
    low_energy = np.mean(D[low_band, :])
    mid_energy = np.mean(D[mid_band, :])
    high_energy = np.mean(D[high_band, :])
    total_energy = low_energy + mid_energy + high_energy
    return {
        'centroid_mean': np.mean(centroid),
        'centroid_std': np.std(centroid),
        'rolloff_mean': np.mean(rolloff),
        'low_energy_ratio': low_energy / total_energy if total_energy > 0 else 0,
        'mid_energy_ratio': mid_energy / total_energy if total_energy > 0 else 0,
        'high_energy_ratio': high_energy / total_energy if total_energy > 0 else 0
    }


def classify_timbre(spectral_features: Dict) -> str:
    low_ratio = spectral_features['low_energy_ratio']
    mid_ratio = spectral_features['mid_energy_ratio']
    high_ratio = spectral_features['high_energy_ratio']
    if low_ratio > 0.4:
        return "Bass-heavy"
    elif high_ratio > 0.35:
        return "Treble-bright"
    elif mid_ratio > 0.5:
        return "Mid-forward"
    else:
        return "Balanced"


# (… The rest of your code remains unchanged …)

# Jumping directly to the previously broken section ↓↓↓

                else:
                    for song in recommendations['fit']:
                        with st.expander(f"🎵 {song['title']} - {song['artist']}"):
                            col1, col2 = st.columns([2, 1])

                            with col1:
                                st.markdown(f"**Key:** {song['key']}")
                                st.markdown(f"**Range:** {song['range']}")
                                st.markdown(f"**Difficulty:** {song['difficulty'].title()}")
                                st.markdown(f"**Tags:** {song['tags']}")

                                st.markdown("**Why this song:**")
                                for reason in song['reasons']:
                                    st.markdown(f"• {reason}")

                            with col2:
                                difficulty_color = {
                                    'beginner': '🟢',
                                    'intermediate': '🟡',
                                    'advanced': '🔴'
                                }
                                st.markdown(f"### {difficulty_color.get(song['difficulty'], '⚪')} {song['difficulty'].upper()}")
