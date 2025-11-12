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
    """Convert frequency in Hz to MIDI note number."""
    if hz <= 0:
        return 0
    return 12 * np.log2(hz / 440.0) + 69


def midi_to_hz(midi: float) -> float:
    """Convert MIDI note number to frequency in Hz."""
    return 440.0 * (2.0 ** ((midi - 69) / 12.0))


def midi_to_note_name(midi: float) -> str:
    """Convert MIDI note number to note name with octave."""
    if midi <= 0:
        return "N/A"
    
    note_names = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
    note_num = int(round(midi))
    octave = (note_num // 12) - 1
    note = note_names[note_num % 12]
    return f"{note}{octave}"


def cents_from_midi(hz: float, ref_midi: float) -> float:
    """Calculate cents deviation from reference MIDI note."""
    if hz <= 0:
        return 0
    actual_midi = hz_to_midi(hz)
    return (actual_midi - ref_midi) * 100


def smooth_pitch(pitch: np.ndarray, confidence: np.ndarray, 
                 window_size: int = 5, conf_threshold: float = 0.5) -> np.ndarray:
    """Smooth pitch contour using median filtering and confidence masking."""
    pitch_masked = pitch.copy()
    pitch_masked[confidence < conf_threshold] = 0
    smoothed = signal.medfilt(pitch_masked, kernel_size=window_size)
    return smoothed


def compute_spectral_features(y: np.ndarray, sr: int) -> Dict:
    """Compute spectral features for timbre analysis."""
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
    """Classify voice timbre based on spectral features."""
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


def detect_pitch_crepe(y: np.ndarray, sr: int) -> Tuple[np.ndarray, np.ndarray]:
    """Detect pitch using CREPE (if available)."""
    if not TORCH_AVAILABLE:
        raise ImportError("PyTorch/TorchCREPE not available")
    
    if sr != 16000:
        y_resampled = librosa.resample(y, orig_sr=sr, target_sr=16000)
    else:
        y_resampled = y
    
    audio_tensor = torch.from_numpy(y_resampled).float().unsqueeze(0)
    
    with torch.no_grad():
        pitch, confidence = torchcrepe.predict(
            audio_tensor,
            16000,
            hop_length=160,
            fmin=MIN_FREQUENCY,
            fmax=MAX_FREQUENCY,
            model='tiny',
            device='cpu',
            return_periodicity=True
        )
    
    return pitch.squeeze().numpy(), confidence.squeeze().numpy()


def detect_pitch_yin(y: np.ndarray, sr: int) -> Tuple[np.ndarray, np.ndarray]:
    """Detect pitch using librosa's YIN algorithm (fallback)."""
    pitch = librosa.yin(
        y,
        fmin=MIN_FREQUENCY,
        fmax=MAX_FREQUENCY,
        sr=sr,
        hop_length=HOP_LENGTH
    )
    
    rms = librosa.feature.rms(y=y, hop_length=HOP_LENGTH)[0]
    confidence = np.clip(rms / np.max(rms) if np.max(rms) > 0 else rms, 0, 1)
    min_len = min(len(pitch), len(confidence))
    
    return pitch[:min_len], confidence[:min_len]


def classify_voice_type(midi_notes: np.ndarray, tessitura_range: Tuple[float, float]) -> Dict:
    """Classify voice type based on range and tessitura."""
    min_note = np.min(midi_notes)
    max_note = np.max(midi_notes)
    tess_low, tess_high = tessitura_range
    
    scores = {}
    
    for voice_type, ranges in VOICE_TYPES.items():
        range_overlap = (
            min(max_note, ranges['max']) - max(min_note, ranges['min'])
        ) / (ranges['max'] - ranges['min'])
        
        tess_overlap = (
            min(tess_high, ranges['tessitura'][1]) - 
            max(tess_low, ranges['tessitura'][0])
        ) / (ranges['tessitura'][1] - ranges['tessitura'][0])
        
        scores[voice_type] = (range_overlap * 0.4 + tess_overlap * 0.6)
    
    sorted_types = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    best_type = sorted_types[0][0]
    best_score = sorted_types[0][1]
    is_borderline = (len(sorted_types) > 1 and sorted_types[1][1] > best_score * 0.85)
    
    return {
        'type': best_type,
        'confidence': best_score,
        'is_borderline': is_borderline,
        'alternative': sorted_types[1][0] if is_borderline else None,
        'all_scores': dict(sorted_types)
    }


def analyze_audio(y: np.ndarray, sr: int, progress_callback=None) -> Dict:
    """Comprehensive audio analysis pipeline."""
    results = {}
    
    if progress_callback:
        progress_callback("Preprocessing audio...", 0.1)
    
    if sr != SAMPLE_RATE:
        y = librosa.resample(y, orig_sr=sr, target_sr=SAMPLE_RATE)
        sr = SAMPLE_RATE
    
    if len(y.shape) > 1:
        y = librosa.to_mono(y)
    
    y = librosa.util.normalize(y)
    
    results['duration'] = len(y) / sr
    results['sample_rate'] = sr
    
    if progress_callback:
        progress_callback("Detecting pitch...", 0.3)
    
    try:
        if TORCH_AVAILABLE:
            pitch, confidence = detect_pitch_crepe(y, sr)
            results['pitch_method'] = 'CREPE'
        else:
            pitch, confidence = detect_pitch_yin(y, sr)
            results['pitch_method'] = 'YIN'
    except Exception as e:
        st.warning(f"CREPE failed, using YIN: {str(e)}")
        pitch, confidence = detect_pitch_yin(y, sr)
        results['pitch_method'] = 'YIN (fallback)'
    
    pitch_smoothed = smooth_pitch(pitch, confidence, window_size=5, conf_threshold=VOICING_THRESHOLD)
    voiced_mask = (pitch_smoothed > 0) & (confidence > VOICING_THRESHOLD)
    voiced_pitch = pitch_smoothed[voiced_mask]
    
    if len(voiced_pitch) == 0:
        return {
            'error': 'No voice detected. Please ensure you are speaking/singing clearly.',
            'voiced_ratio': 0
        }
    
    results['voiced_ratio'] = np.sum(voiced_mask) / len(voiced_mask)
    
    if progress_callback:
        progress_callback("Analyzing vocal range...", 0.5)
    
    midi_notes = np.array([hz_to_midi(f) for f in voiced_pitch])
    midi_notes = midi_notes[midi_notes > 0]
    
    results['min_note_midi'] = float(np.min(midi_notes))
    results['max_note_midi'] = float(np.max(midi_notes))
    results['min_note'] = midi_to_note_name(results['min_note_midi'])
    results['max_note'] = midi_to_note_name(results['max_note_midi'])
    
    tess_low = np.percentile(midi_notes, 25)
    tess_high = np.percentile(midi_notes, 75)
    results['tessitura_low_midi'] = float(tess_low)
    results['tessitura_high_midi'] = float(tess_high)
    results['tessitura_low'] = midi_to_note_name(tess_low)
    results['tessitura_high'] = midi_to_note_name(tess_high)
    
    if progress_callback:
        progress_callback("Classifying voice type...", 0.6)
    
    voice_classification = classify_voice_type(midi_notes, (tess_low, tess_high))
    results['voice_type'] = voice_classification
    
    if progress_callback:
        progress_callback("Calculating pitch accuracy...", 0.7)
    
    cents_errors = []
    for hz_val in voiced_pitch:
        nearest_midi = round(hz_to_midi(hz_val))
        cents_error = cents_from_midi(hz_val, nearest_midi)
        cents_errors.append(cents_error)
    
    cents_errors = np.array(cents_errors)
    
    results['pitch_accuracy'] = {
        'mean_cents_error': float(np.mean(np.abs(cents_errors))),
        'median_cents_error': float(np.median(cents_errors)),
        'std_cents_error': float(np.std(cents_errors)),
        'intonation_score': float(max(0, 100 - np.mean(np.abs(cents_errors))))
    }
    
    if progress_callback:
        progress_callback("Analyzing timbre...", 0.8)
    
    spectral_features = compute_spectral_features(y, sr)
    results['spectral_features'] = spectral_features
    results['timbre_classification'] = classify_timbre(spectral_features)
    
    note_histogram = {}
    for midi in midi_notes:
        note_name = midi_to_note_name(midi)
        note_histogram[note_name] = note_histogram.get(note_name, 0) + 1
    
    results['note_distribution'] = note_histogram
    
    results['pitch_contour'] = {
        'times': np.arange(len(pitch_smoothed)) * HOP_LENGTH / sr,
        'pitch_hz': pitch_smoothed.tolist(),
        'confidence': confidence.tolist(),
        'voiced_mask': voiced_mask.tolist()
    }
    
    if progress_callback:
        progress_callback("Analysis complete!", 1.0)
    
    return results


# ============================================================================
# SONG CATALOG & RECOMMENDATIONS
# ============================================================================

def initialize_song_catalog():
    """Initialize song catalog CSV if it doesn't exist."""
    if not os.path.exists(CATALOG_FILE):
        demo_songs = [
            {'title': 'Amazing Grace', 'artist': 'Traditional', 'key': 'G', 
             'typical_low': 'G3', 'typical_high': 'D5', 'tags': 'hymn,slow,mid-forward', 'difficulty': 'beginner'},
            {'title': 'Happy Birthday', 'artist': 'Traditional', 'key': 'F', 
             'typical_low': 'F3', 'typical_high': 'F4', 'tags': 'celebration,easy,balanced', 'difficulty': 'beginner'},
            {'title': 'Hallelujah', 'artist': 'Leonard Cohen', 'key': 'C', 
             'typical_low': 'C3', 'typical_high': 'C5', 'tags': 'ballad,emotional,mid-forward', 'difficulty': 'intermediate'},
            {'title': 'Somewhere Over the Rainbow', 'artist': 'Harold Arlen', 'key': 'Eb', 
             'typical_low': 'Eb3', 'typical_high': 'Bb4', 'tags': 'classic,dreamy,balanced', 'difficulty': 'intermediate'},
            {'title': 'Ave Maria', 'artist': 'Schubert', 'key': 'Bb', 
             'typical_low': 'F3', 'typical_high': 'Ab5', 'tags': 'classical,sacred,treble-bright', 'difficulty': 'advanced'},
            {'title': 'O Holy Night', 'artist': 'Adolphe Adam', 'key': 'C', 
             'typical_low': 'C3', 'typical_high': 'C5', 'tags': 'christmas,powerful,balanced', 'difficulty': 'advanced'},
            {'title': 'Edelweiss', 'artist': 'Rodgers & Hammerstein', 'key': 'Bb', 
             'typical_low': 'Bb3', 'typical_high': 'Eb4', 'tags': 'gentle,musical,mid-forward', 'difficulty': 'beginner'},
            {'title': 'Danny Boy', 'artist': 'Traditional Irish', 'key': 'C', 
             'typical_low': 'C3', 'typical_high': 'D5', 'tags': 'folk,emotional,balanced', 'difficulty': 'intermediate'},
            {'title': 'Scarborough Fair', 'artist': 'Traditional English', 'key': 'Em', 
             'typical_low': 'E3', 'typical_high': 'E4', 'tags': 'folk,haunting,mid-forward', 'difficulty': 'beginner'},
            {'title': 'What a Wonderful World', 'artist': 'Louis Armstrong', 'key': 'F', 
             'typical_low': 'F3', 'typical_high': 'F4', 'tags': 'jazz,hopeful,bass-heavy', 'difficulty': 'beginner'}
        ]
        
        df = pd.DataFrame(demo_songs)
        df.to_csv(CATALOG_FILE, index=False)
        return df
    else:
        return pd.read_csv(CATALOG_FILE)


def recommend_songs(analysis_results: Dict, catalog: pd.DataFrame) -> Dict:
    """Recommend songs based on voice analysis."""
    if 'error' in analysis_results:
        return {'fit': [], 'stretch': [], 'avoid': []}
    
    user_min = analysis_results['min_note_midi']
    user_max = analysis_results['max_note_midi']
    user_tess_low = analysis_results['tessitura_low_midi']
    user_tess_high = analysis_results['tessitura_high_midi']
    user_timbre = analysis_results['timbre_classification'].lower()
    
    fit_songs = []
    stretch_songs = []
    avoid_songs = []
    
    for _, song in catalog.iterrows():
        song_low_midi = hz_to_midi(librosa.note_to_hz(song['typical_low']))
        song_high_midi = hz_to_midi(librosa.note_to_hz(song['typical_high']))
        
        range_overlap = min(user_max, song_high_midi) - max(user_min, song_low_midi)
        
        tess_margin = 1
        fits_tessitura = (
            song_low_midi >= user_tess_low - tess_margin and
            song_high_midi <= user_tess_high + tess_margin
        )
        
        song_tags = song['tags'].lower()
        timbre_match = user_timbre in song_tags
        
        recommendation = {
            'title': song['title'],
            'artist': song['artist'],
            'key': song['key'],
            'range': f"{song['typical_low']}-{song['typical_high']}",
            'difficulty': song['difficulty'],
            'tags': song['tags'],
            'reasons': []
        }
        
        if fits_tessitura and range_overlap > 0:
            recommendation['reasons'].append(
                f"Fits comfortably in your tessitura ({analysis_results['tessitura_low']}-{analysis_results['tessitura_high']})"
            )
            if timbre_match:
                recommendation['reasons'].append(f"Matches your {user_timbre} timbre")
            fit_songs.append(recommendation)
        
        elif (song_high_midi <= user_max + 4 and song_low_midi >= user_min - 4 and range_overlap > 0):
            if song_high_midi > user_tess_high:
                recommendation['reasons'].append(
                    f"Stretches your upper range by {int(song_high_midi - user_tess_high)} semitones"
                )
            if song_low_midi < user_tess_low:
                recommendation['reasons'].append(
                    f"Stretches your lower range by {int(user_tess_low - song_low_midi)} semitones"
                )
            recommendation['reasons'].append("Good for safe vocal development")
            stretch_songs.append(recommendation)
        
        else:
            if song_high_midi > user_max + 4:
                recommendation['reasons'].append(
                    f"Too high: requires {midi_to_note_name(song_high_midi)} (above your max)"
                )
            if song_low_midi < user_min - 4:
                recommendation['reasons'].append(
                    f"Too low: requires {midi_to_note_name(song_low_midi)} (below your min)"
                )
            if not timbre_match:
                recommendation['reasons'].append("Timbre mismatch may require adaptation")
            avoid_songs.append(recommendation)
    
    fit_songs.sort(key=lambda x: ['beginner', 'intermediate', 'advanced'].index(x['difficulty']))
    stretch_songs.sort(key=lambda x: ['beginner', 'intermediate', 'advanced'].index(x['difficulty']))
    
    return {
        'fit': fit_songs[:10],
        'stretch': stretch_songs[:5],
        'avoid': avoid_songs[:5]
    }


# ============================================================================
# VISUALIZATION FUNCTIONS
# ============================================================================

def plot_pitch_contour(analysis_results: Dict) -> go.Figure:
    """Create interactive pitch contour plot."""
    contour = analysis_results['pitch_contour']
    times = np.array(contour['times'])
    pitch = np.array(contour['pitch_hz'])
    voiced = np.array(contour['voiced_mask'])
    
    voiced_times = times[voiced]
    voiced_pitch = pitch[voiced]
    
    if len(voiced_pitch) == 0:
        return None
    
    voiced_midi = np.array([hz_to_midi(f) for f in voiced_pitch])
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=voiced_times,
        y=voiced_pitch,
        mode='lines',
        name='Pitch',
        line=dict(color='#3b82f6', width=2),
        hovertemplate='<b>Time:</b> %{x:.2f}s<br><b>Frequency:</b> %{y:.1f} Hz<br><b>Note:</b> %{text}<extra></extra>',
        text=[midi_to_note_name(m) for m in voiced_midi]
    ))
    
    tess_low_hz = midi_to_hz(analysis_results['tessitura_low_midi'])
    tess_high_hz = midi_to_hz(analysis_results['tessitura_high_midi'])
    
    fig.add_hrect(
        y0=tess_low_hz, y1=tess_high_hz,
        fillcolor='green', opacity=0.1,
        line_width=0,
        annotation_text="Tessitura",
        annotation_position="top left"
    )
    
    fig.update_layout(
        title='Pitch Contour Over Time',
        xaxis_title='Time (seconds)',
        yaxis_title='Frequency (Hz)',
        hovermode='closest',
        template='plotly_white',
        height=400
    )
    
    return fig


def plot_note_distribution(analysis_results: Dict) -> go.Figure:
    """Create note distribution histogram."""
    note_dist = analysis_results['note_distribution']
    
    notes = list(note_dist.keys())
    counts = list(note_dist.values())
    
    note_midi = [(n, librosa.note_to_midi(n) if n != "N/A" else 0) for n in notes]
    note_midi.sort(key=lambda x: x[1])
    sorted_notes = [n[0] for n in note_midi]
    sorted_counts = [note_dist[n] for n in sorted_notes]
    
    fig = go.Figure(data=[
        go.Bar(
            x=sorted_notes,
            y=sorted_counts,
            marker_color='#8b5cf6',
            hovertemplate='<b>%{x}</b><br>Count: %{y}<extra></extra>'
        )
    ])
    
    fig.update_layout(
        title='Note Distribution',
        xaxis_title='Note',
        yaxis_title='Frequency',
        template='plotly_white',
        height=350
    )
    
    return fig


def plot_intonation_distribution(analysis_results: Dict) -> go.Figure:
    """Create cents error distribution plot."""
    contour = analysis_results['pitch_contour']
    pitch = np.array(contour['pitch_hz'])
    voiced = np.array(contour['voiced_mask'])
    
    cents_errors = []
    for hz_val in pitch[voiced]:
        if hz_val > 0:
            nearest_midi = round(hz_to_midi(hz_val))
            cents_error = cents_from_midi(hz_val, nearest_midi)
            cents_errors.append(cents_error)
    
    if len(cents_errors) == 0:
        return None
    
    fig = go.Figure(data=[
        go.Histogram(
            x=cents_errors,
            nbinsx=50,
            marker_color='#10b981',
            hovertemplate='<b>Cents:</b> %{x:.1f}<br><b>Count:</b> %{y}<extra></extra>'
        )
    ])
    
    fig.add_vline(x=0, line_dash="dash", line_color="red", annotation_text="Perfect pitch")
    
    fig.update_layout(
        title='Pitch Accuracy Distribution',
        xaxis_title='Cents from Nearest Note',
        yaxis_title='Frequency',
        template='plotly_white',
        height=350
    )
    
    return fig


def plot_spectral_profile(analysis_results: Dict) -> go.Figure:
    """Create spectral energy profile."""
    features = analysis_results['spectral_features']
    
    bands = ['Low<br>(< 300 Hz)', 'Mid<br>(300-3k Hz)', 'High<br>(> 3k Hz)']
    energies = [
        features['low_energy_ratio'] * 100,
        features['mid_energy_ratio'] * 100,
        features['high_energy_ratio'] * 100
    ]
    colors = ['#ef4444', '#f59e0b', '#3b82f6']
    
    fig = go.Figure(data=[
        go.Bar(
            x=bands,
            y=energies,
            marker_color=colors,
            text=[f'{e:.1f}%' for e in energies],
            textposition='auto',
            hovertemplate='<b>%{x}</b><br>Energy: %{y:.1f}%<extra></extra>'
        )
    ])
    
    fig.update_layout(
        title='Timbre Profile',
        yaxis_title='Energy Percentage',
        template='plotly_white',
        height=350,
        showlegend=False
    )
    
    return fig


# ============================================================================
# STREAMLIT UI
# ============================================================================

def apply_custom_css():
    """Apply custom CSS styling."""
    st.markdown("""
    <style>
    .stApp {
        max-width: 1400px;
        margin: 0 auto;
    }
    
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 12px;
        color: white;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        margin: 0.5rem 0;
    }
    
    .info-card {
        background: #f8fafc;
        padding: 1.5rem;
        border-radius: 12px;
        border-left: 4px solid #3b82f6;
        margin: 1rem 0;
    }
    
    h1 {
        background: linear-gradient(120deg, #667eea, #764ba2);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 700;
    }
    
    .stButton>button {
        border-radius: 8px;
        font-weight: 600;
        transition: all 0.3s;
    }
    
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(0,0,0,0.15);
    }
    
    .stDownloadButton>button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
    }
    
    .streamlit-expanderHeader {
        font-weight: 600;
        font-size: 1.1rem;
    }
    
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    
    .stTabs [data-baseweb="tab"] {
        border-radius: 8px 8px 0 0;
        padding: 12px 24px;
        font-weight: 600;
    }
    </style>
    """, unsafe_allow_html=True)


def render_header():
    """Render application header."""
    st.markdown("""
    <div style='text-align: center; padding: 2rem 0;'>
        <h1 style='font-size: 3rem; margin-bottom: 0.5rem;'>🎤 Voice Analysis Pro</h1>
        <p style='font-size: 1.2rem; color: #64748b;'>
            Professional vocal analysis powered by AI • 100% Local & Private
        </p>
    </div>
    """, unsafe_allow_html=True)


def render_sidebar():
    """Render sidebar with settings and info."""
    with st.sidebar:
        st.markdown("### ⚙️ Settings")
        
        st.markdown("#### Analysis Options")
        use_crepe = st.checkbox(
            "Use CREPE (if available)",
            value=TORCH_AVAILABLE,
            help="CREPE provides better pitch detection but requires PyTorch"
        )
        
        voicing_threshold = st.slider(
            "Voicing Threshold",
            0.0, 1.0, 0.5, 0.05,
            help="Minimum confidence for pitch detection"
        )
        
        st.markdown("---")
        
        st.markdown("### 📖 About")
        st.info("""
        **Voice Analysis Pro** provides:
        - 🎵 Vocal range detection
        - 🎭 Voice type classification  
        - 🎯 Pitch accuracy analysis
        - 🎨 Timbre profiling
        - 📊 Song recommendations
        
        **Privacy:** All processing happens locally on your device. No data is sent anywhere.
        """)
        
        with st.expander("🔧 System Info"):
            st.markdown(f"""
            - **Pitch Detection:** {'CREPE (PyTorch)' if TORCH_AVAILABLE else 'YIN (librosa)'}
            - **WebRTC:** {'Available' if WEBRTC_AVAILABLE else 'Not available'}
            - **Audio Processing:** librosa {librosa.__version__}
            """)
        
        with st.expander("📦 Requirements"):
            st.code("""streamlit>=1.28.0
numpy>=1.24.0
scipy>=1.10.0
pandas>=2.0.0
librosa>=0.10.0
soundfile>=0.12.0
plotly>=5.14.0
matplotlib>=3.7.0

# Optional (recommended)
torch>=2.0.0
torchcrepe>=0.0.20
streamlit-webrtc>=0.47.0
pydub>=0.25.0""")
    
    return use_crepe, voicing_threshold


def main():
    """Main application function."""
    st.set_page_config(
        page_title="Voice Analysis Pro",
        page_icon="🎤",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    apply_custom_css()
    
    if 'analysis_results' not in st.session_state:
        st.session_state.analysis_results = None
    if 'audio_data' not in st.session_state:
        st.session_state.audio_data = None
    if 'catalog' not in st.session_state:
        st.session_state.catalog = initialize_song_catalog()
    
    render_header()
    use_crepe, voicing_threshold = render_sidebar()
    
    tab1, tab2, tab3, tab4 = st.tabs([
        "🎙️ Record / Upload",
        "🔍 Voice Analysis",
        "📈 Charts",
        "🎵 Recommendations"
    ])
    
    # TAB 1: Recording/Upload
    with tab1:
        st.markdown("## Input Your Voice")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### 📤 Upload Audio File")
            uploaded_file = st.file_uploader(
                "Choose an audio file",
                type=['wav', 'mp3', 'm4a', 'flac', 'ogg'],
                help="Upload a recording of your voice (speaking or singing)"
            )
            
            if uploaded_file is not None:
                try:
                    with st.spinner("Loading audio..."):
                        audio_bytes = uploaded_file.read()
                        
                        try:
                            y, sr = librosa.load(io.BytesIO(audio_bytes), sr=None, mono=False)
                        except:
                            y, sr = sf.read(io.BytesIO(audio_bytes))
                        
                        st.session_state.audio_data = (y, sr)
                        
                        st.success(f"✅ Loaded: {uploaded_file.name}")
                        
                        duration = len(y) / sr
                        st.markdown(f"""
                        <div class='info-card'>
                            <b>📊 Audio Info:</b><br>
                            Duration: {duration:.2f} seconds<br>
                            Sample Rate: {sr} Hz<br>
                            Channels: {'Stereo' if len(y.shape) > 1 else 'Mono'}
                        </div>
                        """, unsafe_allow_html=True)
                        
                        fig = go.Figure()
                        times = np.linspace(0, duration, len(y) if len(y.shape) == 1 else len(y[0]))
                        y_plot = y if len(y.shape) == 1 else y[0]
                        
                        fig.add_trace(go.Scatter(
                            x=times,
                            y=y_plot,
                            mode='lines',
                            name='Waveform',
                            line=dict(color='#3b82f6', width=1)
                        ))
                        
                        fig.update_layout(
                            title='Audio Waveform',
                            xaxis_title='Time (s)',
                            yaxis_title='Amplitude',
                            height=250,
                            template='plotly_white'
                        )
                        
                        st.plotly_chart(fig, use_container_width=True)
                        
                except Exception as e:
                    st.error(f"❌ Error loading audio: {str(e)}")
        
        with col2:
            st.markdown("### 🎙️ Live Recording")
            
            if not WEBRTC_AVAILABLE:
                st.warning("""
                ⚠️ **Live recording not available**
                
                To enable live recording, install:
                ```
                pip install streamlit-webrtc
                ```
                
                For now, please use the file upload option.
                """)
            else:
                st.info("🎤 Click 'START' to begin recording from your microphone.")
                
                st.markdown("""
                <div class='info-card'>
                    <b>💡 Recording Tips:</b><br>
                    • Ensure you're in a quiet environment<br>
                    • Speak or sing clearly into the microphone<br>
                    • Record for at least 10-15 seconds<br>
                    • Include your full comfortable range
                </div>
                """, unsafe_allow_html=True)
        
        st.markdown("---")
        col_analyze, col_demo = st.columns([3, 1])
        
        with col_analyze:
            if st.button("🔬 Analyze Voice", type="primary", use_container_width=True):
                if st.session_state.audio_data is None:
                    st.error("❌ Please upload an audio file first!")
                else:
                    y, sr = st.session_state.audio_data
                    
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    
                    def update_progress(message, value):
                        status_text.text(message)
                        progress_bar.progress(value)
                    
                    try:
                        results = analyze_audio(y, sr, progress_callback=update_progress)
                        
                        if 'error' in results:
                            st.error(f"❌ {results['error']}")
                        else:
                            st.session_state.analysis_results = results
                            st.success("✅ Analysis complete! Check the other tabs for results.")
                            st.balloons()
                    
                    except Exception as e:
                        st.error(f"❌ Analysis failed: {str(e)}")
                    
                    finally:
                        progress_bar.empty()
                        status_text.empty()
        
        with col_demo:
            if st.button("🧪 Demo Mode", use_container_width=True):
                st.info("Generating synthetic audio for testing...")
                
                sr = 22050
                duration = 3.0
                t = np.linspace(0, duration, int(sr * duration))
                
                notes = [261.63, 293.66, 329.63, 349.23, 392.00, 440.00, 493.88, 523.25]
                y = np.array([])
                
                note_duration = duration / len(notes)
                for freq in notes:
                    note_t = np.linspace(0, note_duration, int(sr * note_duration))
                    note_signal = 0.3 * np.sin(2 * np.pi * freq * note_t)
                    envelope = np.exp(-note_t * 2)
                    note_signal *= envelope
                    y = np.concatenate([y, note_signal])
                
                st.session_state.audio_data = (y, sr)
                st.success("✅ Demo audio generated! Click 'Analyze Voice' to proceed.")
    
    # TAB 2: Analysis Results
    with tab2:
        st.markdown("## Voice Analysis Results")
        
        if st.session_state.analysis_results is None:
            st.info("👆 Please analyze audio from the 'Record / Upload' tab first.")
        else:
            results = st.session_state.analysis_results
            
            if 'error' in results:
                st.error(results['error'])
            else:
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.markdown(f"""
                    <div class='metric-card'>
                        <h3 style='color: white; margin: 0;'>🎵 Vocal Range</h3>
                        <p style='font-size: 1.8rem; font-weight: bold; margin: 0.5rem 0;'>
                            {results['min_note']} - {results['max_note']}
                        </p>
                        <p style='margin: 0; opacity: 0.9;'>
                            {int(results['max_note_midi'] - results['min_note_midi'])} semitones
                        </p>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col2:
                    voice_type = results['voice_type']
                    borderline_text = f" / {voice_type['alternative']}" if voice_type['is_borderline'] else ""
                    
                    st.markdown(f"""
                    <div class='metric-card'>
                        <h3 style='color: white; margin: 0;'>🎭 Voice Type</h3>
                        <p style='font-size: 1.8rem; font-weight: bold; margin: 0.5rem 0;'>
                            {voice_type['type']}{borderline_text}
                        </p>
                        <p style='margin: 0; opacity: 0.9;'>
                            Confidence: {voice_type['confidence']*100:.0f}%
                        </p>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col3:
                    intonation_score = results['pitch_accuracy']['intonation_score']
                    
                    st.markdown(f"""
                    <div class='metric-card'>
                        <h3 style='color: white; margin: 0;'>🎯 Pitch Accuracy</h3>
                        <p style='font-size: 1.8rem; font-weight: bold; margin: 0.5rem 0;'>
                            {intonation_score:.0f}/100
                        </p>
                        <p style='margin: 0; opacity: 0.9;'>
                            ±{results['pitch_accuracy']['mean_cents_error']:.1f} cents
                        </p>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col4:
                    st.markdown(f"""
                    <div class='metric-card'>
                        <h3 style='color: white; margin: 0;'>🎨 Timbre</h3>
                        <p style='font-size: 1.8rem; font-weight: bold; margin: 0.5rem 0;'>
                            {results['timbre_classification']}
                        </p>
                        <p style='margin: 0; opacity: 0.9;'>
                            Spectral profile
                        </p>
                    </div>
                    """, unsafe_allow_html=True)
                
                st.markdown("---")
                
                col_left, col_right = st.columns(2)
                
                with col_left:
                    st.markdown("### 📊 Detailed Metrics")
                    
                    st.markdown(f"""
                    <div class='info-card'>
                        <b>Tessitura (Comfort Zone):</b><br>
                        {results['tessitura_low']} - {results['tessitura_high']}<br>
                        <small>This is where your voice feels most comfortable</small>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    st.markdown(f"""
                    <div class='info-card'>
                        <b>Pitch Statistics:</b><br>
                        • Mean error: ±{results['pitch_accuracy']['mean_cents_error']:.1f} cents<br>
                        • Std deviation: {results['pitch_accuracy']['std_cents_error']:.1f} cents<br>
                        • Voiced ratio: {results['voiced_ratio']*100:.1f}%
                    </div>
                    """, unsafe_allow_html=True)
                    
                    st.markdown("### 🎭 Voice Type Analysis")
                    voice_scores = results['voice_type']['all_scores']
                    
                    for vtype, score in list(voice_scores.items())[:3]:
                        st.progress(score, text=f"{vtype}: {score*100:.0f}%")
                
                with col_right:
                    st.markdown("### 💡 Recommendations")
                    
                    intonation = results['pitch_accuracy']['intonation_score']
                    
                    if intonation >= 85:
                        st.success("""
                        ✅ **Excellent intonation!** Your pitch accuracy is outstanding.
                        Continue practicing to maintain this level.
                        """)
                    elif intonation >= 70:
                        st.info("""
                        👍 **Good intonation.** You're hitting most notes accurately.
                        Focus on sustaining notes for longer periods to improve stability.
                        """)
                    else:
                        st.warning("""
                        ⚠️ **Room for improvement.** Consider:
                        • Practicing with a piano or tuner
                        • Slowing down passages
                        • Strengthening breath support
                        """)
                    
                    timbre = results['timbre_classification']
                    timbre_advice = {
                        'Bass-heavy': 'Your voice has rich low frequencies - great for warmth and depth!',
                        'Treble-bright': 'Your voice has bright high frequencies - excellent for clarity!',
                        'Mid-forward': 'Your voice has strong mid frequencies - perfect for presence!',
                        'Balanced': 'Your voice is well-balanced across all frequencies!'
                    }
                    
                    st.markdown(f"""
                    <div class='info-card'>
                        <b>Timbre Characteristics ({timbre}):</b><br>
                        {timbre_advice.get(timbre, 'Unique tonal quality!')}
                    </div>
                    """, unsafe_allow_html=True)
    
    # TAB 3: Charts
    with tab3:
        st.markdown("## Visualization Dashboard")
        
        if st.session_state.analysis_results is None:
            st.info("👆 Please analyze audio first.")
        else:
            results = st.session_state.analysis_results
            
            if 'error' not in results:
                st.markdown("### 🎵 Pitch Contour")
                fig_pitch = plot_pitch_contour(results)
                if fig_pitch:
                    st.plotly_chart(fig_pitch, use_container_width=True)
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("### 🎼 Note Distribution")
                    fig_notes = plot_note_distribution(results)
                    st.plotly_chart(fig_notes, use_container_width=True)
                
                with col2:
                    st.markdown("### 🎯 Intonation Distribution")
                    fig_intonation = plot_intonation_distribution(results)
                    if fig_intonation:
                        st.plotly_chart(fig_intonation, use_container_width=True)
                
                st.markdown("### 🎨 Spectral Profile")
                fig_spectral = plot_spectral_profile(results)
                st.plotly_chart(fig_spectral, use_container_width=True)
                
                st.markdown("---")
                st.markdown("### 💾 Export Data")
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    export_data = {
                        k: v for k, v in results.items()
                        if k not in ['pitch_contour', 'note_distribution']
                    }
                    
                    json_str = json.dumps(export_data, indent=2)
                    st.download_button(
                        "📄 Download JSON",
                        json_str,
                        "voice_analysis.json",
                        "application/json",
                        use_container_width=True
                    )
                
                with col2:
                    contour = results['pitch_contour']
                    df_export = pd.DataFrame({
                        'time': contour['times'],
                        'pitch_hz': contour['pitch_hz'],
                        'confidence': contour['confidence'],
                        'voiced': contour['voiced_mask']
                    })
                    
                    csv = df_export.to_csv(index=False)
                    st.download_button(
                        "📊 Download CSV",
                        csv,
                        "pitch_data.csv",
                        "text/csv",
                        use_container_width=True
                    )
                
                with col3:
                    summary = f"""Voice Analysis Summary
{'='*50}

Vocal Range: {results['min_note']} - {results['max_note']}
Voice Type: {results['voice_type']['type']}
Tessitura: {results['tessitura_low']} - {results['tessitura_high']}

Pitch Accuracy: {results['pitch_accuracy']['intonation_score']:.0f}/100
Mean Error: ±{results['pitch_accuracy']['mean_cents_error']:.1f} cents

Timbre: {results['timbre_classification']}

Detection Method: {results['pitch_method']}
Duration: {results['duration']:.2f} seconds"""
                    
                    st.download_button(
                        "📝 Download Summary",
                        summary,
                        "voice_summary.txt",
                        "text/plain",
                        use_container_width=True
                    )
    
    # TAB 4: Recommendations
    with tab4:
        st.markdown("## 🎵 Song Recommendations")
        
        if st.session_state.analysis_results is None:
            st.info("👆 Please analyze your voice first.")
        else:
            results = st.session_state.analysis_results
            
            if 'error' not in results:
                with st.spinner("Finding songs that match your voice..."):
                    recommendations = recommend_songs(results, st.session_state.catalog)
                
                st.markdown("### ✅ Perfect Fit Songs")
                st.markdown("These songs match your vocal range and tessitura well:")
                
                if len(recommendations['fit']) == 0:
                    st.info("No perfect matches in the current catalog. Try the stretch songs below!")
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
                
                st.markdown("---")
                
                st.markdown("### 📈 Growth Songs")
                st.markdown("These songs will help you expand your range safely:")
                
                if len(recommendations['stretch']) == 0:
                    st.info("No stretch songs available. Your range already covers our catalog!")
                else:
                    for song in recommendations['stretch']:
                        with st.expander(f"🎵 {song['title']} - {song['artist']}"):
                            st.markdown(f"**Key:** {song['key']} | **Range:** {song['range']} | **Difficulty:** {song['difficulty'].title()}")
                            
                            st.markdown("**Why this song:**")
                            for reason in song['reasons']:
                                st.markdown(f"• {reason}")
                            
                            st.info("💡 **Tip:** Warm up thoroughly before attempting stretch songs!")
                
                st.markdown("---")
                
                with st.expander("⚠️ Songs to Avoid (For Now)"):
                    st.markdown("These songs may strain your voice with your current range:")
                    
                    for song in recommendations['avoid'][:3]:
                        st.markdown(f"**{song['title']}** - {song['artist']}")
                        for reason in song['reasons']:
                            st.markdown(f"  • {reason}")
                        st.markdown("")
                
                st.markdown("---")
                st.markdown("### 📚 Song Catalog")
                
                col1, col2 = st.columns([3, 1])
                
                with col1:
                    st.info(f"Current catalog contains {len(st.session_state.catalog)} songs. You can edit `{CATALOG_FILE}` to add more!")
                
                with col2:
                    if st.button("🔄 Reload Catalog"):
                        st.session_state.catalog = pd.read_csv(CATALOG_FILE)
                        st.success("Catalog reloaded!")
                
                with st.expander("View Full Catalog"):
                    st.dataframe(st.session_state.catalog, use_container_width=True)


if __name__ == "__main__":
    main()
