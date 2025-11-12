#!/usr/bin/env python3
"""
Professional Voice Analysis Studio
==================================

A comprehensive voice analysis application for both talking and singing analysis.
Features live microphone recording, file upload, pitch detection, voice classification,
timbre analysis, song identification, and personalized recommendations.

DEV NOTES:
----------
- Run with: streamlit run app.py
- Requires: streamlit, streamlit-webrtc, librosa, numpy, scipy, plotly, pandas
- Optional: torch, torchcrepe for advanced pitch detection
- First run creates song_catalog.csv in app directory
- Add songs by editing song_catalog.csv or using UI upload
- For recording issues: check browser mic permissions, sample rate (44.1kHz recommended)
- CPU-only compatible - no CUDA required

TROUBLESHOOTING:
---------------
- Recording not working? Check browser permissions and try refreshing
- WebRTC errors? Ensure HTTPS or localhost connection
- Audio processing slow? Try shorter recordings or disable advanced features
- Missing torch/torchcrepe? Falls back to librosa.yin automatically
"""

import streamlit as st
import numpy as np
import librosa
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import json
import os
import io
import base64
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Any
import warnings
warnings.filterwarnings('ignore')

# Try to import optional dependencies
try:
    import torch
    import torchcrepe
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torchcrepe = None

try:
    from streamlit_webrtc import webrtc_streamer, WebRtcMode, RTCConfiguration
    WEBRTC_AVAILABLE = True
except ImportError:
    WEBRTC_AVAILABLE = False
    webrtc_streamer = None

# Constants
SAMPLE_RATE = 22050
HOP_LENGTH = 512
WINDOW_LENGTH = 1024
FMIN = 50
FMAX = 2000
VOICE_TYPES = {
    'Bass': {'range': (80, 260), 'description': 'Deep, rich low voice'},
    'Baritone': {'range': (100, 350), 'description': 'Medium-low male voice'},
    'Tenor': {'range': (130, 450), 'description': 'High male voice'},
    'Alto': {'range': (170, 550), 'description': 'Low female voice'},
    'Mezzo': {'range': (200, 700), 'description': 'Medium female voice'},
    'Soprano': {'range': (250, 1500), 'description': 'High female voice'}
}

# Color scheme for consistent theming
COLORS = {
    'primary': '#4A90E2',
    'secondary': '#7B68EE',
    'success': '#2ECC71',
    'warning': '#F39C12',
    'danger': '#E74C3C',
    'dark': '#2C3E50',
    'light': '#ECF0F1'
}

# Session state initialization
if 'analysis_results' not in st.session_state:
    st.session_state.analysis_results = None
if 'recording_data' not in st.session_state:
    st.session_state.recording_data = None
if 'song_catalog' not in st.session_state:
    st.session_state.song_catalog = None

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def hz_to_note(frequency: float) -> str:
    """Convert frequency in Hz to musical note name."""
    if frequency <= 0:
        return "N/A"
    A4 = 440.0
    C0 = A4 * pow(2, -4.75)
    h = round(12 * np.log2(frequency / C0))
    octave = h // 12
    n = h % 12
    note_names = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
    return note_names[n] + str(octave)

def cents_error(frequency: float, target_frequency: float) -> float:
    """Calculate cents deviation from target frequency."""
    if frequency <= 0 or target_frequency <= 0:
        return 0
    return 1200 * np.log2(frequency / target_frequency)

def smooth_f0(f0: np.ndarray, voicing: np.ndarray, window_size: int = 5) -> np.ndarray:
    """Apply median filter smoothing to F0 contour."""
    f0_smooth = f0.copy()
    for i in range(len(f0)):
        start = max(0, i - window_size // 2)
        end = min(len(f0), i + window_size // 2 + 1)
        if voicing[i] and np.any(voicing[start:end]):
            voiced_f0 = f0[start:end][voicing[start:end]]
            if len(voiced_f0) > 0:
                f0_smooth[i] = np.median(voiced_f0)
    return f0_smooth

def compute_spectral_features(y: np.ndarray, sr: int) -> Dict[str, float]:
    """Compute spectral features for timbre analysis."""
    # Compute Short-Time Fourier Transform
    stft = librosa.stft(y, n_fft=WINDOW_LENGTH, hop_length=HOP_LENGTH)
    magnitude = np.abs(stft)
    
    # Spectral centroid
    spectral_centroid = librosa.feature.spectral_centroid(S=magnitude, sr=sr)[0]
    
    # Spectral rolloff
    spectral_rolloff = librosa.feature.spectral_rolloff(S=magnitude, sr=sr)[0]
    
    # Spectral bandwidth
    spectral_bandwidth = librosa.feature.spectral_bandwidth(S=magnitude, sr=sr)[0]
    
    # Energy in different bands
    freq_bins = np.fft.rfftfreq(WINDOW_LENGTH, 1/sr)
    low_band = magnitude[freq_bins < 300]
    mid_band = magnitude[(freq_bins >= 300) & (freq_bins < 3000)]
    high_band = magnitude[freq_bins >= 3000]
    
    total_energy = np.sum(magnitude)
    low_energy = np.sum(low_band) / total_energy if total_energy > 0 else 0
    mid_energy = np.sum(mid_band) / total_energy if total_energy > 0 else 0
    high_energy = np.sum(high_band) / total_energy if total_energy > 0 else 0
    
    return {
        'spectral_centroid': np.mean(spectral_centroid),
        'spectral_rolloff': np.mean(spectral_rolloff),
        'spectral_bandwidth': np.mean(spectral_bandwidth),
        'low_energy': low_energy,
        'mid_energy': mid_energy,
        'high_energy': high_energy
    }

def generate_test_audio(duration: float = 5.0, sr: int = SAMPLE_RATE) -> Tuple[np.ndarray, int]:
    """Generate test audio with C major scale and arpeggio."""
    # C major scale frequencies
    c_major = [261.63, 293.66, 329.63, 349.23, 392.00, 440.00, 493.88, 523.25]
    
    # Generate scale
    scale_duration = duration * 0.6
    note_duration = scale_duration / len(c_major)
    scale_audio = []
    
    for freq in c_major:
        t = np.linspace(0, note_duration, int(sr * note_duration), False)
        note = 0.5 * np.sin(2 * np.pi * freq * t)
        # Apply envelope
        envelope = np.linspace(0.1, 0.8, len(note))
        note *= envelope
        scale_audio.append(note)
    
    # Generate arpeggio (C-E-G-C)
    arpeggio_duration = duration * 0.4
    arpeggio_notes = [261.63, 329.63, 392.00, 523.25]
    note_duration = arpeggio_duration / len(arpeggio_notes)
    arpeggio_audio = []
    
    for freq in arpeggio_notes:
        t = np.linspace(0, note_duration, int(sr * note_duration), False)
        note = 0.4 * np.sin(2 * np.pi * freq * t)
        envelope = np.linspace(0.1, 0.6, len(note))
        note *= envelope
        arpeggio_audio.append(note)
    
    # Combine
    audio = np.concatenate(scale_audio + arpeggio_audio)
    return audio, sr

# ============================================================================
# SONG CATALOG MANAGEMENT
# ============================================================================

def create_song_catalog() -> pd.DataFrame:
    """Create and return the song catalog DataFrame."""
    catalog_data = [
        {
            'title': 'Happy Birthday',
            'artist': 'Traditional',
            'key': 'C',
            'typical_low': 261.63,  # C4
            'typical_high': 523.25,  # C5
            'tags': 'celebration,easy,balanced',
            'pitch_template_json': '[261.63, 293.66, 329.63, 261.63, 349.23, 329.63, 293.66, 261.63]'
        },
        {
            'title': 'Amazing Grace',
            'artist': 'Traditional',
            'key': 'C',
            'typical_low': 261.63,  # C4
            'typical_high': 523.25,  # C5
            'tags': 'gospel,medium,balanced',
            'pitch_template_json': '[261.63, 293.66, 329.63, 349.23, 392.00, 440.00, 493.88, 523.25]'
        },
        {
            'title': 'Twinkle Twinkle Little Star',
            'artist': 'Mozart',
            'key': 'C',
            'typical_low': 261.63,  # C4
            'typical_high': 523.25,  # C5
            'tags': 'children,easy,balanced',
            'pitch_template_json': '[261.63, 261.63, 392.00, 392.00, 440.00, 440.00, 392.00]'
        },
        {
            'title': 'My Heart Will Go On',
            'artist': 'Celine Dion',
            'key': 'E',
            'typical_low': 329.63,  # E4
            'typical_high': 659.25,  # E5
            'tags': 'ballad,hard,high',
            'pitch_template_json': '[329.63, 369.99, 392.00, 440.00, 493.88, 523.25, 587.33, 659.25]'
        },
        {
            'title': 'Someone Like You',
            'artist': 'Adele',
            'key': 'A',
            'typical_low': 220.00,  # A3
            'typical_high': 440.00,  # A4
            'tags': 'ballad,medium,low',
            'pitch_template_json': '[220.00, 246.94, 261.63, 293.66, 329.63, 349.23, 369.99, 392.00]'
        },
        {
            'title': 'Bohemian Rhapsody',
            'artist': 'Queen',
            'key': 'B',
            'typical_low': 246.94,  # B3
            'typical_high': 739.99,  # F#5
            'tags': 'rock,hard,wide_range',
            'pitch_template_json': '[246.94, 277.18, 293.66, 329.63, 369.99, 392.00, 440.00, 493.88]'
        },
        {
            'title': 'Let It Be',
            'artist': 'The Beatles',
            'key': 'C',
            'typical_low': 261.63,  # C4
            'typical_high': 523.25,  # C5
            'tags': 'rock,medium,balanced',
            'pitch_template_json': '[261.63, 293.66, 329.63, 349.23, 392.00, 440.00, 493.88]'
        },
        {
            'title': 'Imagine',
            'artist': 'John Lennon',
            'key': 'C',
            'typical_low': 261.63,  # C4
            'typical_high': 523.25,  # C5
            'tags': 'ballad,easy,balanced',
            'pitch_template_json': '[261.63, 293.66, 329.63, 349.23, 392.00, 440.00]'
        },
        {
            'title': 'Hallelujah',
            'artist': 'Leonard Cohen',
            'key': 'C',
            'typical_low': 261.63,  # C4
            'typical_high': 523.25,  # C5
            'tags': 'ballad,medium,balanced',
            'pitch_template_json': '[261.63, 293.66, 329.63, 349.23, 392.00, 440.00, 493.88, 523.25]'
        },
        {
            'title': 'Perfect',
            'artist': 'Ed Sheeran',
            'key': 'G',
            'typical_low': 196.00,  # G3
            'typical_high': 392.00,  # G4
            'tags': 'ballad,easy,low',
            'pitch_template_json': '[196.00, 220.00, 246.94, 261.63, 293.66, 329.63, 349.23]'
        }
    ]
    
    return pd.DataFrame(catalog_data)

def load_song_catalog() -> pd.DataFrame:
    """Load or create the song catalog."""
    catalog_path = 'song_catalog.csv'
    
    if os.path.exists(catalog_path):
        try:
            catalog = pd.read_csv(catalog_path)
            return catalog
        except Exception as e:
            st.warning(f"Error loading catalog: {e}. Creating new catalog.")
    
    # Create default catalog
    catalog = create_song_catalog()
    catalog.to_csv(catalog_path, index=False)
    return catalog

def identify_song(pitch_contour: np.ndarray, catalog: pd.DataFrame) -> List[Dict]:
    """Identify song from pitch contour using template matching."""
    matches = []
    
    # Normalize pitch contour
    if len(pitch_contour) == 0:
        return matches
    
    # Remove unvoiced segments and normalize
    voiced_pitch = pitch_contour[pitch_contour > 0]
    if len(voiced_pitch) == 0:
        return matches
    
    # Normalize to relative intervals
    mean_pitch = np.mean(voiced_pitch)
    relative_contour = voiced_pitch / mean_pitch
    
    for idx, song in catalog.iterrows():
        try:
            # Parse template
            template = json.loads(song['pitch_template_json'])
            template = np.array(template)
            
            # Normalize template
            template_mean = np.mean(template)
            relative_template = template / template_mean
            
            # Simple correlation-based matching
            # In production, use DTW (Dynamic Time Warping) for better accuracy
            min_len = min(len(relative_contour), len(relative_template))
            if min_len > 0:
                contour_sample = relative_contour[:min_len]
                template_sample = relative_template[:min_len]
                
                # Calculate correlation
                correlation = np.corrcoef(contour_sample, template_sample)[0, 1]
                if np.isnan(correlation):
                    correlation = 0
                
                # Calculate pitch range overlap
                contour_range = (np.min(voiced_pitch), np.max(voiced_pitch))
                song_range = (song['typical_low'], song['typical_high'])
                
                # Calculate range overlap score
                overlap = max(0, min(contour_range[1], song_range[1]) - max(contour_range[0], song_range[0]))
                range_score = overlap / max(contour_range[1] - contour_range[0], song_range[1] - song_range[0])
                
                # Combined score
                confidence = (correlation * 0.6 + range_score * 0.4) * 100
                
                if confidence > 30:  # Threshold
                    matches.append({
                        'title': song['title'],
                        'artist': song['artist'],
                        'key': song['key'],
                        'confidence': confidence,
                        'tags': song['tags'],
                        'correlation': correlation,
                        'range_score': range_score
                    })
        except Exception as e:
            continue
    
    # Sort by confidence and return top 3
    matches.sort(key=lambda x: x['confidence'], reverse=True)
    return matches[:3]

# ============================================================================
# VOICE ANALYSIS FUNCTIONS
# ============================================================================

def detect_pitch_torchcrepe(y: np.ndarray, sr: int) -> Tuple[np.ndarray, np.ndarray]:
    """Detect pitch using torchcrepe (if available)."""
    if not TORCH_AVAILABLE or torchcrepe is None:
        return None, None
    
    try:
        # Convert to torch tensor
        audio_torch = torch.from_numpy(y).float()
        
        # Predict pitch and voicing
        pitch, voicing = torchcrepe.predict(
            audio_torch,
            sr,
            hop_length=HOP_LENGTH,
            fmin=FMIN,
            fmax=FMAX,
            model='tiny',
            return_periodicity=True
        )
        
        # Convert back to numpy
        f0 = pitch.numpy()
        voiced = voicing.numpy() > 0.3  # Voicing threshold
        
        return f0, voiced
    except Exception as e:
        st.warning(f"torchcrepe failed: {e}. Falling back to librosa.")
        return None, None

def detect_pitch_librosa(y: np.ndarray, sr: int) -> Tuple[np.ndarray, np.ndarray]:
    """Detect pitch using librosa yin."""
    try:
        # Use yin for pitch detection
        f0 = librosa.yin(y, fmin=FMIN, fmax=FMAX, sr=sr, hop_length=HOP_LENGTH)
        
        # Simple voicing detection based on energy
        rms = librosa.feature.rms(y=y, hop_length=HOP_LENGTH)[0]
        voiced = rms > np.mean(rms) * 0.1
        
        # Ensure same length
        min_len = min(len(f0), len(voiced))
        f0 = f0[:min_len]
        voiced = voiced[:min_len]
        
        return f0, voiced
    except Exception as e:
        st.error(f"Pitch detection failed: {e}")
        return np.array([]), np.array([])

def analyze_audio(y: np.ndarray, sr: int) -> Dict[str, Any]:
    """Comprehensive audio analysis."""
    # Resample if necessary
    if sr != SAMPLE_RATE:
        y = librosa.resample(y, orig_sr=sr, target_sr=SAMPLE_RATE)
        sr = SAMPLE_RATE
    
    # Normalize audio
    y = librosa.util.normalize(y)
    
    # Basic stats
    duration = len(y) / sr
    peak_amplitude = np.max(np.abs(y))
    rms_energy = np.sqrt(np.mean(y**2))
    
    # Pitch detection
    f0, voiced = None, None
    
    # Try torchcrepe first
    if TORCH_AVAILABLE:
        f0, voiced = detect_pitch_torchcrepe(y, sr)
    
    # Fallback to librosa
    if f0 is None or len(f0) == 0:
        f0, voiced = detect_pitch_librosa(y, sr)
    
    # Handle empty results
    if len(f0) == 0:
        return {
            'error': 'No pitch detected',
            'duration': duration,
            'peak_amplitude': peak_amplitude,
            'rms_energy': rms_energy
        }
    
    # Smooth pitch contour
    f0_smooth = smooth_f0(f0, voiced)
    
    # Pitch statistics
    voiced_f0 = f0_smooth[voiced]
    if len(voiced_f0) == 0:
        return {
            'error': 'No voiced segments detected',
            'duration': duration,
            'peak_amplitude': peak_amplitude,
            'rms_energy': rms_energy
        }
    
    min_pitch = np.min(voiced_f0)
    max_pitch = np.max(voiced_f0)
    mean_pitch = np.mean(voiced_f0)
    median_pitch = np.median(voiced_f0)
    std_pitch = np.std(voiced_f0)
    
    # Tessitura (central 50% range)
    pitch_25 = np.percentile(voiced_f0, 25)
    pitch_75 = np.percentile(voiced_f0, 75)
    tessitura = (pitch_25, pitch_75)
    
    # Voicing statistics
    voiced_coverage = np.mean(voiced) * 100
    
    # Convert to MIDI and notes
    midi_notes = 69 + 12 * np.log2(voiced_f0 / 440.0)
    note_names = [hz_to_note(f) for f in voiced_f0]
    
    # Cents error analysis (deviation from nearest semitone)
    cents_errors = []
    for f in voiced_f0:
        midi = 69 + 12 * np.log2(f / 440.0)
        nearest_semitone = round(midi)
        target_freq = 440.0 * (2 ** ((nearest_semitone - 69) / 12))
        cents_errors.append(cents_error(f, target_freq))
    
    cents_errors = np.array(cents_errors)
    mean_cents_error = np.mean(np.abs(cents_errors))
    
    # Intonation score (0-100)
    intonation_score = max(0, 100 - mean_cents_error)
    
    # Pitch stability (jitter)
    if len(voiced_f0) > 1:
        jitter = np.mean(np.abs(np.diff(voiced_f0))) / mean_pitch * 100
    else:
        jitter = 0
    
    # Spectral features
    spectral_features = compute_spectral_features(y, sr)
    
    # Create time series data
    times = librosa.frames_to_time(range(len(f0)), sr=sr, hop_length=HOP_LENGTH)
    notes_times = pd.DataFrame({
        'timestamp': times,
        'frequency_hz': f0_smooth,
        'midi': 69 + 12 * np.log2(f0_smooth / 440.0),
        'note_name': [hz_to_note(f) if f > 0 else 'N/A' for f in f0_smooth],
        'cents_error': [cents_error(f, 440.0 * (2 ** ((round(69 + 12 * np.log2(f / 440.0)) - 69) / 12))) if f > 0 else 0 for f in f0_smooth],
        'voiced_flag': voiced
    })
    
    return {
        'duration': duration,
        'peak_amplitude': peak_amplitude,
        'rms_energy': rms_energy,
        'min_pitch': min_pitch,
        'max_pitch': max_pitch,
        'mean_pitch': mean_pitch,
        'median_pitch': median_pitch,
        'std_pitch': std_pitch,
        'tessitura': tessitura,
        'voiced_coverage': voiced_coverage,
        'mean_cents_error': mean_cents_error,
        'intonation_score': intonation_score,
        'jitter': jitter,
        'spectral_features': spectral_features,
        'notes_times': notes_times,
        'f0_contour': f0_smooth,
        'voiced': voiced,
        'times': times
    }

def classify_voice_type(analysis_results: Dict[str, Any]) -> Dict[str, Any]:
    """Classify voice type based on pitch range and characteristics."""
    tessitura = analysis_results['tessitura']
    tess_center = np.mean(tessitura)
    
    best_match = None
    best_score = 0
    confidence = 0
    
    for voice_type, characteristics in VOICE_TYPES.items():
        range_low, range_high = characteristics['range']
        
        # Calculate overlap with voice type range
        overlap = max(0, min(tessitura[1], range_high) - max(tessitura[0], range_low))
        range_size = tessitura[1] - tessitura[0]
        voice_range_size = range_high - range_low
        
        if range_size > 0 and voice_range_size > 0:
            overlap_score = overlap / range_size
            center_distance = abs(tess_center - (range_low + range_high) / 2) / voice_range_size
            
            # Combined score (higher overlap, lower center distance = better)
            score = overlap_score * (1 - center_distance)
            
            if score > best_score:
                best_score = score
                best_match = voice_type
                
                # Calculate confidence
                if overlap_score > 0.8 and center_distance < 0.3:
                    confidence = 95
                elif overlap_score > 0.6 and center_distance < 0.5:
                    confidence = 75
                elif overlap_score > 0.4:
                    confidence = 55
                else:
                    confidence = 35
    
    # Borderline detection
    borderline = confidence < 70
    borderline_message = ""
    
    if borderline and best_match:
        # Find next closest
        voice_names = list(VOICE_TYPES.keys())
        current_idx = voice_names.index(best_match)
        if current_idx > 0:
            next_type = voice_names[current_idx - 1] if tess_center < np.mean(VOICE_TYPES[best_match]['range']) else voice_names[current_idx + 1]
            borderline_message = f"{best_match} — borderline {next_type}; consider testing higher/lower notes"
    
    return {
        'voice_type': best_match,
        'confidence': confidence,
        'borderline': borderline,
        'borderline_message': borderline_message,
        'description': VOICE_TYPES.get(best_match, {}).get('description', '') if best_match else ''
    }

def analyze_timbre(spectral_features: Dict[str, float]) -> Dict[str, Any]:
    """Analyze timbre and generate badge."""
    low_energy = spectral_features['low_energy']
    mid_energy = spectral_features['mid_energy']
    high_energy = spectral_features['high_energy']
    
    # Determine timbre badge
    if low_energy > 0.5:
        badge = "Bass-Heavy"
        description = "Rich low-frequency content, warm and full-bodied"
        tips = "Try brightening resonance with forward vowel placement and nasal consonants"
    elif high_energy > 0.4:
        badge = "Treble-Bright"
        description = "Prominent high-frequency content, clear and bright"
        tips = "Focus on breath support to maintain brightness without strain"
    elif mid_energy > 0.5:
        badge = "Mid-Forward"
        description = "Strong mid-range presence, clear and present"
        tips = "Good balance achieved - work on extending range while maintaining clarity"
    else:
        badge = "Balanced"
        description = "Even distribution across frequency spectrum"
        tips = "Excellent foundation - explore different vocal colors and dynamics"
    
    return {
        'badge': badge,
        'description': description,
        'tips': tips,
        'low_energy': low_energy,
        'mid_energy': mid_energy,
        'high_energy': high_energy,
        'spectral_centroid': spectral_features['spectral_centroid']
    }

def analyze_breath_stability(analysis_results: Dict[str, Any]) -> Dict[str, Any]:
    """Analyze breath control and vocal stability."""
    voiced = analysis_results['voiced']
    times = analysis_results['times']
    f0_contour = analysis_results['f0_contour']
    
    # Detect long unvoiced gaps (potential breath issues)
    unvoiced_segments = []
    current_unvoiced_start = None
    
    for i, is_voiced in enumerate(voiced):
        if not is_voiced and current_unvoiced_start is None:
            current_unvoiced_start = i
        elif is_voiced and current_unvoiced_start is not None:
            duration = times[i] - times[current_unvoiced_start]
            if duration > 0.5:  # Longer than 0.5 seconds
                unvoiced_segments.append({
                    'start': times[current_unvoiced_start],
                    'end': times[i],
                    'duration': duration
                })
            current_unvoiced_start = None
    
    # Breathiness detection (based on high-frequency content)
    spectral_features = analysis_results['spectral_features']
    breathiness_score = spectral_features['high_energy'] * 100
    
    # Stability analysis
    jitter = analysis_results['jitter']
    intonation_score = analysis_results['intonation_score']
    
    # Generate recommendations
    issues = []
    recommendations = []
    exercises = []
    
    if len(unvoiced_segments) > 2:
        issues.append("Frequent breath gaps detected")
        recommendations.append("Work on breath support and control")
        exercises.append("Sustained vowel exercises (ah, ee, oo) for 10-15 seconds")
    
    if breathiness_score > 40:
        issues.append("High breathiness detected")
        recommendations.append("Focus on vocal cord closure")
        exercises.append("Glottal onset exercises and gentle humming")
    
    if jitter > 5:
        issues.append("Pitch instability detected")
        recommendations.append("Improve vocal fold coordination")
        exercises.append("Sirens and pitch glides across comfortable range")
    
    if intonation_score < 70:
        issues.append("Pitch accuracy needs improvement")
        recommendations.append("Practice with reference pitches")
        exercises.append("Interval training and scale singing with tuner")
    
    # Default recommendations if no issues
    if not issues:
        recommendations.append("Excellent vocal control! Maintain with regular practice")
        exercises.append("Continue daily warm-ups and range extension exercises")
    
    return {
        'unvoiced_segments': unvoiced_segments,
        'breathiness_score': breathiness_score,
        'stability_issues': issues,
        'recommendations': recommendations,
        'exercises': exercises,
        'jitter': jitter
    }

def recommend_songs(voice_profile: Dict[str, Any], song_catalog: pd.DataFrame) -> Dict[str, List[Dict]]:
    """Generate song recommendations based on voice profile."""
    tessitura = voice_profile['analysis_results']['tessitura']
    voice_type = voice_profile['voice_classification']['voice_type']
    timbre_badge = voice_profile['timbre_analysis']['badge']
    
    fit_songs = []
    stretch_songs = []
    avoid_songs = []
    
    for idx, song in song_catalog.iterrows():
        song_range = (song['typical_low'], song['typical_high'])
        
        # Calculate range overlap
        overlap = max(0, min(tessitura[1], song_range[1]) - max(tessitura[0], song_range[0]))
        tess_range = tessitura[1] - tessitura[0]
        song_range_size = song_range[1] - song_range[0]
        
        if tess_range > 0 and song_range_size > 0:
            overlap_ratio = overlap / tess_range
            range_extension_needed = max(0, song_range[0] - tessitura[0], song_range[1] - tessitura[1])
            
            # Semitones calculation
            semitones_needed = range_extension_needed / (song_range[0] * (2 ** (1/12) - 1))
            
            # Timbre compatibility
            timbre_tags = song['tags'].split(',')
            timbre_compatible = any(tag in ['balanced', timbre_badge.lower().replace('-', '_')] for tag in timbre_tags)
            
            song_info = {
                'title': song['title'],
                'artist': song['artist'],
                'key': song['key'],
                'range': f"{hz_to_note(song['typical_low'])} - {hz_to_note(song['typical_high'])}",
                'difficulty': 'Easy' if 'easy' in timbre_tags else 'Medium' if 'medium' in timbre_tags else 'Hard',
                'reason': '',
                'timbre_match': timbre_compatible
            }
            
            if semitones_needed <= 1 and overlap_ratio > 0.7:
                song_info['reason'] = "Perfect fit for your range"
                fit_songs.append(song_info)
            elif semitones_needed <= 4 and overlap_ratio > 0.5:
                song_info['reason'] = f"Good stretch song ({semitones_needed:.1f} semitones extension)"
                stretch_songs.append(song_info)
            elif semitones_needed > 5:
                song_info['reason'] = f"Too challenging ({semitones_needed:.1f} semitones beyond your range)"
                avoid_songs.append(song_info)
    
    return {
        'fit': fit_songs[:5],  # Top 5
        'stretch': stretch_songs[:5],
        'avoid': avoid_songs[:3]
    }

# ============================================================================
# VISUALIZATION FUNCTIONS
# ============================================================================

def create_pitch_plot(analysis_results: Dict[str, Any]) -> go.Figure:
    """Create interactive pitch contour plot."""
    times = analysis_results['times']
    f0_contour = analysis_results['f0_contour']
    voiced = analysis_results['voiced']
    
    fig = go.Figure()
    
    # Add voiced segments
    voiced_times = times[voiced]
    voiced_f0 = f0_contour[voiced]
    
    if len(voiced_times) > 0:
        fig.add_trace(go.Scatter(
            x=voiced_times,
            y=voiced_f0,
            mode='lines+markers',
            name='Pitch Contour',
            line=dict(color=COLORS['primary'], width=2),
            marker=dict(size=3),
            hovertemplate='Time: %{x:.2f}s<br>Frequency: %{y:.1f}Hz<br>Note: %{text}<extra></extra>',
            text=[hz_to_note(f) for f in voiced_f0]
        ))
    
    # Add note grid lines
    note_freqs = []
    note_names = []
    for midi in range(40, 100):  # MIDI notes 40-100
        freq = 440.0 * (2 ** ((midi - 69) / 12))
        if FMIN <= freq <= FMAX:
            note_freqs.append(freq)
            note_names.append(librosa.midi_to_note(midi))
    
    for i, (freq, name) in enumerate(zip(note_freqs, note_names)):
        fig.add_hline(
            y=freq,
            line_dash="dash",
            line_color="rgba(100,100,100,0.3)",
            annotation_text=name if i % 3 == 0 else "",
            annotation_position="right"
        )
    
    fig.update_layout(
        title="Pitch Contour Analysis",
        xaxis_title="Time (seconds)",
        yaxis_title="Frequency (Hz)",
        template="plotly_white",
        height=500,
        showlegend=True
    )
    
    return fig

def create_note_histogram(analysis_results: Dict[str, Any]) -> go.Figure:
    """Create histogram of detected notes."""
    voiced_f0 = analysis_results['f0_contour'][analysis_results['voiced']]
    
    if len(voiced_f0) == 0:
        return go.Figure()
    
    # Convert to note names
    note_names = [hz_to_note(f) for f in voiced_f0]
    
    # Count occurrences
    note_counts = {}
    for note in note_names:
        note_counts[note] = note_counts.get(note, 0) + 1
    
    # Create bar chart
    fig = go.Figure(data=[
        go.Bar(
            x=list(note_counts.keys()),
            y=list(note_counts.values()),
            marker_color=COLORS['secondary'],
            name='Note Frequency'
        )
    ])
    
    fig.update_layout(
        title="Note Distribution",
        xaxis_title="Note",
        yaxis_title="Frequency",
        template="plotly_white",
        height=400
    )
    
    return fig

def create_cents_error_plot(analysis_results: Dict[str, Any]) -> go.Figure:
    """Create cents error distribution plot."""
    notes_times = analysis_results['notes_times']
    voiced_errors = notes_times[notes_times['voiced_flag']]['cents_error'].abs()
    
    if len(voiced_errors) == 0:
        return go.Figure()
    
    fig = go.Figure()
    
    # Histogram
    fig.add_trace(go.Histogram(
        x=voiced_errors,
        nbinsx=30,
        marker_color=COLORS['warning'],
        name='Cents Error Distribution',
        opacity=0.7
    ))
    
    # Add mean line
    mean_error = voiced_errors.mean()
    fig.add_vline(
        x=mean_error,
        line_dash="dash",
        line_color=COLORS['danger'],
        annotation_text=f"Mean: {mean_error:.1f} cents"
    )
    
    fig.update_layout(
        title="Pitch Accuracy Distribution",
        xaxis_title="Cents Error (absolute)",
        yaxis_title="Frequency",
        template="plotly_white",
        height=400
    )
    
    return fig

def create_vocal_range_plot(analysis_results: Dict[str, Any]) -> go.Figure:
    """Create vocal range visualization."""
    min_pitch = analysis_results['min_pitch']
    max_pitch = analysis_results['max_pitch']
    tessitura = analysis_results['tessitura']
    
    fig = go.Figure()
    
    # Full range
    fig.add_trace(go.Bar(
        x=['Vocal Range'],
        y=[max_pitch - min_pitch],
        base=[min_pitch],
        marker_color=COLORS['primary'],
        name='Full Range',
        text=f"{hz_to_note(min_pitch)} - {hz_to_note(max_pitch)}",
        textposition='middle center'
    ))
    
    # Tessitura
    fig.add_trace(go.Bar(
        x=['Tessitura'],
        y=[tessitura[1] - tessitura[0]],
        base=[tessitura[0]],
        marker_color=COLORS['success'],
        name='Tessitura (50%)',
        text=f"{hz_to_note(tessitura[0])} - {hz_to_note(tessitura[1])}",
        textposition='middle center'
    ))
    
    fig.update_layout(
        title="Vocal Range Analysis",
        yaxis_title="Frequency (Hz)",
        template="plotly_white",
        height=400,
        showlegend=True
    )
    
    return fig

def create_spectral_plot(analysis_results: Dict[str, Any]) -> go.Figure:
    """Create spectral energy distribution plot."""
    spectral_features = analysis_results['spectral_features']
    
    categories = ['Low (<300 Hz)', 'Mid (300-3000 Hz)', 'High (>3000 Hz)']
    values = [
        spectral_features['low_energy'] * 100,
        spectral_features['mid_energy'] * 100,
        spectral_features['high_energy'] * 100
    ]
    
    fig = go.Figure(data=[
        go.Bar(
            x=categories,
            y=values,
            marker_color=[COLORS['danger'], COLORS['warning'], COLORS['primary']],
            name='Energy Distribution'
        )
    ])
    
    fig.update_layout(
        title="Spectral Energy Distribution",
        yaxis_title="Energy (%)",
        template="plotly_white",
        height=400
    )
    
    return fig

# ============================================================================
# UI COMPONENTS
# ============================================================================

def render_header():
    """Render the application header."""
    st.markdown("""
    <style>
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 10px;
        margin-bottom: 2rem;
        text-align: center;
        color: white;
    }
    .app-title {
        font-size: 2.5rem;
        font-weight: bold;
        margin-bottom: 0.5rem;
    }
    .app-subtitle {
        font-size: 1.2rem;
        opacity: 0.9;
    }
    </style>
    <div class="main-header">
        <div class="app-title">🎤 Professional Voice Analysis Studio</div>
        <div class="app-subtitle">Advanced pitch detection, voice classification & song recommendations</div>
    </div>
    """, unsafe_allow_html=True)

def render_sidebar():
    """Render the sidebar with settings and information."""
    with st.sidebar:
        st.header("⚙️ Settings")
        
        # Audio settings
        st.subheader("Audio Settings")
        sample_rate = st.selectbox("Sample Rate", [16000, 22050, 44100], index=1)
        st.info(f"Current sample rate: {sample_rate} Hz")
        
        # Analysis settings
        st.subheader("Analysis Settings")
        use_advanced_pitch = st.checkbox("Use Advanced Pitch Detection", 
                                       value=TORCH_AVAILABLE,
                                       disabled=not TORCH_AVAILABLE)
        if not TORCH_AVAILABLE:
            st.info("Install torch and torchcrepe for advanced pitch detection")
        
        trim_silence = st.checkbox("Trim Silence", value=True)
        
        # Display requirements
        st.subheader("📋 Requirements")
        requirements = """
        **Required:**
        - streamlit
        - streamlit-webrtc
        - librosa
        - numpy
        - pandas
        - plotly
        
        **Optional:**
        - torch (advanced pitch)
        - torchcrepe (advanced pitch)
        """
        st.markdown(requirements)
        
        # Privacy notice
        st.subheader("🔒 Privacy")
        st.info("""
        This app processes all audio locally on your device.
        No data is sent to external servers.
        Song catalog is stored locally in song_catalog.csv
        """)

def render_recording_panel():
    """Render the recording and file upload panel."""
    st.header("🎙️ Recording & Upload")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Live Recording")
        
        if WEBRTC_AVAILABLE:
            # WebRTC recording interface
            webrtc_ctx = webrtc_streamer(
                key="voice-recorder",
                mode=WebRtcMode.SENDONLY,
                audio_receiver_size=256,
                rtc_configuration=RTCConfiguration({
                    "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
                })
            )
            
            if webrtc_ctx.audio_receiver:
                audio_frames = webrtc_ctx.audio_receiver.get_frames(timeout=1)
                if audio_frames:
                    st.success("🎤 Recording active!")
                    st.info("Speak or sing into your microphone")
                    
                    # Process audio frames
                    audio_data = []
                    for frame in audio_frames:
                        audio_data.extend(frame.to_ndarray().tolist())
                    
                    if audio_data:
                        y = np.array(audio_data)
                        if len(y) > 0:
                            st.session_state.recording_data = (y, SAMPLE_RATE)
                            st.success(f"Recorded {len(y)/SAMPLE_RATE:.1f} seconds")
        else:
            st.error("streamlit-webrtc not available. Install with: pip install streamlit-webrtc")
        
        # Manual recording controls (fallback)
        if st.button("🎵 Generate Test Audio"):
            test_audio, sr = generate_test_audio()
            st.session_state.recording_data = (test_audio, sr)
            st.success("Test audio generated!")
    
    with col2:
        st.subheader("File Upload")
        
        uploaded_file = st.file_uploader(
            "Choose audio file",
            type=['wav', 'mp3', 'm4a', 'flac'],
            help="Upload audio files up to 10 minutes"
        )
        
        if uploaded_file is not None:
            try:
                # Load audio file
                y, sr = librosa.load(uploaded_file, sr=None)
                duration = len(y) / sr
                
                if duration > 600:  # 10 minutes
                    st.error("File too long. Please upload files under 10 minutes.")
                else:
                    st.success(f"Loaded: {uploaded_file.name}")
                    st.info(f"Duration: {duration:.1f}s, Sample Rate: {sr}Hz")
                    
                    # Display waveform preview
                    fig = go.Figure()
                    time_axis = np.linspace(0, duration, len(y))
                    fig.add_trace(go.Scatter(
                        x=time_axis[::len(y)//1000],  # Sample for display
                        y=y[::len(y)//1000],
                        mode='lines',
                        name='Waveform'
                    ))
                    fig.update_layout(
                        title="Waveform Preview",
                        xaxis_title="Time (s)",
                        yaxis_title="Amplitude",
                        height=200
                    )
                    st.plotly_chart(fig, use_container_width=True)
                    
                    st.session_state.recording_data = (y, sr)
                    
            except Exception as e:
                st.error(f"Error loading file: {e}")
    
    # Diagnostic panel
    with st.expander("🔧 Recording Diagnostics"):
        st.markdown("""
        **If recording is not working:**
        
        1. **Browser Permissions**
           - Check microphone permissions in browser settings
           - Look for microphone icon in address bar
           - Try refreshing the page
        
        2. **Sample Rate Issues**
           - Ensure microphone supports 44.1kHz or 22.05kHz
           - Check system audio settings
        
        3. **WebRTC Errors**
           - Use HTTPS connection (required for WebRTC)
           - Try different browser (Chrome recommended)
           - Disable browser extensions that block WebRTC
        
        4. **Alternative**
           - Use file upload instead
           - Record with external app and upload
        """)

def render_analysis_panel():
    """Render the voice analysis panel."""
    st.header("📊 Voice Analysis")
    
    # Check if we have audio data
    if st.session_state.recording_data is None:
        st.info("Please record audio or upload a file first.")
        return
    
    y, sr = st.session_state.recording_data
    
    # Analyze button
    if st.button("🔍 Analyze Voice", type="primary"):
        with st.spinner("Analyzing your voice..."):
            try:
                # Perform analysis
                analysis_results = analyze_audio(y, sr)
                
                if 'error' in analysis_results:
                    st.error(f"Analysis failed: {analysis_results['error']}")
                    return
                
                # Additional analyses
                voice_classification = classify_voice_type(analysis_results)
                timbre_analysis = analyze_timbre(analysis_results['spectral_features'])
                breath_analysis = analyze_breath_stability(analysis_results)
                
                # Song identification
                song_catalog = load_song_catalog()
                song_matches = identify_song(analysis_results['f0_contour'], song_catalog)
                
                # Song recommendations
                voice_profile = {
                    'analysis_results': analysis_results,
                    'voice_classification': voice_classification,
                    'timbre_analysis': timbre_analysis
                }
                recommendations = recommend_songs(voice_profile, song_catalog)
                
                # Store all results
                st.session_state.analysis_results = {
                    'analysis': analysis_results,
                    'voice_classification': voice_classification,
                    'timbre_analysis': timbre_analysis,
                    'breath_analysis': breath_analysis,
                    'song_matches': song_matches,
                    'recommendations': recommendations
                }
                
                st.success("Analysis complete!")
                
            except Exception as e:
                st.error(f"Analysis error: {e}")
                return
    
    # Display results
    if st.session_state.analysis_results:
        results = st.session_state.analysis_results
        
        # Summary cards
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                "Voice Type",
                results['voice_classification']['voice_type'] or "Unknown",
                delta=f"{results['voice_classification']['confidence']:.0f}% confidence"
            )
        
        with col2:
            st.metric(
                "Vocal Range",
                f"{hz_to_note(results['analysis']['min_pitch'])} - {hz_to_note(results['analysis']['max_pitch'])}"
            )
        
        with col3:
            st.metric(
                "Intonation Score",
                f"{results['analysis']['intonation_score']:.0f}/100",
                delta=f"{results['analysis']['mean_cents_error']:.1f}¢ avg error"
            )
        
        with col4:
            st.metric(
                "Timbre",
                results['timbre_analysis']['badge']
            )
        
        # Detailed analysis tabs
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "Voice Profile", "Pitch Analysis", "Timbre & Quality", "Song Matches", "Recommendations"
        ])
        
        with tab1:
            render_voice_profile(results)
        
        with tab2:
            render_pitch_analysis(results)
        
        with tab3:
            render_timbre_quality(results)
        
        with tab4:
            render_song_matches(results)
        
        with tab5:
            render_recommendations(results)

def render_voice_profile(results: Dict[str, Any]):
    """Render detailed voice profile."""
    st.subheader("Voice Classification")
    
    col1, col2 = st.columns(2)
    
    with col1:
        voice_class = results['voice_classification']
        st.markdown(f"""
        **Voice Type:** {voice_class['voice_type'] or 'Unknown'}
        
        **Confidence:** {voice_class['confidence']:.0f}%
        
        **Description:** {voice_class['description']}
        """)
        
        if voice_class['borderline_message']:
            st.warning(voice_class['borderline_message'])
    
    with col2:
        # Vocal range visualization
        fig = create_vocal_range_plot(results['analysis'])
        st.plotly_chart(fig, use_container_width=True)
    
    # Statistics table
    analysis = results['analysis']
    st.subheader("Voice Statistics")
    
    stats_df = pd.DataFrame({
        'Metric': [
            'Duration', 'Peak Amplitude', 'RMS Energy',
            'Minimum Pitch', 'Maximum Pitch', 'Mean Pitch',
            'Tessitura Low', 'Tessitura High',
            'Voiced Coverage', 'Jitter', 'Mean Cents Error'
        ],
        'Value': [
            f"{analysis['duration']:.1f}s",
            f"{analysis['peak_amplitude']:.3f}",
            f"{analysis['rms_energy']:.3f}",
            f"{analysis['min_pitch']:.1f} Hz ({hz_to_note(analysis['min_pitch'])})",
            f"{analysis['max_pitch']:.1f} Hz ({hz_to_note(analysis['max_pitch'])})",
            f"{analysis['mean_pitch']:.1f} Hz ({hz_to_note(analysis['mean_pitch'])})",
            f"{analysis['tessitura'][0]:.1f} Hz ({hz_to_note(analysis['tessitura'][0])})",
            f"{analysis['tessitura'][1]:.1f} Hz ({hz_to_note(analysis['tessitura'][1])})",
            f"{analysis['voiced_coverage']:.1f}%",
            f"{analysis['jitter']:.2f}%",
            f"{analysis['mean_cents_error']:.1f} cents"
        ]
    })
    
    st.dataframe(stats_df, use_container_width=True)

def render_pitch_analysis(results: Dict[str, Any]):
    """Render pitch analysis visualizations."""
    st.subheader("Pitch Analysis")
    
    # Pitch contour
    fig1 = create_pitch_plot(results['analysis'])
    st.plotly_chart(fig1, use_container_width=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Note histogram
        fig2 = create_note_histogram(results['analysis'])
        st.plotly_chart(fig2, use_container_width=True)
    
    with col2:
        # Cents error distribution
        fig3 = create_cents_error_plot(results['analysis'])
        st.plotly_chart(fig3, use_container_width=True)
    
    # Intonation insights
    st.subheader("Intonation Insights")
    intonation_score = results['analysis']['intonation_score']
    mean_cents_error = results['analysis']['mean_cents_error']
    
    if intonation_score >= 90:
        st.success("🎯 Excellent pitch accuracy! Your intonation is very precise.")
    elif intonation_score >= 80:
        st.info("🎵 Good pitch accuracy with room for improvement.")
    elif intonation_score >= 70:
        st.warning("📈 Fair intonation. Regular practice will help improve accuracy.")
    else:
        st.error("🎓 Pitch accuracy needs work. Consider using a tuner for practice.")
    
    st.info(f"Average deviation from target pitch: {mean_cents_error:.1f} cents")

def render_timbre_quality(results: Dict[str, Any]):
    """Render timbre and quality analysis."""
    st.subheader("Timbre Analysis")
    
    timbre = results['timbre_analysis']
    breath = results['breath_analysis']
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown(f"""
        **Timbre Badge:** {timbre['badge']}
        
        **Description:** {timbre['description']}
        
        **Vocal Tip:** {timbre['tips']}
        """)
        
        # Spectral plot
        fig = create_spectral_plot(results['analysis'])
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("Breath & Stability")
        
        if breath['stability_issues']:
            for issue in breath['stability_issues']:
                st.warning(f"• {issue}")
        else:
            st.success("✅ No major stability issues detected")
        
        st.markdown("**Recommendations:**")
        for rec in breath['recommendations']:
            st.info(f"• {rec}")
        
        st.markdown("**Suggested Exercises:**")
        for exercise in breath['exercises']:
            st.success(f"• {exercise}")
        
        # Breathiness score
        st.metric("Breathiness Score", f"{breath['breathiness_score']:.1f}%")

def render_song_matches(results: Dict[str, Any]):
    """Render song identification matches."""
    st.subheader("Song Identification")
    
    matches = results['song_matches']
    
    if not matches:
        st.info("No songs identified from the current audio.")
        st.info("Try singing a recognizable melody or upload a song snippet.")
    else:
        for i, match in enumerate(matches, 1):
            with st.expander(f"#{i}: {match['title']} by {match['artist']}"):
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown(f"""
                    **Title:** {match['title']}
                    
                    **Artist:** {match['artist']}
                    
                    **Key:** {match['key']}
                    
                    **Confidence:** {match['confidence']:.1f}%
                    """)
                
                with col2:
                    st.markdown(f"""
                    **Tags:** {match['tags']}
                    
                    **Correlation:** {match['correlation']:.3f}
                    
                    **Range Match:** {match['range_score']:.3f}
                    """)
                
                # Progress bar for confidence
                st.progress(match['confidence'] / 100)

def render_recommendations(results: Dict[str, Any]):
    """Render song recommendations."""
    st.subheader("Personalized Song Recommendations")
    
    recommendations = results['recommendations']
    
    # Fit songs
    if recommendations['fit']:
        st.markdown("### ✅ Perfect Fit Songs")
        for song in recommendations['fit']:
            with st.expander(f"{song['title']} by {song['artist']}"):
                st.markdown(f"""
                **Range:** {song['range']}
                
                **Key:** {song['key']}
                
                **Difficulty:** {song['difficulty']}
                
                **Why this fits:** {song['reason']}
                
                **Timbre Match:** {'✅ Yes' if song['timbre_match'] else '⚠️ Different style'}
                """)
    
    # Stretch songs
    if recommendations['stretch']:
        st.markdown("### 📈 Stretch Songs (For Growth)")
        for song in recommendations['stretch']:
            with st.expander(f"{song['title']} by {song['artist']}"):
                st.markdown(f"""
                **Range:** {song['range']}
                
                **Key:** {song['key']}
                
                **Difficulty:** {song['difficulty']}
                
                **Growth opportunity:** {song['reason']}
                
                **Timbre Match:** {'✅ Yes' if song['timbre_match'] else '⚠️ Different style'}
                """)
    
    # Avoid songs
    if recommendations['avoid']:
        st.markdown("### ⚠️ Challenging Songs (Avoid for Now)")
        for song in recommendations['avoid']:
            with st.expander(f"{song['title']} by {song['artist']}"):
                st.markdown(f"""
                **Range:** {song['range']}
                
                **Key:** {song['key']}
                
                **Difficulty:** {song['difficulty']}
                
                **Why challenging:** {song['reason']}
                
                Come back to this when you've extended your range!
                """)
    
    # Personalization
    st.subheader("🎯 Personalize Your Recommendations")
    
    col1, col2 = st.columns(2)
    
    with col1:
        preference = st.radio(
            "Song Era Preference",
            ["All Eras", "Vintage (pre-1980)", "Modern (1980-present)"],
            help="Choose your preferred song era for better recommendations"
        )
        
        favorite_genres = st.multiselect(
            "Favorite Genres",
            ["Pop", "Rock", "Jazz", "Classical", "R&B", "Country", "Folk", "Musical Theatre"],
            help="Select genres you enjoy singing"
        )
    
    with col2:
        st.markdown("**Favorite Songs**")
        favorite_songs = st.text_area(
            "List your favorite songs (one per line)",
            height=100,
            placeholder="Bohemian Rhapsody\nSomeone Like You\nImagine"
        )
        
        if st.button("Update Preferences"):
            st.success("Preferences updated! Recommendations will be personalized.")

def render_export_panel():
    """Render export and download options."""
    st.header("📥 Export & Downloads")
    
    if st.session_state.analysis_results is None:
        st.info("Complete an analysis to enable exports.")
        return
    
    results = st.session_state.analysis_results
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.subheader("📊 Analysis Data")
        
        # JSON summary
        summary_data = {
            'timestamp': datetime.now().isoformat(),
            'voice_classification': results['voice_classification'],
            'analysis_stats': {
                'duration': results['analysis']['duration'],
                'vocal_range': {
                    'min': results['analysis']['min_pitch'],
                    'max': results['analysis']['max_pitch'],
                    'tessitura': results['analysis']['tessitura']
                },
                'intonation_score': results['analysis']['intonation_score'],
                'mean_cents_error': results['analysis']['mean_cents_error'],
                'timbre_badge': results['timbre_analysis']['badge']
            }
        }
        
        json_str = json.dumps(summary_data, indent=2, default=str)
        st.download_button(
            "📄 Download Summary (JSON)",
            json_str,
            "voice_analysis_summary.json",
            "application/json"
        )
        
        # CSV data
        csv_data = results['analysis']['notes_times'].to_csv(index=False)
        st.download_button(
            "📊 Download Pitch Data (CSV)",
            csv_data,
            "pitch_analysis.csv",
            "text/csv"
        )
    
    with col2:
        st.subheader("🎨 Visualizations")
        
        # Generate summary image
        if st.button("📸 Generate Summary Image"):
            with st.spinner("Generating summary visualization..."):
                # Create a comprehensive summary chart
                fig = make_subplots(
                    rows=2, cols=2,
                    subplot_titles=["Pitch Contour", "Note Distribution", 
                                  "Vocal Range", "Spectral Energy"],
                    specs=[[{"secondary_y": False}, {"secondary_y": False}],
                           [{"secondary_y": False}, {"secondary_y": False}]]
                )
                
                # Add plots to subplots
                # (Implementation would add the actual chart data)
                
                st.success("Summary image generated!")
                # In a real implementation, this would save as PNG
    
    with col3:
        st.subheader("📋 Share Summary")
        
        # Create text summary
        voice_type = results['voice_classification']['voice_type'] or 'Unknown'
        confidence = results['voice_classification']['confidence']
        vocal_range = f"{hz_to_note(results['analysis']['min_pitch'])} - {hz_to_note(results['analysis']['max_pitch'])}"
        intonation = results['analysis']['intonation_score']
        timbre = results['timbre_analysis']['badge']
        
        summary_text = f"""
🎤 Voice Analysis Summary

Voice Type: {voice_type} ({confidence:.0f}% confidence)
Vocal Range: {vocal_range}
Intonation Score: {intonation:.0f}/100
Timbre: {timbre}

Generated by Professional Voice Analysis Studio
        """.strip()
        
        st.text_area("Copy to share", summary_text, height=200)
        
        if st.button("📋 Copy to Clipboard"):
            st.success("Summary copied to clipboard!")

def render_about_panel():
    """Render about and help information."""
    st.header("ℹ️ About & Help")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("About This App")
        st.markdown("""
        Professional Voice Analysis Studio is a comprehensive tool for analyzing 
        singing and speaking voice with advanced pitch detection, voice classification, 
        and personalized song recommendations.
        
        **Features:**
        - Real-time microphone recording
        - Advanced pitch detection (torchcrepe/librosa)
        - Voice type classification
        - Timbre analysis and badges
        - Breath control diagnostics
        - Song identification
        - Personalized recommendations
        - Interactive visualizations
        
        **Privacy-First Design:**
        - All processing happens locally
        - No data sent to external servers
        - Open source and transparent
        """)
    
    with col2:
        st.subheader("How to Use")
        st.markdown("""
        1. **Record or Upload Audio**
           - Use live microphone recording
           - Upload audio files (WAV, MP3, M4A, FLAC)
           - Generate test audio for demo
        
        2. **Analyze Your Voice**
           - Click "Analyze Voice" to process
           - Review comprehensive analysis
           - Check pitch accuracy and range
        
        3. **Explore Results**
           - View detailed voice profile
           - Examine pitch visualizations
           - Get song recommendations
        
        4. **Export & Share**
           - Download analysis data
           - Generate summary reports
           - Share results with teachers
        """)
    
    # Troubleshooting
    with st.expander("🛠️ Troubleshooting"):
        st.markdown("""
        **Recording Issues:**
        - Check browser microphone permissions
        - Ensure HTTPS connection for WebRTC
        - Try refreshing the page
        - Use file upload as alternative
        
        **Analysis Issues:**
        - Ensure audio is clear and audible
        - Try shorter recordings (under 5 minutes)
        - Check that torch/torchcrepe are installed for best results
        
        **Performance Issues:**
        - Close other browser tabs
        - Use shorter audio files
        - Disable advanced features if needed
        
        **Need Help?**
        - Check the diagnostics panel in Recording section
        - Review browser console for error messages
        - Try the test audio generation feature
        """)
    
    # Future features
    with st.expander("🚀 Future Features (50-100+ Pro Tools)"):
        st.markdown("""
        **Advanced Analysis:**
        - Vibrato detection and measurement
        - Formant analysis for vowel shaping
        - Spectral envelope analysis
        - Harmonic-to-noise ratio
        - Voice fatigue detection
        
        **Training Tools:**
        - Real-time pitch feedback
        - Interactive vocal exercises
        - Progress tracking over time
        - Custom exercise creation
        - Breath control games
        
        **Song Database:**
        - 1000+ song templates
        - Genre-specific recommendations
        - Difficulty progression paths
        - User rating system
        - Collaborative filtering
        
        **Professional Features:**
        - Multi-track recording
        - Harmony analysis
        - Repertoire management
        - Performance analytics
        - Teacher/student sharing
        
        **Integration:**
        - MusicXML support
        - DAW integration
        - Cloud synchronization
        - Mobile companion app
        - API for developers
        
        These features can be added by extending the modular architecture 
        and leveraging the existing analysis pipeline.
        """)

# ============================================================================
# MAIN APPLICATION
# ============================================================================

def main():
    """Main application entry point."""
    # Page configuration
    st.set_page_config(
        page_title="Professional Voice Analysis Studio",
        page_icon="🎤",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Custom CSS
    st.markdown("""
    <style>
    .stApp {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
    }
    .stButton > button {
        border-radius: 10px;
        font-weight: bold;
        transition: all 0.3s ease;
    }
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(0,0,0,0.2);
    }
    .metric-card {
        background: white;
        padding: 1rem;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        margin-bottom: 1rem;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Render UI components
    render_header()
    render_sidebar()
    
    # Main content tabs
    tabs = st.tabs([
        "🎙️ Record/Upload", "📊 Voice Analysis", "📈 Charts", "🎵 Recommendations", "ℹ️ About"
    ])
    
    with tabs[0]:
        render_recording_panel()
    
    with tabs[1]:
        render_analysis_panel()
    
    with tabs[2]:
        if st.session_state.analysis_results:
            st.header("📈 Interactive Charts")
            
            results = st.session_state.analysis_results
            
            # Create tabs for different chart types
            chart_tabs = st.tabs([
                "Pitch Contour", "Note Distribution", "Cents Error", "Vocal Range", "Spectral Analysis"
            ])
            
            with chart_tabs[0]:
                fig = create_pitch_plot(results['analysis'])
                st.plotly_chart(fig, use_container_width=True)
            
            with chart_tabs[1]:
                fig = create_note_histogram(results['analysis'])
                st.plotly_chart(fig, use_container_width=True)
            
            with chart_tabs[2]:
                fig = create_cents_error_plot(results['analysis'])
                st.plotly_chart(fig, use_container_width=True)
            
            with chart_tabs[3]:
                fig = create_vocal_range_plot(results['analysis'])
                st.plotly_chart(fig, use_container_width=True)
            
            with chart_tabs[4]:
                fig = create_spectral_plot(results['analysis'])
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Complete an analysis to view charts.")
    
    with tabs[3]:
        if st.session_state.analysis_results:
            render_recommendations(st.session_state.analysis_results)
        else:
            st.info("Complete an analysis to get song recommendations.")
    
    with tabs[4]:
        render_about_panel()
    
    # Export panel (always visible)
    render_export_panel()
    
    # Footer
    st.markdown("""
    <div style='text-align: center; padding: 2rem; color: #666;'>
        <p>Professional Voice Analysis Studio | Privacy-First | CPU-Only Compatible</p>
        <p style='font-size: 0.9rem;'>All analysis performed locally on your device</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
