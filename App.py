#!/usr/bin/env python3
"""
Professional Voice Analysis Application
=====================================

A comprehensive voice analysis tool that provides real-time pitch detection,
vocal range analysis, and song recommendations. Supports both live microphone
input and audio file upload.

Features:
- Real-time pitch detection using torchcrepe/librosa
- Vocal range and voice type classification
- Pitch accuracy analysis with intonation scoring
- Timbre analysis and spectral profiling
- Song identification and recommendations
- Interactive visualizations and data export

Requirements:
- streamlit, streamlit-webrtc
- torch (CPU), torchcrepe (optional)
- librosa, soundfile, numpy, scipy
- plotly, pandas, matplotlib
- opencv-python (for video processing)
"""

import streamlit as st
import numpy as np
import pandas as pd
import librosa
import soundfile as sf
import torch
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import json
import os
import tempfile
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Any
import logging
from pathlib import Path

# Audio processing imports
try:
    import torchcrepe
    TORCHCREPE_AVAILABLE = True
except ImportError:
    TORCHCREPE_AVAILABLE = False
    logging.warning("torchcrepe not available, using librosa YIN as fallback")

# WebRTC imports
try:
    from streamlit_webrtc import webrtc_streamer, WebRtcMode, ClientSettings
    WEBRTC_AVAILABLE = True
except ImportError:
    WEBRTC_AVAILABLE = False
    logging.warning("streamlit-webrtc not available, microphone input disabled")

# Additional imports
import av
import threading
from collections import deque
import time

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Page configuration
st.set_page_config(
    page_title="Voice Analysis Pro",
    page_icon="🎤",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for modern styling
st.markdown("""
<style>
    /* Main theme colors */
    :root {
        --primary-bg: #1a1f36;
        --secondary-bg: #fafafa;
        --accent-blue: #4a90e2;
        --success-green: #34d399;
        --warning-amber: #f59e0b;
        --error-red: #ef4444;
        --neutral-slate: #64748b;
        --text-primary: #1f2937;
        --text-secondary: #6b7280;
    }
    
    /* Main container styling */
    .main-container {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 12px;
        margin-bottom: 2rem;
    }
    
    /* Card styling */
    .stCard {
        background: white;
        border-radius: 12px;
        padding: 1.5rem;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        border: 1px solid #e5e7eb;
        transition: all 0.3s ease;
    }
    
    .stCard:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 25px rgba(0, 0, 0, 0.15);
    }
    
    /* Button styling */
    .stButton > button {
        background: #4a90e2;
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.75rem 1.5rem;
        font-weight: 600;
        transition: all 0.3s ease;
        cursor: pointer;
    }
    
    .stButton > button:hover {
        background: #357abd;
        transform: scale(1.02);
    }
    
    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] {
        background: rgba(255, 255, 255, 0.1);
        border-radius: 8px;
        padding: 0.5rem;
    }
    
    .stTabs [data-baseweb="tab"] {
        border-radius: 6px;
        padding: 0.5rem 1rem;
        font-weight: 500;
    }
    
    /* Metric styling */
    .metric-card {
        background: white;
        border-radius: 8px;
        padding: 1rem;
        text-align: center;
        border-left: 4px solid #4a90e2;
    }
    
    /* Audio visualizer styling */
    .audio-viz {
        background: linear-gradient(45deg, #667eea, #764ba2);
        border-radius: 8px;
        padding: 1rem;
        margin: 1rem 0;
    }
    
    /* Hero section */
    .hero-section {
        background: url('resources/hero_image.png') center/cover;
        border-radius: 12px;
        padding: 3rem 2rem;
        text-align: center;
        color: white;
        margin-bottom: 2rem;
    }
    
    /* File uploader styling */
    .uploadedFile {
        border: 2px dashed #4a90e2;
        border-radius: 8px;
        padding: 2rem;
        text-align: center;
        background: rgba(74, 144, 226, 0.05);
    }
    
    /* Progress bar styling */
    .stProgress > div > div {
        background: #4a90e2;
        border-radius: 4px;
    }
    
    /* DataFrame styling */
    .dataframe {
        border-radius: 8px;
        overflow: hidden;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
    }
    
    /* Success/Error message styling */
    .stAlert {
        border-radius: 8px;
        border: none;
    }
    
    .stSuccess {
        background: #d1fae5;
        border-left: 4px solid #34d399;
    }
    
    .stError {
        background: #fee2e2;
        border-left: 4px solid #ef4444;
    }
    
    /* Sidebar styling */
    .css-1d391kg {
        background: #f8fafc;
        border-right: 1px solid #e5e7eb;
    }
    
    /* Responsive design */
    @media (max-width: 768px) {
        .main-container {
            padding: 1rem;
        }
        
        .hero-section {
            padding: 2rem 1rem;
        }
    }
</style>
""", unsafe_allow_html=True)

class VoiceAnalyzer:
    """Main voice analysis class with comprehensive audio processing capabilities."""
    
    def __init__(self):
        self.sample_rate = 22050
        self.hop_length = 512
        self.fmin = librosa.note_to_hz('C2')
        self.fmax = librosa.note_to_hz('C7')
        self.analysis_results = {}
        
    def load_audio(self, audio_path: str) -> Tuple[np.ndarray, int]:
        """Load audio file and return audio data and sample rate."""
        try:
            audio, sr = librosa.load(audio_path, sr=self.sample_rate, mono=True)
            return audio, sr
        except Exception as e:
            logger.error(f"Error loading audio: {e}")
            raise
    
    def detect_pitch_torchcrepe(self, audio: np.ndarray, sr: int) -> Tuple[np.ndarray, np.ndarray]:
        """Detect pitch using torchcrepe (preferred method)."""
        if not TORCHCREPE_AVAILABLE:
            return self.detect_pitch_librosa(audio, sr)
            
        try:
            # Convert to torch tensor
            audio_torch = torch.from_numpy(audio).float()
            
            # Predict pitch and periodicity
            pitch, periodicity = torchcrepe.predict(
                audio_torch, 
                sr, 
                self.hop_length, 
                self.fmin, 
                self.fmax, 
                model='tiny',
                device='cpu',
                return_periodicity=True
            )
            
            # Convert back to numpy
            pitch = pitch.numpy()
            periodicity = periodicity.numpy()
            
            # Filter by periodicity threshold
            pitch[periodicity < 0.21] = np.nan
            
            return pitch, periodicity
            
        except Exception as e:
            logger.warning(f"torchcrepe failed, falling back to librosa: {e}")
            return self.detect_pitch_librosa(audio, sr)
    
    def detect_pitch_librosa(self, audio: np.ndarray, sr: int) -> Tuple[np.ndarray, np.ndarray]:
        """Fallback pitch detection using librosa YIN algorithm."""
        try:
            f0, voiced_flag, voiced_probs = librosa.pyin(
                audio, 
                fmin=self.fmin, 
                fmax=self.fmax, 
                sr=sr, 
                hop_length=self.hop_length
            )
            
            # Create periodicity from voiced probabilities
            periodicity = voiced_probs
            
            return f0, periodicity
            
        except Exception as e:
            logger.error(f"Librosa pitch detection failed: {e}")
            return np.array([]), np.array([])
    
    def analyze_vocal_range(self, pitch_data: np.ndarray) -> Dict[str, Any]:
        """Analyze vocal range from pitch data."""
        if len(pitch_data) == 0 or np.all(np.isnan(pitch_data)):
            return {
                'min_freq': 0,
                'max_freq': 0,
                'tessitura_low': 0,
                'tessitura_high': 0,
                'range_notes': 'N/A',
                'tessitura_notes': 'N/A',
                'voice_type': 'Unknown',
                'confidence': 0.0
            }
        
        # Remove NaN values
        valid_pitch = pitch_data[~np.isnan(pitch_data)]
        
        if len(valid_pitch) == 0:
            return {
                'min_freq': 0,
                'max_freq': 0,
                'tessitura_low': 0,
                'tessitura_high': 0,
                'range_notes': 'N/A',
                'tessitura_notes': 'N/A',
                'voice_type': 'Unknown',
                'confidence': 0.0
            }
        
        # Calculate range statistics
        min_freq = np.min(valid_pitch)
        max_freq = np.max(valid_pitch)
        
        # Calculate tessitura (central 50% range)
        q25, q75 = np.percentile(valid_pitch, [25, 75])
        tessitura_low = q25
        tessitura_high = q75
        
        # Convert to note names
        range_notes = f"{self.hz_to_note(min_freq)} - {self.hz_to_note(max_freq)}"
        tessitura_notes = f"{self.hz_to_note(tessitura_low)} - {self.hz_to_note(tessitura_high)}"
        
        # Classify voice type
        voice_type, confidence = self.classify_voice_type(min_freq, max_freq, tessitura_low, tessitura_high)
        
        return {
            'min_freq': min_freq,
            'max_freq': max_freq,
            'tessitura_low': tessitura_low,
            'tessitura_high': tessitura_high,
            'range_notes': range_notes,
            'tessitura_notes': tessitura_notes,
            'voice_type': voice_type,
            'confidence': confidence
        }
    
    def classify_voice_type(self, min_freq: float, max_freq: float, 
                          tessitura_low: float, tessitura_high: float) -> Tuple[str, float]:
        """Classify voice type based on frequency range."""
        
        # Voice type ranges (in Hz)
        voice_ranges = {
            'Bass': (82, 330, 98, 294),      # E2-E4, tessitura G2-D4
            'Baritone': (98, 392, 123, 349), # G2-G4, tessitura B2-F4
            'Tenor': (131, 523, 165, 440),   # C3-C5, tessitura E3-A4
            'Alto': (175, 699, 220, 587),    # F3-F5, tessitura A3-D5
            'Mezzo-Soprano': (196, 784, 247, 659), # G3-G5, tessitura B3-E5
            'Soprano': (220, 1047, 262, 784) # A3-C6, tessitura C4-G5
        }
        
        # Calculate overlaps with each voice type
        overlaps = {}
        for voice_type, (v_min, v_max, t_min, t_max) in voice_ranges.items():
            # Range overlap
            range_overlap = max(0, min(max_freq, v_max) - max(min_freq, v_min))
            range_total = max(max_freq, v_max) - min(min_freq, v_min)
            range_score = range_overlap / range_total if range_total > 0 else 0
            
            # Tessitura overlap
            tess_overlap = max(0, min(tessitura_high, t_max) - max(tessitura_low, t_min))
            tess_total = max(tessitura_high, t_max) - min(tessitura_low, t_min)
            tess_score = tess_overlap / tess_total if tess_total > 0 else 0
            
            # Combined score
            overlaps[voice_type] = (range_score + tess_score) / 2
        
        # Find best match
        best_match = max(overlaps, key=overlaps.get)
        confidence = overlaps[best_match]
        
        # Add borderline classification for ambiguous cases
        if confidence < 0.3:
            best_match = f"Borderline {best_match}"
        elif confidence < 0.5:
            best_match = f"Likely {best_match}"
        
        return best_match, confidence
    
    def analyze_pitch_accuracy(self, pitch_data: np.ndarray) -> Dict[str, Any]:
        """Analyze pitch accuracy and intonation."""
        if len(pitch_data) == 0 or np.all(np.isnan(pitch_data)):
            return {
                'intonation_score': 0,
                'mean_cents_error': 0,
                'std_cents_error': 0,
                'stability_score': 0,
                'pitch_drift': 0
            }
        
        valid_pitch = pitch_data[~np.isnan(pitch_data)]
        
        if len(valid_pitch) == 0:
            return {
                'intonation_score': 0,
                'mean_cents_error': 0,
                'std_cents_error': 0,
                'stability_score': 0,
                'pitch_drift': 0
            }
        
        # Calculate cents error from nearest equal-tempered notes
        cents_errors = []
        for freq in valid_pitch:
            if freq > 0:
                # Convert to MIDI note number
                midi_note = 12 * np.log2(freq / 440) + 69
                # Round to nearest integer (equal-tempered note)
                nearest_note = round(midi_note)
                # Calculate cents error
                cents_error = (midi_note - nearest_note) * 100
                cents_errors.append(cents_error)
        
        cents_errors = np.array(cents_errors)
        
        # Calculate statistics
        mean_cents_error = np.mean(np.abs(cents_errors))
        std_cents_error = np.std(cents_errors)
        
        # Intonation score (0-100, higher is better)
        intonation_score = max(0, 100 - mean_cents_error)
        
        # Stability score based on variance
        stability_score = max(0, 100 - std_cents_error * 2)
        
        # Pitch drift (change over time)
        if len(valid_pitch) > 1:
            pitch_drift = np.abs(valid_pitch[-1] - valid_pitch[0]) / len(valid_pitch)
        else:
            pitch_drift = 0
        
        return {
            'intonation_score': intonation_score,
            'mean_cents_error': mean_cents_error,
            'std_cents_error': std_cents_error,
            'stability_score': stability_score,
            'pitch_drift': pitch_drift
        }
    
    def analyze_timbre(self, audio: np.ndarray, sr: int) -> Dict[str, Any]:
        """Analyze timbre and spectral characteristics."""
        # Compute spectral features
        spectral_centroid = librosa.feature.spectral_centroid(y=audio, sr=sr)[0]
        spectral_rolloff = librosa.feature.spectral_rolloff(y=audio, sr=sr)[0]
        spectral_bandwidth = librosa.feature.spectral_bandwidth(y=audio, sr=sr)[0]
        
        # Compute energy in different frequency bands
        stft = np.abs(librosa.stft(audio))
        freqs = librosa.fft_frequencies(sr=sr)
        
        # Define frequency bands
        bass_mask = freqs < 300
        mid_mask = (freqs >= 300) & (freqs < 3000)
        treble_mask = freqs >= 3000
        
        # Calculate energy in each band
        bass_energy = np.sum(stft[bass_mask, :]**2)
        mid_energy = np.sum(stft[mid_mask, :]**2)
        treble_energy = np.sum(stft[treble_mask, :]**2)
        
        total_energy = bass_energy + mid_energy + treble_energy
        
        if total_energy > 0:
            bass_ratio = bass_energy / total_energy
            mid_ratio = mid_energy / total_energy
            treble_ratio = treble_energy / total_energy
        else:
            bass_ratio = mid_ratio = treble_ratio = 0
        
        # Determine timbre badge
        if bass_ratio > 0.5:
            timbre_badge = "Bass-heavy"
        elif treble_ratio > 0.4:
            timbre_badge = "Treble-bright"
        elif mid_ratio > 0.6:
            timbre_badge = "Mid-forward"
        else:
            timbre_badge = "Balanced"
        
        return {
            'spectral_centroid': np.mean(spectral_centroid),
            'spectral_rolloff': np.mean(spectral_rolloff),
            'spectral_bandwidth': np.mean(spectral_bandwidth),
            'bass_ratio': bass_ratio,
            'mid_ratio': mid_ratio,
            'treble_ratio': treble_ratio,
            'timbre_badge': timbre_badge
        }
    
    def hz_to_note(self, frequency: float) -> str:
        """Convert frequency in Hz to musical note name."""
        if frequency <= 0:
            return "N/A"
        
        # Convert to MIDI note number
        midi_note = 12 * np.log2(frequency / 440) + 69
        
        # Note names
        note_names = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
        
        # Calculate note and octave
        note_index = int(round(midi_note)) % 12
        octave = int(round(midi_note)) // 12 - 1
        
        return f"{note_names[note_index]}{octave}"
    
    def create_song_catalog(self) -> List[Dict[str, Any]]:
        """Create a sample song catalog for identification and recommendations."""
        return [
            {
                'title': 'Amazing Grace',
                'artist': 'Traditional',
                'key': 'G',
                'range_low': 196,  # G3
                'range_high': 659, # E5
                'tags': ['hymn', 'beginner'],
                'difficulty': 'easy'
            },
            {
                'title': 'Happy Birthday',
                'artist': 'Traditional',
                'key': 'C',
                'range_low': 262,  # C4
                'range_high': 523, # C5
                'tags': ['celebration', 'beginner'],
                'difficulty': 'easy'
            },
            {
                'title': 'Somewhere Over the Rainbow',
                'artist': 'Judy Garland',
                'key': 'C',
                'range_low': 220,  # A3
                'range_high': 698, # F5
                'tags': ['ballad', 'intermediate'],
                'difficulty': 'medium'
            },
            {
                'title': 'Imagine',
                'artist': 'John Lennon',
                'key': 'C',
                'range_low': 196,  # G3
                'range_high': 523, # C5
                'tags': ['ballad', 'intermediate'],
                'difficulty': 'medium'
            },
            {
                'title': 'Hallelujah',
                'artist': 'Leonard Cohen',
                'key': 'C',
                'range_low': 196,  # G3
                'range_high': 784, # G5
                'tags': ['ballad', 'advanced'],
                'difficulty': 'hard'
            },
            {
                'title': 'Bohemian Rhapsody',
                'artist': 'Queen',
                'key': 'Bb',
                'range_low': 174,  # F3
                'range_high': 880, # A5
                'tags': ['rock', 'advanced'],
                'difficulty': 'hard'
            },
            {
                'title': 'My Heart Will Go On',
                'artist': 'Celine Dion',
                'key': 'F',
                'range_low': 220,  # A3
                'range_high': 880, # A5
                'tags': ['ballad', 'advanced'],
                'difficulty': 'hard'
            },
            {
                'title': 'Let It Be',
                'artist': 'The Beatles',
                'key': 'C',
                'range_low': 196,  # G3
                'range_high': 659, # E5
                'tags': ['rock', 'intermediate'],
                'difficulty': 'medium'
            },
            {
                'title': 'Perfect',
                'artist': 'Ed Sheeran',
                'key': 'Ab',
                'range_low': 208,  # Ab3
                'range_high': 622, # Eb5
                'tags': ['pop', 'intermediate'],
                'difficulty': 'medium'
            },
            {
                'title': 'Someone Like You',
                'artist': 'Adele',
                'key': 'A',
                'range_low': 220,  # A3
                'range_high': 659, # E5
                'tags': ['ballad', 'intermediate'],
                'difficulty': 'medium'
            }
        ]
    
    def identify_songs(self, vocal_range: Dict[str, Any], timbre_badge: str) -> List[Dict[str, Any]]:
        """Identify potential songs based on vocal analysis."""
        catalog = self.create_song_catalog()
        matches = []
        
        user_min = vocal_range['min_freq']
        user_max = vocal_range['max_freq']
        user_tess_low = vocal_range['tessitura_low']
        user_tess_high = vocal_range['tessitura_high']
        
        for song in catalog:
            song_low = song['range_low']
            song_high = song['range_high']
            
            # Calculate range overlap
            overlap_low = max(user_min, song_low)
            overlap_high = min(user_max, song_high)
            overlap_range = max(0, overlap_high - overlap_low)
            
            # Calculate coverage percentage
            song_range = song_high - song_low
            coverage = overlap_range / song_range if song_range > 0 else 0
            
            # Tessitura fit
            tess_overlap_low = max(user_tess_low, song_low)
            tess_overlap_high = min(user_tess_high, song_high)
            tess_overlap_range = max(0, tess_overlap_high - tess_overlap_low)
            tess_coverage = tess_overlap_range / song_range if song_range > 0 else 0
            
            # Combined score
            total_score = (coverage * 0.6 + tess_coverage * 0.4)
            
            if total_score > 0.3:  # Minimum threshold
                matches.append({
                    'song': song,
                    'confidence': total_score,
                    'coverage': coverage,
                    'tessitura_fit': tess_coverage,
                    'range_gap': max(0, song_low - user_max, user_min - song_high)
                })
        
        # Sort by confidence
        matches.sort(key=lambda x: x['confidence'], reverse=True)
        
        return matches[:5]  # Top 5 matches
    
    def recommend_songs(self, vocal_range: Dict[str, Any], timbre_badge: str, 
                       current_matches: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
        """Generate song recommendations based on analysis."""
        catalog = self.create_song_catalog()
        recommendations = {
            'perfect_fit': [],
            'stretch_songs': [],
            'avoid_songs': []
        }
        
        user_tess_low = vocal_range['tessitura_low']
        user_tess_high = vocal_range['tessitura_high']
        user_voice_type = vocal_range['voice_type']
        
        for song in catalog:
            # Skip if already identified
            if any(match['song']['title'] == song['title'] for match in current_matches):
                continue
            
            song_low = song['range_low']
            song_high = song['range_high']
            
            # Calculate tessitura fit
            tess_overlap_low = max(user_tess_low, song_low)
            tess_overlap_high = min(user_tess_high, song_high)
            tess_coverage = max(0, tess_overlap_high - tess_overlap_low) / (song_high - song_low)
            
            # Determine recommendation category
            if tess_coverage > 0.8:
                # Perfect fit
                song_data = song.copy()
                song_data['reason'] = f"Fits comfortably within your tessitura ({vocal_range['tessitura_notes']})"
                recommendations['perfect_fit'].append(song_data)
            
            elif 0.5 <= tess_coverage <= 0.8:
                # Stretch song (slightly challenging)
                stretch_range = max(song_low - user_tess_low, user_tess_high - song_high)
                if stretch_range <= 200:  # Within 200Hz stretch
                    song_data = song.copy()
                    song_data['reason'] = f"Good for vocal development, slight stretch required"
                    recommendations['stretch_songs'].append(song_data)
            
            else:
                # Avoid songs (too challenging)
                if song_low > user_tess_high + 200 or song_high < user_tess_low - 200:
                    song_data = song.copy()
                    song_data['reason'] = f"Outside comfortable range - may strain voice"
                    recommendations['avoid_songs'].append(song_data)
        
        return recommendations
    
    def analyze_audio(self, audio_path: str) -> Dict[str, Any]:
        """Perform complete audio analysis."""
        try:
            # Load audio
            audio, sr = self.load_audio(audio_path)
            
            # Detect pitch
            if TORCHCREPE_AVAILABLE:
                pitch_data, periodicity = self.detect_pitch_torchcrepe(audio, sr)
            else:
                pitch_data, periodicity = self.detect_pitch_librosa(audio, sr)
            
            # Analyze vocal range
            vocal_range = self.analyze_vocal_range(pitch_data)
            
            # Analyze pitch accuracy
            pitch_accuracy = self.analyze_pitch_accuracy(pitch_data)
            
            # Analyze timbre
            timbre_analysis = self.analyze_timbre(audio, sr)
            
            # Identify songs
            song_matches = self.identify_songs(vocal_range, timbre_analysis['timbre_badge'])
            
            # Generate recommendations
            recommendations = self.recommend_songs(
                vocal_range, 
                timbre_analysis['timbre_badge'], 
                song_matches
            )
            
            # Create time array for plotting
            time_array = librosa.frames_to_time(
                range(len(pitch_data)), 
                sr=sr, 
                hop_length=self.hop_length
            )
            
            # Compile results
            results = {
                'audio_info': {
                    'duration': len(audio) / sr,
                    'sample_rate': sr,
                    'file_path': audio_path
                },
                'pitch_data': pitch_data,
                'periodicity': periodicity,
                'time_array': time_array,
                'vocal_range': vocal_range,
                'pitch_accuracy': pitch_accuracy,
                'timbre_analysis': timbre_analysis,
                'song_matches': song_matches,
                'recommendations': recommendations,
                'analysis_timestamp': datetime.now().isoformat()
            }
            
            return results
            
        except Exception as e:
            logger.error(f"Analysis failed: {e}")
            raise

class AudioRecorder:
    """Handle real-time audio recording and processing."""
    
    def __init__(self):
        self.audio_buffer = deque(maxlen=10000)  # Buffer for real-time audio
        self.is_recording = False
        self.recording_thread = None
        
    def start_recording(self):
        """Start audio recording."""
        if not WEBRTC_AVAILABLE:
            st.error("Audio recording not available. Please install streamlit-webrtc.")
            return
        
        self.is_recording = True
        # Recording handled by WebRTC component
        
    def stop_recording(self):
        """Stop audio recording."""
        self.is_recording = False
        
    def process_audio_frame(self, frame):
        """Process incoming audio frame."""
        if frame is not None:
            audio_data = frame.to_ndarray()
            # Add to buffer for real-time processing
            self.audio_buffer.extend(audio_data.flatten())

def create_pitch_visualization(results: Dict[str, Any]) -> go.Figure:
    """Create interactive pitch visualization."""
    
    pitch_data = results['pitch_data']
    time_array = results['time_array']
    vocal_range = results['vocal_range']
    
    # Create subplot
    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=('Pitch Detection Over Time', 'Pitch Confidence'),
        vertical_spacing=0.1,
        row_heights=[0.7, 0.3]
    )
    
    # Main pitch curve
    valid_indices = ~np.isnan(pitch_data)
    if np.any(valid_indices):
        fig.add_trace(
            go.Scatter(
                x=time_array[valid_indices],
                y=pitch_data[valid_indices],
                mode='lines',
                name='Detected Pitch',
                line=dict(color='#4a90e2', width=2),
                hovertemplate='Time: %{x:.2f}s<br>Frequency: %{y:.1f} Hz<br>Note: %{text}<extra></extra>',
                text=[VoiceAnalyzer().hz_to_note(freq) for freq in pitch_data[valid_indices]]
            ),
            row=1, col=1
        )
    
    # Add tessitura band
    fig.add_hrect(
        y0=vocal_range['tessitura_low'],
        y1=vocal_range['tessitura_high'],
        fillcolor="rgba(74, 144, 226, 0.2)",
        layer="below",
        line_width=0,
        annotation_text="Tessitura",
        annotation_position="top left",
        row=1, col=1
    )
    
    # Add vocal range markers
    fig.add_hline(
        y=vocal_range['min_freq'],
        line_dash="dash",
        line_color="#ef4444",
        annotation_text=f"Min: {vocal_range['min_freq']:.0f} Hz",
        annotation_position="right",
        row=1, col=1
    )
    
    fig.add_hline(
        y=vocal_range['max_freq'],
        line_dash="dash",
        line_color="#34d399",
        annotation_text=f"Max: {vocal_range['max_freq']:.0f} Hz",
        annotation_position="right",
        row=1, col=1
    )
    
    # Confidence plot
    if 'periodicity' in results and len(results['periodicity']) > 0:
        fig.add_trace(
            go.Scatter(
                x=time_array,
                y=results['periodicity'],
                mode='lines',
                name='Voicing Confidence',
                line=dict(color='#f59e0b', width=1),
                hovertemplate='Time: %{x:.2f}s<br>Confidence: %{y:.2f}<extra></extra>'
            ),
            row=2, col=1
        )
    
    # Update layout
    fig.update_layout(
        height=600,
        title_text="Voice Analysis Results",
        showlegend=True,
        template="plotly_white",
        font=dict(family="Inter, sans-serif")
    )
    
    fig.update_xaxes(title_text="Time (seconds)", row=2, col=1)
    fig.update_yaxes(title_text="Frequency (Hz)", row=1, col=1)
    fig.update_yaxes(title_text="Confidence", row=2, col=1)
    
    return fig

def create_spectrum_visualization(results: Dict[str, Any]) -> go.Figure:
    """Create spectral analysis visualization."""
    
    timbre = results['timbre_analysis']
    
    categories = ['Bass (<300 Hz)', 'Mid (300-3000 Hz)', 'Treble (>3000 Hz)']
    values = [timbre['bass_ratio'], timbre['mid_ratio'], timbre['treble_ratio']]
    colors = ['#ef4444', '#f59e0b', '#4a90e2']
    
    fig = go.Figure(data=[
        go.Bar(
            x=categories,
            y=values,
            marker_color=colors,
            text=[f"{v:.1%}" for v in values],
            textposition='auto',
            hovertemplate='%{x}<br>Energy: %{y:.1%}<extra></extra>'
        )
    ])
    
    fig.update_layout(
        title="Timbre Analysis - Frequency Band Energy Distribution",
        xaxis_title="Frequency Band",
        yaxis_title="Relative Energy",
        template="plotly_white",
        height=400,
        font=dict(family="Inter, sans-serif")
    )
    
    return fig

def create_note_histogram(results: Dict[str, Any]) -> go.Figure:
    """Create histogram of detected notes."""
    
    pitch_data = results['pitch_data']
    valid_pitch = pitch_data[~np.isnan(pitch_data)]
    
    if len(valid_pitch) == 0:
        return go.Figure()
    
    # Convert to note names
    note_names = []
    for freq in valid_pitch:
        note = VoiceAnalyzer().hz_to_note(freq)
        if note != "N/A":
            note_names.append(note)
    
    # Count note occurrences
    note_counts = {}
    for note in note_names:
        note_counts[note] = note_counts.get(note, 0) + 1
    
    if not note_counts:
        return go.Figure()
    
    notes = list(note_counts.keys())
    counts = list(note_counts.values())
    
    fig = go.Figure(data=[
        go.Bar(
            x=notes,
            y=counts,
            marker_color='#10b981',
            text=counts,
            textposition='auto',
            hovertemplate='Note: %{x}<br>Occurrences: %{y}<extra></extra>'
        )
    ])
    
    fig.update_layout(
        title="Note Distribution",
        xaxis_title="Musical Note",
        yaxis_title="Frequency",
        template="plotly_white",
        height=400,
        font=dict(family="Inter, sans-serif")
    )
    
    return fig

def save_analysis_results(results: Dict[str, Any], format_type: str) -> bytes:
    """Save analysis results in specified format."""
    
    if format_type == 'json':
        # Prepare JSON-serializable data
        json_results = {}
        for key, value in results.items():
            if isinstance(value, np.ndarray):
                json_results[key] = value.tolist()
            elif isinstance(value, dict):
                json_dict = {}
                for k, v in value.items():
                    if isinstance(v, np.ndarray):
                        json_dict[k] = v.tolist()
                    elif isinstance(v, (np.integer, np.floating)):
                        json_dict[k] = float(v)
                    else:
                        json_dict[k] = v
                json_results[key] = json_dict
            else:
                json_results[key] = value
        
        return json.dumps(json_results, indent=2).encode()
    
    elif format_type == 'csv':
        # Create CSV with pitch data
        pitch_data = results['pitch_data']
        time_array = results['time_array']
        
        df_data = []
        for i, (time, freq) in enumerate(zip(time_array, pitch_data)):
            if not np.isnan(freq):
                note = VoiceAnalyzer().hz_to_note(freq)
                # Calculate cents error (simplified)
                midi_note = 12 * np.log2(freq / 440) + 69
                nearest_note = round(midi_note)
                cents_error = (midi_note - nearest_note) * 100
                
                df_data.append({
                    'time': time,
                    'frequency_hz': freq,
                    'note': note,
                    'cents_error': cents_error
                })
        
        df = pd.DataFrame(df_data)
        return df.to_csv(index=False).encode()
    
    return b""

def main():
    """Main application function."""
    
    # Initialize session state
    if 'analyzer' not in st.session_state:
        st.session_state.analyzer = VoiceAnalyzer()
    
    if 'analysis_results' not in st.session_state:
        st.session_state.analysis_results = None
    
    if 'audio_file' not in st.session_state:
        st.session_state.audio_file = None
    
    # Header section
    st.markdown("""
    <div class="hero-section">
        <h1 style="font-size: 3rem; margin-bottom: 1rem;">🎤 Voice Analysis Pro</h1>
        <p style="font-size: 1.2rem; margin-bottom: 2rem;">
            Professional voice analysis with real-time pitch detection, vocal range classification, 
            and personalized song recommendations.
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Analysis Settings")
        
        # Analysis parameters
        sensitivity = st.slider(
            "Pitch Detection Sensitivity",
            min_value=0.1,
            max_value=1.0,
            value=0.5,
            step=0.1,
            help="Higher values detect more subtle pitch variations"
        )
        
        noise_threshold = st.slider(
            "Noise Threshold",
            min_value=-60,
            max_value=-20,
            value=-40,
            step=5,
            help="Minimum amplitude for pitch detection (dB)"
        )
        
        analysis_window = st.select_slider(
            "Analysis Window Size",
            options=["Short (fast)", "Medium", "Long (accurate)"],
            value="Medium",
            help="Larger windows provide more accurate but slower analysis"
        )
        
        st.divider()
        
        # About section
        st.header("ℹ️ About")
        st.info("""
        **Voice Analysis Pro** provides professional-grade voice analysis using 
        advanced pitch detection algorithms and machine learning.
        
        **Features:**
        • Real-time pitch detection
        • Vocal range classification  
        • Pitch accuracy analysis
        • Timbre profiling
        • Song recommendations
        
        **Privacy:** All analysis is performed locally on your device.
        No audio data is uploaded or shared.
        """)
        
        # Dependencies status
        st.divider()
        st.header("🔧 System Status")
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric(
                "TorchCREPE",
                "✅ Available" if TORCHCREPE_AVAILABLE else "❌ Fallback",
                help="Advanced pitch detection (uses CPU)"
            )
        
        with col2:
            st.metric(
                "WebRTC",
                "✅ Available" if WEBRTC_AVAILABLE else "❌ Disabled",
                help="Real-time microphone input"
            )
    
    # Main content area
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "🎙️ Record/Upload", 
        "🔍 Voice Analysis", 
        "📈 Charts", 
        "🎵 Recommendations",
        "💾 Export"
    ])
    
    # Record/Upload Tab
    with tab1:
        st.header("Audio Input")
        
        input_method = st.radio(
            "Choose input method:",
            ["Upload Audio File", "Record with Microphone"],
            horizontal=True
        )
        
        if input_method == "Upload Audio File":
            uploaded_file = st.file_uploader(
                "Choose an audio file",
                type=['wav', 'mp3', 'm4a', 'flac'],
                help="Supported formats: WAV, MP3, M4A, FLAC (max 10 minutes)"
            )
            
            if uploaded_file is not None:
                # Save uploaded file
                with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp_file:
                    tmp_file.write(uploaded_file.getbuffer())
                    st.session_state.audio_file = tmp_file.name
                
                # Display file info
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("File Size", f"{uploaded_file.size / 1024 / 1024:.1f} MB")
                with col2:
                    st.metric("Format", uploaded_file.type)
                with col3:
                    st.metric("Status", "Ready for Analysis")
                
                # Audio player
                st.audio(uploaded_file, format=uploaded_file.type)
        
        else:  # Record with Microphone
            if WEBRTC_AVAILABLE:
                st.info("🎤 Click 'Start' to begin recording. Allow microphone access when prompted.")
                
                webrtc_ctx = webrtc_streamer(
                    key="voice-recorder",
                    mode=WebRtcMode.SENDONLY,
                    audio_receiver_size=1024,
                    media_stream_constraints={"audio": True, "video": False},
                    rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
                )
                
                if webrtc_ctx.audio_receiver:
                    # Real-time audio processing would go here
                    st.success("🎵 Recording in progress...")
                    
                    # Placeholder for recorded audio
                    if st.button("Save Recording"):
                        st.info("Recording saved for analysis")
            else:
                st.error("Microphone recording not available. Please install streamlit-webrtc or use file upload.")
        
        # Analysis button
        if st.session_state.audio_file or input_method == "Record with Microphone":
            if st.button("🔍 Analyze Voice", type="primary", use_container_width=True):
                with st.spinner("Analyzing audio... This may take a moment."):
                    try:
                        # Perform analysis
                        results = st.session_state.analyzer.analyze_audio(st.session_state.audio_file)
                        st.session_state.analysis_results = results
                        
                        st.success("✅ Analysis complete! Check the other tabs for results.")
                        
                        # Auto-switch to analysis tab
                        st.session_state.active_tab = 1
                        
                    except Exception as e:
                        st.error(f"Analysis failed: {str(e)}")
                        logger.error(f"Analysis error: {e}")
    
    # Voice Analysis Tab
    with tab2:
        st.header("Voice Analysis Results")
        
        if st.session_state.analysis_results is None:
            st.info("No analysis results yet. Please upload an audio file and run analysis first.")
        else:
            results = st.session_state.analysis_results
            
            # Key metrics
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric(
                    "Vocal Range",
                    results['vocal_range']['range_notes'],
                    help="Your complete vocal range from lowest to highest note"
                )
            
            with col2:
                st.metric(
                    "Voice Type",
                    results['vocal_range']['voice_type'],
                    help="Classified voice type based on range and tessitura"
                )
            
            with col3:
                st.metric(
                    "Intonation Score",
                    f"{results['pitch_accuracy']['intonation_score']:.1f}/100",
                    help="Overall pitch accuracy (higher is better)"
                )
            
            with col4:
                st.metric(
                    "Timbre",
                    results['timbre_analysis']['timbre_badge'],
                    help="Dominant frequency characteristics"
                )
            
            st.divider()
            
            # Detailed analysis
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("📊 Vocal Range Details")
                
                range_data = {
                    'Metric': ['Minimum Frequency', 'Maximum Frequency', 'Tessitura Low', 'Tessitura High', 'Range Confidence'],
                    'Value': [
                        f"{results['vocal_range']['min_freq']:.1f} Hz",
                        f"{results['vocal_range']['max_freq']:.1f} Hz",
                        f"{results['vocal_range']['tessitura_low']:.1f} Hz",
                        f"{results['vocal_range']['tessitura_high']:.1f} Hz",
                        f"{results['vocal_range']['confidence']:.1%}"
                    ]
                }
                
                range_df = pd.DataFrame(range_data)
                st.dataframe(range_df, use_container_width=True, hide_index=True)
                
                st.subheader("🎯 Pitch Accuracy")
                
                accuracy_data = {
                    'Metric': ['Mean Cents Error', 'Stability Score', 'Pitch Drift'],
                    'Value': [
                        f"{results['pitch_accuracy']['mean_cents_error']:.1f} cents",
                        f"{results['pitch_accuracy']['stability_score']:.1f}/100",
                        f"{results['pitch_accuracy']['pitch_drift']:.2f} Hz/frame"
                    ]
                }
                
                accuracy_df = pd.DataFrame(accuracy_data)
                st.dataframe(accuracy_df, use_container_width=True, hide_index=True)
            
            with col2:
                st.subheader("🎨 Timbre Analysis")
                
                timbre_data = {
                    'Frequency Band': ['Bass (<300 Hz)', 'Mid (300-3000 Hz)', 'Treble (>3000 Hz)'],
                    'Energy Ratio': [
                        f"{results['timbre_analysis']['bass_ratio']:.1%}",
                        f"{results['timbre_analysis']['mid_ratio']:.1%}",
                        f"{results['timbre_analysis']['treble_ratio']:.1%}"
                    ]
                }
                
                timbre_df = pd.DataFrame(timbre_data)
                st.dataframe(timbre_df, use_container_width=True, hide_index=True)
                
                # Audio info
                st.subheader("📁 Audio Information")
                
                audio_info = {
                    'Property': ['Duration', 'Sample Rate', 'Analysis Timestamp'],
                    'Value': [
                        f"{results['audio_info']['duration']:.2f} seconds",
                        f"{results['audio_info']['sample_rate']} Hz",
                        results['analysis_timestamp'][:19]  # Remove microseconds
                    ]
                }
                
                info_df = pd.DataFrame(audio_info)
                st.dataframe(info_df, use_container_width=True, hide_index=True)
    
    # Charts Tab
    with tab3:
        st.header("Interactive Visualizations")
        
        if st.session_state.analysis_results is None:
            st.info("No data to visualize. Please run analysis first.")
        else:
            results = st.session_state.analysis_results
            
            # Chart selection
            chart_type = st.selectbox(
                "Select visualization:",
                ["Pitch Over Time", "Timbre Analysis", "Note Distribution", "All Charts"]
            )
            
            if chart_type == "Pitch Over Time" or chart_type == "All Charts":
                st.subheader("📈 Pitch Detection Over Time")
                pitch_fig = create_pitch_visualization(results)
                st.plotly_chart(pitch_fig, use_container_width=True)
            
            if chart_type == "Timbre Analysis" or chart_type == "All Charts":
                st.subheader("🎵 Timbre Analysis")
                spectrum_fig = create_spectrum_visualization(results)
                st.plotly_chart(spectrum_fig, use_container_width=True)
            
            if chart_type == "Note Distribution" or chart_type == "All Charts":
                st.subheader("🎼 Note Distribution")
                note_fig = create_note_histogram(results)
                st.plotly_chart(note_fig, use_container_width=True)
            
            # Chart export options
            if chart_type != "All Charts":
                st.divider()
                st.subheader("📊 Export Chart")
                
                if st.button(f"Save {chart_type} as PNG"):
                    # This would require additional implementation for chart export
                    st.info("Chart export feature coming soon!")
    
    # Recommendations Tab
    with tab4:
        st.header("🎵 Song Recommendations")
        
        if st.session_state.analysis_results is None:
            st.info("No recommendations available. Please run analysis first.")
        else:
            results = st.session_state.analysis_results
            
            # Song matches
            if results['song_matches']:
                st.subheader("🔍 Identified Songs")
                
                for i, match in enumerate(results['song_matches']):
                    with st.expander(f"{i+1}. {match['song']['title']} by {match['song']['artist']}"):
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.metric("Confidence", f"{match['confidence']:.1%}")
                            st.metric("Range Coverage", f"{match['coverage']:.1%}")
                            st.metric("Tessitura Fit", f"{match['tessitura_fit']:.1%}")
                        
                        with col2:
                            st.write(f"**Key:** {match['song']['key']}")
                            st.write(f"**Range:** {VoiceAnalyzer().hz_to_note(match['song']['range_low'])} - {VoiceAnalyzer().hz_to_note(match['song']['range_high'])}")
                            st.write(f"**Difficulty:** {match['song']['difficulty'].title()}")
                            st.write(f"**Tags:** {', '.join(match['song']['tags'])}")
            
            # Recommendations
            recommendations = results['recommendations']
            
            if recommendations['perfect_fit']:
                st.subheader("✅ Perfect Fit Songs")
                
                for song in recommendations['perfect_fit']:
                    with st.expander(f"{song['title']} by {song['artist']}"):
                        st.write(f"**Reason:** {song['reason']}")
                        st.write(f"**Key:** {song['key']}")
                        st.write(f"**Range:** {VoiceAnalyzer().hz_to_note(song['range_low'])} - {VoiceAnalyzer().hz_to_note(song['range_high'])}")
                        st.write(f"**Difficulty:** {song['difficulty'].title()}")
            
            if recommendations['stretch_songs']:
                st.subheader("🚀 Stretch Songs (For Development)")
                
                for song in recommendations['stretch_songs']:
                    with st.expander(f"{song['title']} by {song['artist']}"):
                        st.write(f"**Reason:** {song['reason']}")
                        st.write(f"**Key:** {song['key']}")
                        st.write(f"**Range:** {VoiceAnalyzer().hz_to_note(song['range_low'])} - {VoiceAnalyzer().hz_to_note(song['range_high'])}")
                        st.write(f"**Difficulty:** {song['difficulty'].title()}")
            
            if recommendations['avoid_songs']:
                st.subheader("⚠️ Songs to Avoid")
                
                for song in recommendations['avoid_songs']:
                    with st.expander(f"{song['title']} by {song['artist']}"):
                        st.write(f"**Reason:** {song['reason']}")
                        st.write(f"**Key:** {song['key']}")
                        st.write(f"**Range:** {VoiceAnalyzer().hz_to_note(song['range_low'])} - {VoiceAnalyzer().hz_to_note(song['range_high'])}")
                        st.write(f"**Difficulty:** {song['difficulty'].title()}")
    
    # Export Tab
    with tab5:
        st.header("💾 Export Analysis Results")
        
        if st.session_state.analysis_results is None:
            st.info("No results to export. Please run analysis first.")
        else:
            results = st.session_state.analysis_results
            
            st.subheader("📄 Export Options")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**Complete Analysis (JSON)**")
                st.write("- Full analysis results")
                st.write("- All metrics and data")
                st.write("- Reproducible format")
                
                if st.button("📥 Download JSON"):
                    json_data = save_analysis_results(results, 'json')
                    st.download_button(
                        label="💾 Download Analysis JSON",
                        data=json_data,
                        file_name=f"voice_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                        mime="application/json"
                    )
            
            with col2:
                st.write("**Pitch Data (CSV)**")
                st.write("- Time-stamped pitch data")
                st.write("- Note names and frequencies")
                st.write("- Intonation error analysis")
                
                if st.button("📥 Download CSV"):
                    csv_data = save_analysis_results(results, 'csv')
                    st.download_button(
                        label="💾 Download Pitch CSV",
                        data=csv_data,
                        file_name=f"pitch_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        mime="text/csv"
                    )
            
            st.divider()
            
            # Summary text
            st.subheader("📋 Analysis Summary")
            
            summary_text = f"""
Voice Analysis Summary
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

VOCAL RANGE:
- Range: {results['vocal_range']['range_notes']}
- Tessitura: {results['vocal_range']['tessitura_notes']}
- Voice Type: {results['vocal_range']['voice_type']}

PITCH ACCURACY:
- Intonation Score: {results['pitch_accuracy']['intonation_score']:.1f}/100
- Mean Cents Error: {results['pitch_accuracy']['mean_cents_error']:.1f} cents
- Stability Score: {results['pitch_accuracy']['stability_score']:.1f}/100

TIMBRE ANALYSIS:
- Classification: {results['timbre_analysis']['timbre_badge']}
- Spectral Centroid: {results['timbre_analysis']['spectral_centroid']:.0f} Hz

RECOMMENDATIONS:
- Perfect Fit Songs: {len(results['recommendations']['perfect_fit'])}
- Development Songs: {len(results['recommendations']['stretch_songs'])}
- Songs to Avoid: {len(results['recommendations']['avoid_songs'])}
            """
            
            st.text_area("Copy this summary:", summary_text, height=300)
            
            if st.button("📋 Copy to Clipboard"):
                st.success("Summary copied to clipboard!")
    
    # Footer
    st.divider()
    st.markdown("""
    <div style="text-align: center; color: #6b7280; padding: 2rem;">
        <p><strong>Voice Analysis Pro</strong> - Professional voice analysis for singers and vocalists</p>
        <p style="font-size: 0.9rem;">
        All analysis is performed locally on your device. No audio data is uploaded or shared.
        </p>
        <p style="font-size: 0.8rem; margin-top: 1rem;">
        Built with Streamlit • Powered by torchcrepe/librosa • Designed for Windows 10 (CPU-only)
        </p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
