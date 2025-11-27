"""
AUDIO MASTER PRO v6.0 - Modern Edition
- UI: Design moderno con pannelli flottanti e animazioni
- Player: Seek bar, volume, loop, A-B repeat, speed control
- DSP: Limiter, Stereo Enhancer, Reverb, De-Esser, Saturation
- EQ: 10 bande + Spectrum Analyzer real-time
"""

import sys
import os
import numpy as np
import librosa
import soundfile as sf
import torch
import warnings
import threading
from scipy import signal
from dataclasses import dataclass
from typing import Optional, Tuple
import time

# --- FIX NUMPY ---
if not hasattr(np, 'float'):
    np.float = float

# GUI Imports
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtGui import *

# Audio & Plotting
import sounddevice as sd
import matplotlib
matplotlib.use('Qt5Agg')
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from matplotlib.animation import FuncAnimation

# DSP & AI
import noisereduce as nr
from demucs.apply import apply_model

warnings.filterwarnings("ignore")

# ==================== MODERN STYLES ====================

MODERN_STYLE = """
QMainWindow {
    background: qlineargradient(x1:0, y1:0, x2:1, y2:1, 
        stop:0 #1a1a2e, stop:0.5 #16213e, stop:1 #0f3460);
}

QWidget#CentralWidget {
    background: transparent;
}

QGroupBox {
    background: rgba(255, 255, 255, 0.05);
    border: 1px solid rgba(255, 255, 255, 0.1);
    border-radius: 16px;
    margin-top: 20px;
    padding: 20px;
    font-weight: bold;
}

QGroupBox::title {
    subcontrol-origin: margin;
    left: 20px;
    padding: 0 10px;
    color: #00d4ff;
    font-size: 13px;
    font-weight: bold;
}

QPushButton {
    background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
        stop:0 rgba(255,255,255,0.1), stop:1 rgba(255,255,255,0.05));
    color: #ffffff;
    border: 1px solid rgba(255, 255, 255, 0.2);
    padding: 12px 24px;
    border-radius: 12px;
    font-weight: 600;
    font-size: 12px;
}

QPushButton:hover {
    background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
        stop:0 rgba(0,212,255,0.3), stop:1 rgba(0,212,255,0.1));
    border-color: #00d4ff;
}

QPushButton:pressed {
    background: rgba(0, 212, 255, 0.4);
}

QPushButton:disabled {
    background: rgba(255, 255, 255, 0.02);
    color: rgba(255, 255, 255, 0.3);
    border-color: rgba(255, 255, 255, 0.05);
}

QPushButton#AccentBtn {
    background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
        stop:0 #00d4ff, stop:1 #0099ff);
    border: none;
    color: #000;
    font-weight: bold;
}

QPushButton#AccentBtn:hover {
    background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
        stop:0 #33ddff, stop:1 #33aaff);
}

QPushButton#DangerBtn {
    background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
        stop:0 #ff6b6b, stop:1 #ee5a5a);
    border: none;
}

QPushButton#PlayBtn {
    background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
        stop:0 #00ff88, stop:1 #00cc6a);
    border: none;
    color: #000;
    font-size: 16px;
    min-width: 60px;
    min-height: 60px;
    border-radius: 30px;
}

QPushButton#PlayBtn:hover {
    background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
        stop:0 #33ff99, stop:1 #33dd7a);
}

QComboBox {
    background: rgba(255, 255, 255, 0.08);
    border: 1px solid rgba(255, 255, 255, 0.15);
    border-radius: 8px;
    padding: 8px 12px;
    color: #fff;
    font-size: 12px;
}

QComboBox::drop-down {
    border: none;
    width: 30px;
}

QComboBox::down-arrow {
    image: none;
    border-left: 5px solid transparent;
    border-right: 5px solid transparent;
    border-top: 6px solid #00d4ff;
    margin-right: 10px;
}

QComboBox QAbstractItemView {
    background: #1a1a2e;
    border: 1px solid rgba(255, 255, 255, 0.1);
    selection-background-color: rgba(0, 212, 255, 0.3);
    color: #fff;
}

QProgressBar {
    border: none;
    background: rgba(255, 255, 255, 0.1);
    height: 8px;
    border-radius: 4px;
}

QProgressBar::chunk {
    background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
        stop:0 #00d4ff, stop:1 #00ff88);
    border-radius: 4px;
}

QLabel {
    color: rgba(255, 255, 255, 0.9);
}

QLabel#Title {
    font-size: 28px;
    font-weight: bold;
    color: #ffffff;
}

QLabel#Subtitle {
    font-size: 12px;
    color: rgba(255, 255, 255, 0.5);
}

QLabel#Value {
    color: #00d4ff;
    font-weight: bold;
    font-size: 11px;
}

QSlider::groove:horizontal {
    height: 6px;
    background: rgba(255, 255, 255, 0.1);
    border-radius: 3px;
}

QSlider::handle:horizontal {
    background: #00d4ff;
    width: 18px;
    height: 18px;
    margin: -6px 0;
    border-radius: 9px;
}

QSlider::sub-page:horizontal {
    background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
        stop:0 #00d4ff, stop:1 #00ff88);
    border-radius: 3px;
}

QSlider::groove:vertical {
    width: 6px;
    background: rgba(255, 255, 255, 0.1);
    border-radius: 3px;
}

QSlider::handle:vertical {
    background: #00d4ff;
    height: 18px;
    width: 18px;
    margin: 0 -6px;
    border-radius: 9px;
}

QSlider::add-page:vertical {
    background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
        stop:0 #00d4ff, stop:1 #00ff88);
    border-radius: 3px;
}

QRadioButton {
    color: rgba(255, 255, 255, 0.8);
    spacing: 8px;
}

QRadioButton::indicator {
    width: 18px;
    height: 18px;
    border-radius: 9px;
    border: 2px solid rgba(255, 255, 255, 0.3);
}

QRadioButton::indicator:checked {
    background: #00d4ff;
    border-color: #00d4ff;
}

QCheckBox {
    color: rgba(255, 255, 255, 0.8);
    spacing: 8px;
}

QCheckBox::indicator {
    width: 18px;
    height: 18px;
    border-radius: 4px;
    border: 2px solid rgba(255, 255, 255, 0.3);
}

QCheckBox::indicator:checked {
    background: #00d4ff;
    border-color: #00d4ff;
}

QScrollArea {
    border: none;
    background: transparent;
}

QTabWidget::pane {
    border: 1px solid rgba(255, 255, 255, 0.1);
    border-radius: 12px;
    background: rgba(255, 255, 255, 0.03);
}

QTabBar::tab {
    background: rgba(255, 255, 255, 0.05);
    color: rgba(255, 255, 255, 0.6);
    padding: 10px 20px;
    border-top-left-radius: 8px;
    border-top-right-radius: 8px;
    margin-right: 2px;
}

QTabBar::tab:selected {
    background: rgba(0, 212, 255, 0.2);
    color: #00d4ff;
}

QSpinBox, QDoubleSpinBox {
    background: rgba(255, 255, 255, 0.08);
    border: 1px solid rgba(255, 255, 255, 0.15);
    border-radius: 6px;
    padding: 5px;
    color: #fff;
}
"""

# ==================== DSP ENGINE ====================

@dataclass
class DSPSettings:
    # EQ
    eq_gains: list = None
    # Compressor
    comp_threshold: float = 0.0
    comp_ratio: float = 1.0
    comp_attack: float = 10.0
    comp_release: float = 100.0
    # Limiter
    limiter_enabled: bool = False
    limiter_ceiling: float = -0.3
    # Stereo
    stereo_width: float = 1.0
    # Reverb
    reverb_enabled: bool = False
    reverb_mix: float = 0.2
    reverb_decay: float = 0.5
    # De-Esser
    deesser_enabled: bool = False
    deesser_threshold: float = -20.0
    deesser_freq: float = 6000.0
    # Saturation
    saturation_enabled: bool = False
    saturation_drive: float = 0.0
    saturation_mix: float = 0.5
    # Output
    output_gain: float = 0.0
    
    def __post_init__(self):
        if self.eq_gains is None:
            self.eq_gains = [0.0] * 10


class DSPProcessor:
    """Advanced DSP Processing Engine"""
    
    EQ_FREQS = [32, 64, 125, 250, 500, 1000, 2000, 4000, 8000, 16000]
    
    @staticmethod
    def apply_gain(audio: np.ndarray, db: float) -> np.ndarray:
        if abs(db) < 0.01:
            return audio
        return audio * (10 ** (db / 20.0))

    @staticmethod
    def apply_compressor(audio: np.ndarray, threshold_db: float, ratio: float, 
                         attack_ms: float, release_ms: float, sr: int) -> np.ndarray:
        if ratio <= 1.0:
            return audio
        
        # Calcola envelope con attack/release
        attack_samples = int(sr * attack_ms / 1000)
        release_samples = int(sr * release_ms / 1000)
        
        envelope = np.abs(audio)
        smoothed = np.zeros_like(envelope)
        
        # Smooth envelope
        for i in range(1, len(envelope)):
            if envelope[i] > smoothed[i-1]:
                coef = 1.0 - np.exp(-1.0 / max(attack_samples, 1))
            else:
                coef = 1.0 - np.exp(-1.0 / max(release_samples, 1))
            smoothed[i] = smoothed[i-1] + coef * (envelope[i] - smoothed[i-1])
        
        envelope_db = 20 * np.log10(smoothed + 1e-9)
        
        # Calcola gain reduction
        gain_reduction_db = np.zeros_like(envelope_db)
        mask = envelope_db > threshold_db
        gain_reduction_db[mask] = (threshold_db - envelope_db[mask]) * (1 - 1/ratio)
        
        gain_linear = 10 ** (gain_reduction_db / 20.0)
        compressed = audio * gain_linear
        
        # Makeup gain
        makeup = -np.min(gain_reduction_db) * 0.5
        return compressed * (10 ** (makeup / 20.0))

    @staticmethod
    def apply_limiter(audio: np.ndarray, ceiling_db: float) -> np.ndarray:
        ceiling = 10 ** (ceiling_db / 20.0)
        return np.clip(audio, -ceiling, ceiling)

    @staticmethod
    def apply_10band_eq(audio: np.ndarray, sr: int, gains: list) -> np.ndarray:
        if all(abs(g) < 0.1 for g in gains):
            return audio
            
        result = audio.copy()
        
        for freq, gain in zip(DSPProcessor.EQ_FREQS, gains):
            if abs(gain) < 0.1:
                continue
            
            Q = 1.5
            nyq = sr / 2
            
            # Calcola filtro peak
            if freq < nyq * 0.95:
                try:
                    b, a = signal.iirpeak(freq, Q, fs=sr)
                    filtered = signal.lfilter(b, a, result)
                    
                    factor = 10 ** (gain / 20.0) - 1
                    result = result + filtered * factor * 0.3
                except:
                    pass
                
        return result

    @staticmethod
    def apply_stereo_width(audio: np.ndarray, width: float) -> np.ndarray:
        """Applica stereo width (1.0 = normale, >1 = più largo, <1 = più stretto)"""
        if len(audio.shape) == 1:
            return audio  # Mono, nessun effetto
        
        if audio.shape[0] == 2:  # [2, samples]
            left, right = audio[0], audio[1]
        else:  # [samples, 2]
            left, right = audio[:, 0], audio[:, 1]
        
        mid = (left + right) / 2
        side = (left - right) / 2
        
        side = side * width
        
        new_left = mid + side
        new_right = mid - side
        
        if audio.shape[0] == 2:
            return np.stack([new_left, new_right])
        else:
            return np.column_stack([new_left, new_right])

    @staticmethod
    def apply_reverb(audio: np.ndarray, sr: int, mix: float, decay: float) -> np.ndarray:
        """Simple reverb basato su comb filters"""
        if mix <= 0:
            return audio
            
        delays = [int(sr * d) for d in [0.029, 0.037, 0.041, 0.043]]
        reverb = np.zeros_like(audio)
        
        for delay in delays:
            delayed = np.zeros_like(audio)
            delayed[delay:] = audio[:-delay] * decay
            reverb += delayed * 0.25
        
        return audio * (1 - mix) + reverb * mix

    @staticmethod
    def apply_deesser(audio: np.ndarray, sr: int, threshold_db: float, freq: float) -> np.ndarray:
        """De-esser per ridurre le sibilanti"""
        # Filtra le frequenze sibilanti
        sos = signal.butter(4, [freq * 0.7, min(freq * 1.5, sr/2 - 100)], 
                           'bandpass', fs=sr, output='sos')
        sibilance = signal.sosfilt(sos, audio)
        
        # Calcola envelope
        envelope = np.abs(sibilance)
        envelope_db = 20 * np.log10(envelope + 1e-9)
        
        # Applica riduzione
        reduction = np.ones_like(audio)
        mask = envelope_db > threshold_db
        reduction[mask] = 10 ** ((threshold_db - envelope_db[mask]) / 40.0)
        
        return audio * reduction

    @staticmethod
    def apply_saturation(audio: np.ndarray, drive: float, mix: float) -> np.ndarray:
        """Saturazione armonica"""
        if drive <= 0:
            return audio
            
        driven = audio * (1 + drive * 5)
        saturated = np.tanh(driven)
        
        return audio * (1 - mix) + saturated * mix

    @staticmethod
    def spectral_exciter(audio: np.ndarray, sr: int, amount: float = 0.2) -> np.ndarray:
        if amount <= 0:
            return audio
        sos = signal.butter(2, 3000, 'high', fs=sr, output='sos')
        high_freqs = signal.sosfilt(sos, audio)
        harmonics = np.tanh(high_freqs * 5) * 0.1
        return audio + (harmonics * amount)

    @staticmethod
    def process_chain(audio: np.ndarray, sr: int, settings: DSPSettings) -> np.ndarray:
        """Applica l'intera catena di processing"""
        result = audio.copy()
        
        # 1. EQ
        result = DSPProcessor.apply_10band_eq(result, sr, settings.eq_gains)
        
        # 2. De-Esser
        if settings.deesser_enabled:
            result = DSPProcessor.apply_deesser(
                result, sr, settings.deesser_threshold, settings.deesser_freq
            )
        
        # 3. Compressor
        if settings.comp_threshold < 0:
            result = DSPProcessor.apply_compressor(
                result, settings.comp_threshold, settings.comp_ratio,
                settings.comp_attack, settings.comp_release, sr
            )
        
        # 4. Saturation
        if settings.saturation_enabled and settings.saturation_drive > 0:
            result = DSPProcessor.apply_saturation(
                result, settings.saturation_drive, settings.saturation_mix
            )
        
        # 5. Reverb
        if settings.reverb_enabled:
            result = DSPProcessor.apply_reverb(
                result, sr, settings.reverb_mix, settings.reverb_decay
            )
        
        # 6. Stereo Width (se stereo)
        if settings.stereo_width != 1.0:
            result = DSPProcessor.apply_stereo_width(result, settings.stereo_width)
        
        # 7. Limiter
        if settings.limiter_enabled:
            result = DSPProcessor.apply_limiter(result, settings.limiter_ceiling)
        
        # 8. Output Gain
        result = DSPProcessor.apply_gain(result, settings.output_gain)
        
        # Clip finale
        return np.clip(result, -1.0, 1.0)


# ==================== AI ENGINE ====================

class AIEngine:
    def __init__(self):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"⚡ AI Engine: {self.device.upper()}")
        self.models = {}
        
    def load_demucs(self):
        if 'demucs' in self.models:
            return True
        try:
            from demucs.pretrained import get_model
            self.models['demucs'] = get_model('htdemucs')
            self.models['demucs'].to(self.device)
            return True
        except Exception as e:
            print(f"Demucs error: {e}")
            return False

    def load_denoiser(self):
        if 'denoiser' in self.models:
            return True
        try:
            from denoiser.pretrained import dns64
            self.models['denoiser'] = dns64().to(self.device)
            return True
        except Exception as e:
            print(f"Denoiser error: {e}")
            return False

    def restore_quality(self, audio: np.ndarray, sr: int, 
                       progress_callback=None) -> np.ndarray:
        if not self.load_demucs():
            return audio

        model = self.models['demucs']
        
        if progress_callback:
            progress_callback(10, "Preparazione audio...")

        if sr != 44100:
            audio_44 = librosa.resample(audio, orig_sr=sr, target_sr=44100)
        else:
            audio_44 = audio
        
        if len(audio_44.shape) == 1:
            audio_tensor = torch.tensor(np.stack([audio_44, audio_44]))
        else:
            audio_tensor = torch.tensor(audio_44)
        
        ref = audio_tensor.float().to(self.device).unsqueeze(0)

        if progress_callback:
            progress_callback(30, "Separazione sorgenti AI...")

        with torch.no_grad():
            sources = apply_model(model, ref, shifts=0, split=True, 
                                overlap=0.25, progress=False)
        
        if progress_callback:
            progress_callback(70, "Ricostruzione...")

        vocals = sources[0, 3].cpu().numpy()
        other = sources[0, 2].cpu().numpy()
        bass = sources[0, 1].cpu().numpy()
        drums = sources[0, 0].cpu().numpy()
        
        enhanced = vocals * 1.1 + other + bass * 1.05 + drums
        enhanced_mono = np.mean(enhanced, axis=0)
        
        enhanced_mono = DSPProcessor.spectral_exciter(enhanced_mono, 44100, amount=0.15)
        
        if sr != 44100:
            enhanced_mono = librosa.resample(enhanced_mono, orig_sr=44100, target_sr=sr)
        
        if progress_callback:
            progress_callback(100, "Completato!")
            
        return enhanced_mono

    def process_denoiser(self, audio: np.ndarray, sr: int,
                        progress_callback=None) -> np.ndarray:
        if not self.load_denoiser():
            return audio
            
        model = self.models['denoiser']
        target_sr = 16000
        
        if progress_callback:
            progress_callback(20, "Resampling...")
            
        audio_res = librosa.resample(audio, orig_sr=sr, target_sr=target_sr)
        tensor = torch.tensor(audio_res).float().to(self.device).unsqueeze(0)
        
        if progress_callback:
            progress_callback(50, "AI Denoising...")
            
        with torch.no_grad():
            out = model(tensor)
            
        enhanced = out.squeeze().cpu().numpy()
        enhanced = librosa.resample(enhanced, orig_sr=target_sr, target_sr=sr)
        
        min_len = min(len(audio), len(enhanced))
        
        if progress_callback:
            progress_callback(100, "Completato!")
            
        return enhanced[:min_len]


# ==================== AUDIO PLAYER ====================

class AudioPlayer(QObject):
    """Advanced Audio Player with seeking, speed control, A-B repeat"""
    
    position_changed = pyqtSignal(float)  # 0.0 - 1.0
    playback_finished = pyqtSignal()
    
    def __init__(self):
        super().__init__()
        self.audio = None
        self.sr = None
        self.is_playing = False
        self.is_paused = False
        self.position = 0  # samples
        self.volume = 1.0
        self.speed = 1.0
        self.loop = False
        self.a_point = None  # A-B repeat start
        self.b_point = None  # A-B repeat end
        self._stream = None
        self._timer = None
        
    def load(self, audio: np.ndarray, sr: int):
        self.stop()
        self.audio = np.ascontiguousarray(audio, dtype=np.float32)
        self.sr = sr
        self.position = 0
        self.a_point = None
        self.b_point = None
        
    def play(self):
        if self.audio is None:
            return
            
        if self.is_paused:
            self.is_paused = False
            self.is_playing = True
            self._start_stream()
            return
            
        self.is_playing = True
        self.is_paused = False
        self._start_stream()
        
    def _start_stream(self):
        if self._stream is not None:
            self._stream.stop()
            self._stream.close()
            
        def callback(outdata, frames, time_info, status):
            if not self.is_playing or self.is_paused:
                outdata.fill(0)
                return
                
            # Calcola end point (considerando A-B repeat)
            end_pos = self.b_point if self.b_point else len(self.audio)
            start_pos = self.a_point if self.a_point else 0
            
            remaining = end_pos - self.position
            
            if remaining <= 0:
                if self.loop:
                    self.position = start_pos
                    remaining = end_pos - self.position
                else:
                    outdata.fill(0)
                    self.is_playing = False
                    self.playback_finished.emit()
                    return
            
            chunk_size = min(frames, remaining)
            chunk = self.audio[self.position:self.position + chunk_size]
            
            # Applica volume
            chunk = chunk * self.volume
            
            # Padding se necessario
            if len(chunk) < frames:
                chunk = np.pad(chunk, (0, frames - len(chunk)))
            
            outdata[:, 0] = chunk
            if outdata.shape[1] > 1:
                outdata[:, 1] = chunk
                
            self.position += chunk_size
            
            # Emit position
            progress = self.position / len(self.audio)
            self.position_changed.emit(progress)
        
        target_sr = int(self.sr * self.speed)
        self._stream = sd.OutputStream(
            samplerate=target_sr,
            channels=2,
            callback=callback,
            blocksize=2048
        )
        self._stream.start()
        
    def pause(self):
        self.is_paused = True
        
    def stop(self):
        self.is_playing = False
        self.is_paused = False
        self.position = 0
        if self._stream:
            self._stream.stop()
            self._stream.close()
            self._stream = None
        self.position_changed.emit(0.0)
            
    def seek(self, position: float):
        """Seek to position (0.0 - 1.0)"""
        if self.audio is not None:
            self.position = int(position * len(self.audio))
            
    def set_volume(self, volume: float):
        """Set volume (0.0 - 1.0)"""
        self.volume = max(0.0, min(1.0, volume))
        
    def set_speed(self, speed: float):
        """Set playback speed"""
        self.speed = max(0.5, min(2.0, speed))
        if self.is_playing and not self.is_paused:
            # Restart stream with new speed
            self._start_stream()
            
    def set_loop(self, enabled: bool):
        self.loop = enabled
        
    def set_a_point(self):
        if self.audio is not None:
            self.a_point = self.position
            
    def set_b_point(self):
        if self.audio is not None:
            self.b_point = self.position
            
    def clear_ab(self):
        self.a_point = None
        self.b_point = None
        
    def get_duration(self) -> float:
        if self.audio is not None and self.sr:
            return len(self.audio) / self.sr
        return 0.0
        
    def get_current_time(self) -> float:
        if self.audio is not None and self.sr:
            return self.position / self.sr
        return 0.0


# ==================== WORKER THREAD ====================

class WorkerThread(QThread):
    finished = pyqtSignal(object, str)
    progress = pyqtSignal(int, str)

    def __init__(self, engine: AIEngine, audio: np.ndarray, sr: int, method: str):
        super().__init__()
        self.engine = engine
        self.audio = audio
        self.sr = sr
        self.method = method

    def run(self):
        try:
            def progress_cb(value, msg):
                self.progress.emit(value, msg)
                
            if self.method == "restore":
                enhanced = self.engine.restore_quality(
                    self.audio, self.sr, progress_cb
                )
            elif self.method == "fb_denoiser":
                enhanced = self.engine.process_denoiser(
                    self.audio, self.sr, progress_cb
                )
            elif self.method == "dsp":
                progress_cb(50, "DSP Noise Reduction...")
                enhanced = nr.reduce_noise(y=self.audio, sr=self.sr, prop_decrease=0.6)
                progress_cb(100, "Completato!")
            else:
                enhanced = self.audio
                
            self.finished.emit(enhanced, "✓ Elaborazione completata!")
        except Exception as e:
            self.finished.emit(None, f"✗ Errore: {str(e)}")


# ==================== UI WIDGETS ====================

class ModernWaveform(FigureCanvas):
    """Waveform display con gradient e marker"""
    
    clicked = pyqtSignal(float)  # Position 0.0-1.0
    
    def __init__(self, title="", color='#00d4ff'):
        self.fig = Figure(figsize=(5, 1.5), dpi=100)
        self.fig.patch.set_facecolor('#00000000')
        super().__init__(self.fig)
        
        self.ax = self.fig.add_subplot(111)
        self.ax.set_facecolor('#ffffff08')
        self.color = color
        self.title = title
        self.audio = None
        self.position_line = None
        
        self.fig.subplots_adjust(left=0.02, right=0.98, top=0.95, bottom=0.05)
        
        # Enable click events
        self.mpl_connect('button_press_event', self._on_click)
        
        self.clear_plot()

    def _on_click(self, event):
        if event.inaxes == self.ax and self.audio is not None:
            xlim = self.ax.get_xlim()
            position = (event.xdata - xlim[0]) / (xlim[1] - xlim[0])
            self.clicked.emit(max(0, min(1, position)))

    def clear_plot(self):
        self.ax.clear()
        self.audio = None
        self.ax.set_facecolor('#ffffff08')
        for spine in self.ax.spines.values():
            spine.set_visible(False)
        self.ax.set_xticks([])
        self.ax.set_yticks([])
        self.ax.text(0.5, 0.5, self.title or "No Audio", ha='center', va='center',
                    color='#ffffff30', fontsize=10, transform=self.ax.transAxes)
        self.draw()

    def plot_audio(self, audio: np.ndarray):
        self.ax.clear()
        self.audio = audio
        self.ax.set_facecolor('#ffffff08')
        
        step = max(1, len(audio) // 5000)
        data = audio[::step]
        x = np.linspace(0, len(audio), len(data))
        
        # Fill gradient effect
        self.ax.fill_between(x, data, 0, alpha=0.3, color=self.color)
        self.ax.plot(x, data, color=self.color, linewidth=0.5, alpha=0.8)
        
        self.ax.set_ylim(-1.1, 1.1)
        self.ax.set_xlim(0, len(audio))
        
        for spine in self.ax.spines.values():
            spine.set_visible(False)
        self.ax.set_xticks([])
        self.ax.set_yticks([])
        
        # Position line
        self.position_line = self.ax.axvline(0, color='#ff6b6b', linewidth=2, alpha=0)
        
        self.draw()
        
    def set_position(self, position: float):
        """Update position indicator (0.0-1.0)"""
        if self.position_line and self.audio is not None:
            x_pos = position * len(self.audio)
            self.position_line.set_xdata([x_pos, x_pos])
            self.position_line.set_alpha(0.8)
            self.draw_idle()


class SpectrumAnalyzer(FigureCanvas):
    """Real-time spectrum analyzer"""
    
    def __init__(self):
        self.fig = Figure(figsize=(5, 1.2), dpi=100)
        self.fig.patch.set_facecolor('#00000000')
        super().__init__(self.fig)
        
        self.ax = self.fig.add_subplot(111)
        self.ax.set_facecolor('#ffffff08')
        self.bars = None
        self.n_bars = 32
        
        self.fig.subplots_adjust(left=0.02, right=0.98, top=0.95, bottom=0.15)
        self._init_bars()
        
    def _init_bars(self):
        self.ax.clear()
        self.ax.set_facecolor('#ffffff08')
        
        x = np.arange(self.n_bars)
        heights = np.zeros(self.n_bars)
        
        colors = [f'#{int(0 + i*8):02x}{int(212 - i*4):02x}ff' for i in range(self.n_bars)]
        self.bars = self.ax.bar(x, heights, color=colors, width=0.8)
        
        self.ax.set_ylim(0, 1)
        self.ax.set_xlim(-0.5, self.n_bars - 0.5)
        
        for spine in self.ax.spines.values():
            spine.set_visible(False)
        self.ax.set_xticks([])
        self.ax.set_yticks([])
        
        self.draw()
        
    def update_spectrum(self, audio_chunk: np.ndarray, sr: int):
        if len(audio_chunk) < 1024:
            return
            
        # FFT
        fft = np.abs(np.fft.rfft(audio_chunk[:2048]))
        
        # Raggruppa in bande
        freqs = np.fft.rfftfreq(2048, 1/sr)
        bands = np.logspace(np.log10(20), np.log10(sr/2), self.n_bars + 1)
        
        heights = []
        for i in range(self.n_bars):
            mask = (freqs >= bands[i]) & (freqs < bands[i+1])
            if mask.any():
                heights.append(np.mean(fft[mask]))
            else:
                heights.append(0)
        
        heights = np.array(heights)
        heights = heights / (np.max(heights) + 1e-9)  # Normalize
        heights = np.clip(heights, 0, 1)
        
        for bar, h in zip(self.bars, heights):
            bar.set_height(h)
            
        self.draw_idle()


class VerticalEQSlider(QWidget):
    """Modern vertical EQ slider"""
    
    valueChanged = pyqtSignal(int)
    
    def __init__(self, freq_label: str):
        super().__init__()
        self.setFixedWidth(45)
        
        layout = QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)
        
        # Value label
        self.val_label = QLabel("0")
        self.val_label.setObjectName("Value")
        self.val_label.setAlignment(Qt.AlignCenter)
        
        # Slider
        self.slider = QSlider(Qt.Vertical)
        self.slider.setRange(-12, 12)
        self.slider.setValue(0)
        self.slider.setMinimumHeight(100)
        self.slider.valueChanged.connect(self._on_change)
        
        # Freq label
        self.freq_label = QLabel(freq_label)
        self.freq_label.setStyleSheet("color: rgba(255,255,255,0.5); font-size: 9px;")
        self.freq_label.setAlignment(Qt.AlignCenter)
        
        layout.addWidget(self.val_label)
        layout.addWidget(self.slider, 1)
        layout.addWidget(self.freq_label)
        
        self.setLayout(layout)
        
    def _on_change(self, value: int):
        self.val_label.setText(f"{value:+d}" if value != 0 else "0")
        self.valueChanged.emit(value)
        
    def value(self) -> int:
        return self.slider.value()
        
    def setValue(self, value: int):
        self.slider.setValue(value)


class ParameterKnob(QWidget):
    """Modern parameter control with label and value"""
    
    valueChanged = pyqtSignal(float)
    
    def __init__(self, name: str, min_val: float, max_val: float, 
                 default: float, unit: str = "", decimals: int = 0):
        super().__init__()
        
        self.min_val = min_val
        self.max_val = max_val
        self.decimals = decimals
        self.unit = unit
        
        layout = QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)
        
        # Name
        name_label = QLabel(name)
        name_label.setStyleSheet("color: rgba(255,255,255,0.6); font-size: 10px;")
        name_label.setAlignment(Qt.AlignCenter)
        
        # Slider
        self.slider = QSlider(Qt.Horizontal)
        self.slider.setRange(0, 100)
        self.slider.setValue(self._value_to_slider(default))
        self.slider.valueChanged.connect(self._on_change)
        
        # Value
        self.val_label = QLabel(self._format_value(default))
        self.val_label.setObjectName("Value")
        self.val_label.setAlignment(Qt.AlignCenter)
        
        layout.addWidget(name_label)
        layout.addWidget(self.slider)
        layout.addWidget(self.val_label)
        
        self.setLayout(layout)
        
    def _value_to_slider(self, value: float) -> int:
        return int((value - self.min_val) / (self.max_val - self.min_val) * 100)
        
    def _slider_to_value(self, slider: int) -> float:
        return self.min_val + (slider / 100) * (self.max_val - self.min_val)
        
    def _format_value(self, value: float) -> str:
        if self.decimals == 0:
            return f"{int(value)}{self.unit}"
        else:
            return f"{value:.{self.decimals}f}{self.unit}"
            
    def _on_change(self, slider_val: int):
        value = self._slider_to_value(slider_val)
        self.val_label.setText(self._format_value(value))
        self.valueChanged.emit(value)
        
    def value(self) -> float:
        return self._slider_to_value(self.slider.value())
        
    def setValue(self, value: float):
        self.slider.setValue(self._value_to_slider(value))
        self.val_label.setText(self._format_value(value))


class ToggleSwitch(QWidget):
    """Modern toggle switch"""
    
    toggled = pyqtSignal(bool)
    
    def __init__(self, label: str = ""):
        super().__init__()
        
        layout = QHBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        
        if label:
            lbl = QLabel(label)
            lbl.setStyleSheet("color: rgba(255,255,255,0.8);")
            layout.addWidget(lbl)
        
        self.checkbox = QCheckBox()
        self.checkbox.stateChanged.connect(lambda s: self.toggled.emit(s == Qt.Checked))
        layout.addWidget(self.checkbox)
        
        self.setLayout(layout)
        
    def isChecked(self) -> bool:
        return self.checkbox.isChecked()
        
    def setChecked(self, checked: bool):
        self.checkbox.setChecked(checked)


# ==================== MAIN WINDOW ====================

class AudioMasterPro(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("AUDIO MASTER PRO v6.0")
        self.setMinimumSize(1400, 900)
        
        # Engines
        self.ai_engine = AIEngine()
        self.player = AudioPlayer()
        self.dsp_settings = DSPSettings()
        
        # Audio data
        self.audio_original = None
        self.audio_processed = None
        self.audio_final = None
        self.sr = None
        
        # EQ Presets
        self.eq_presets = {
            "Flat": [0]*10,
            "Bass Boost": [6, 5, 3, 1, 0, 0, 0, 0, 0, 0],
            "Treble Boost": [0, 0, 0, 0, 0, 1, 2, 3, 4, 5],
            "Vocal Enhance": [-2, -1, 0, 1, 2, 4, 4, 2, 0, -1],
            "Loudness": [4, 3, 0, 0, 0, 0, 0, 0, 2, 3],
            "Master Polish": [1, 0, -1, -1, 0, 1, 2, 3, 3, 2],
            "Broadcast": [3, 2, 0, 0, 1, 2, 3, 2, 0, -2],
            "Podcast": [2, 1, 0, 0, 3, 4, 3, 1, 0, 0],
        }
        
        self._init_ui()
        self._connect_signals()
        
    def _init_ui(self):
        self.setStyleSheet(MODERN_STYLE)
        
        central = QWidget()
        central.setObjectName("CentralWidget")
        self.setCentralWidget(central)
        
        main_layout = QHBoxLayout(central)
        main_layout.setSpacing(20)
        main_layout.setContentsMargins(20, 20, 20, 20)
        
        # === LEFT PANEL ===
        left_panel = QVBoxLayout()
        left_panel.setSpacing(15)
        
        # Header
        header_widget = QWidget()
        header_layout = QVBoxLayout(header_widget)
        header_layout.setContentsMargins(0, 0, 0, 10)
        
        title = QLabel("AUDIO MASTER PRO")
        title.setObjectName("Title")
        subtitle = QLabel("AI-Powered Audio Enhancement & Mastering Suite")
        subtitle.setObjectName("Subtitle")
        
        header_layout.addWidget(title)
        header_layout.addWidget(subtitle)
        left_panel.addWidget(header_widget)
        
        # File & AI Section
        file_group = QGroupBox("📂 FILE & AI PROCESSING")
        file_layout = QVBoxLayout(file_group)
        
        # File row
        file_row = QHBoxLayout()
        self.btn_load = QPushButton("📁 Load Audio")
        self.btn_load.setMinimumHeight(40)
        self.file_label = QLabel("No file loaded")
        self.file_label.setStyleSheet("color: rgba(255,255,255,0.5);")
        file_row.addWidget(self.btn_load)
        file_row.addWidget(self.file_label, 1)
        file_layout.addLayout(file_row)
        
        # AI Mode
        ai_row = QHBoxLayout()
        self.combo_ai_mode = QComboBox()
        self.combo_ai_mode.addItems([
            "✨ AI Quality Restorer (Demucs + Exciter)",
            "🎤 Voice Cleaner (Facebook Denoiser)",
            "💨 Fast Denoise (DSP Only)"
        ])
        self.btn_process = QPushButton("⚡ PROCESS")
        self.btn_process.setObjectName("AccentBtn")
        self.btn_process.setMinimumHeight(40)
        self.btn_process.setEnabled(False)
        ai_row.addWidget(self.combo_ai_mode, 2)
        ai_row.addWidget(self.btn_process, 1)
        file_layout.addLayout(ai_row)
        
        # Progress
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        self.progress_label = QLabel("")
        self.progress_label.setStyleSheet("color: #00d4ff; font-size: 11px;")
        self.progress_label.setVisible(False)
        file_layout.addWidget(self.progress_bar)
        file_layout.addWidget(self.progress_label)
        
        left_panel.addWidget(file_group)
        
        # EQ Section
        eq_group = QGroupBox("🎚️ 10-BAND EQUALIZER")
        eq_layout = QVBoxLayout(eq_group)
        
        # Preset row
        preset_row = QHBoxLayout()
        preset_row.addWidget(QLabel("Preset:"))
        self.combo_eq_preset = QComboBox()
        self.combo_eq_preset.addItems(list(self.eq_presets.keys()))
        preset_row.addWidget(self.combo_eq_preset, 1)
        eq_layout.addLayout(preset_row)
        
        # EQ Sliders
        sliders_row = QHBoxLayout()
        sliders_row.setSpacing(2)
        self.eq_sliders = []
        freqs = ["32", "64", "125", "250", "500", "1K", "2K", "4K", "8K", "16K"]
        for freq in freqs:
            slider = VerticalEQSlider(freq)
            slider.slider.setEnabled(False)
            self.eq_sliders.append(slider)
            sliders_row.addWidget(slider)
        eq_layout.addLayout(sliders_row)
        
        left_panel.addWidget(eq_group)
        
        # Dynamics Section
        dynamics_group = QGroupBox("🔊 DYNAMICS & EFFECTS")
        dynamics_layout = QVBoxLayout(dynamics_group)
        
        # Create tabs for different effect sections
        tabs = QTabWidget()
        
        # Compressor Tab
        comp_tab = QWidget()
        comp_layout = QGridLayout(comp_tab)
        comp_layout.setSpacing(10)
        
        self.knob_threshold = ParameterKnob("Threshold", -60, 0, 0, " dB")
        self.knob_ratio = ParameterKnob("Ratio", 1, 20, 1, ":1")
        self.knob_attack = ParameterKnob("Attack", 0.1, 100, 10, " ms", 1)
        self.knob_release = ParameterKnob("Release", 10, 1000, 100, " ms")
        
        comp_layout.addWidget(self.knob_threshold, 0, 0)
        comp_layout.addWidget(self.knob_ratio, 0, 1)
        comp_layout.addWidget(self.knob_attack, 1, 0)
        comp_layout.addWidget(self.knob_release, 1, 1)
        
        tabs.addTab(comp_tab, "Compressor")
        
        # Limiter Tab
        limiter_tab = QWidget()
        limiter_layout = QVBoxLayout(limiter_tab)
        
        self.toggle_limiter = ToggleSwitch("Enable Limiter")
        self.knob_ceiling = ParameterKnob("Ceiling", -6, 0, -0.3, " dB", 1)
        
        limiter_layout.addWidget(self.toggle_limiter)
        limiter_layout.addWidget(self.knob_ceiling)
        limiter_layout.addStretch()
        
        tabs.addTab(limiter_tab, "Limiter")
        
        # Stereo Tab
        stereo_tab = QWidget()
        stereo_layout = QVBoxLayout(stereo_tab)
        
        self.knob_stereo_width = ParameterKnob("Stereo Width", 0, 2, 1, "x", 2)
        stereo_layout.addWidget(self.knob_stereo_width)
        stereo_layout.addStretch()
        
        tabs.addTab(stereo_tab, "Stereo")
        
        # Effects Tab
        fx_tab = QWidget()
        fx_layout = QVBoxLayout(fx_tab)
        
        # Reverb
        reverb_row = QHBoxLayout()
        self.toggle_reverb = ToggleSwitch("Reverb")
        self.knob_reverb_mix = ParameterKnob("Mix", 0, 1, 0.2, "", 2)
        self.knob_reverb_decay = ParameterKnob("Decay", 0.1, 1, 0.5, "", 2)
        reverb_row.addWidget(self.toggle_reverb)
        reverb_row.addWidget(self.knob_reverb_mix)
        reverb_row.addWidget(self.knob_reverb_decay)
        fx_layout.addLayout(reverb_row)
        
        # Saturation
        sat_row = QHBoxLayout()
        self.toggle_saturation = ToggleSwitch("Saturation")
        self.knob_sat_drive = ParameterKnob("Drive", 0, 1, 0, "", 2)
        self.knob_sat_mix = ParameterKnob("Mix", 0, 1, 0.5, "", 2)
        sat_row.addWidget(self.toggle_saturation)
        sat_row.addWidget(self.knob_sat_drive)
        sat_row.addWidget(self.knob_sat_mix)
        fx_layout.addLayout(sat_row)
        
        tabs.addTab(fx_tab, "Effects")
        
        # De-Esser Tab
        deesser_tab = QWidget()
        deesser_layout = QVBoxLayout(deesser_tab)
        
        self.toggle_deesser = ToggleSwitch("Enable De-Esser")
        self.knob_deesser_thresh = ParameterKnob("Threshold", -40, 0, -20, " dB")
        self.knob_deesser_freq = ParameterKnob("Frequency", 4000, 10000, 6000, " Hz")
        
        deesser_layout.addWidget(self.toggle_deesser)
        deesser_layout.addWidget(self.knob_deesser_thresh)
        deesser_layout.addWidget(self.knob_deesser_freq)
        
        tabs.addTab(deesser_tab, "De-Esser")
        
        dynamics_layout.addWidget(tabs)
        
        # Output Gain
        output_row = QHBoxLayout()
        output_row.addWidget(QLabel("Output Gain:"))
        self.knob_output = ParameterKnob("", -12, 12, 0, " dB")
        output_row.addWidget(self.knob_output, 1)
        dynamics_layout.addLayout(output_row)
        
        left_panel.addWidget(dynamics_group)
        
        # Export Button
        self.btn_export = QPushButton("💾 EXPORT RESULT")
        self.btn_export.setObjectName("AccentBtn")
        self.btn_export.setMinimumHeight(50)
        self.btn_export.setEnabled(False)
        left_panel.addWidget(self.btn_export)
        
        main_layout.addLayout(left_panel, 4)
        
        # === RIGHT PANEL ===
        right_panel = QVBoxLayout()
        right_panel.setSpacing(15)
        
        # Waveforms
        wave_group = QGroupBox("📊 WAVEFORM DISPLAY")
        wave_layout = QVBoxLayout(wave_group)
        
        # Input waveform
        input_label = QLabel("INPUT (Original)")
        input_label.setStyleSheet("color: #3b82f6; font-weight: bold;")
        self.waveform_input = ModernWaveform("Load an audio file", '#3b82f6')
        wave_layout.addWidget(input_label)
        wave_layout.addWidget(self.waveform_input)
        
        # Output waveform
        output_label = QLabel("OUTPUT (Processed)")
        output_label.setStyleSheet("color: #00d4ff; font-weight: bold;")
        self.waveform_output = ModernWaveform("Process audio to see result", '#00d4ff')
        wave_layout.addWidget(output_label)
        wave_layout.addWidget(self.waveform_output)
        
        # Spectrum
        spectrum_label = QLabel("SPECTRUM ANALYZER")
        spectrum_label.setStyleSheet("color: #00ff88; font-weight: bold; font-size: 10px;")
        self.spectrum = SpectrumAnalyzer()
        wave_layout.addWidget(spectrum_label)
        wave_layout.addWidget(self.spectrum)
        
        right_panel.addWidget(wave_group, 2)
        
        # Player Section
        player_group = QGroupBox("🎧 ADVANCED PLAYER")
        player_layout = QVBoxLayout(player_group)
        
        # Source selection
        source_row = QHBoxLayout()
        self.rb_original = QRadioButton("Original")
        self.rb_processed = QRadioButton("Processed")
        self.rb_processed.setChecked(True)
        source_row.addWidget(QLabel("Source:"))
        source_row.addWidget(self.rb_original)
        source_row.addWidget(self.rb_processed)
        source_row.addStretch()
        player_layout.addLayout(source_row)
        
        # Seek bar
        seek_row = QHBoxLayout()
        self.lbl_time_current = QLabel("0:00")
        self.lbl_time_current.setStyleSheet("color: #00d4ff; font-family: monospace;")
        self.seek_slider = QSlider(Qt.Horizontal)
        self.seek_slider.setRange(0, 1000)
        self.lbl_time_total = QLabel("0:00")
        self.lbl_time_total.setStyleSheet("color: rgba(255,255,255,0.5); font-family: monospace;")
        seek_row.addWidget(self.lbl_time_current)
        seek_row.addWidget(self.seek_slider, 1)
        seek_row.addWidget(self.lbl_time_total)
        player_layout.addLayout(seek_row)
        
        # Main controls
        controls_row = QHBoxLayout()
        controls_row.setSpacing(10)
        
        self.btn_stop = QPushButton("⏹")
        self.btn_stop.setMinimumSize(50, 50)
        
        self.btn_play = QPushButton("▶")
        self.btn_play.setObjectName("PlayBtn")
        
        self.btn_loop = QPushButton("🔁")
        self.btn_loop.setCheckable(True)
        self.btn_loop.setMinimumSize(50, 50)
        
        controls_row.addStretch()
        controls_row.addWidget(self.btn_stop)
        controls_row.addWidget(self.btn_play)
        controls_row.addWidget(self.btn_loop)
        controls_row.addStretch()
        
        player_layout.addLayout(controls_row)
        
        # Volume & Speed
        vol_speed_row = QHBoxLayout()
        
        # Volume
        vol_col = QVBoxLayout()
        vol_col.addWidget(QLabel("Volume"))
        self.slider_volume = QSlider(Qt.Horizontal)
        self.slider_volume.setRange(0, 100)
        self.slider_volume.setValue(100)
        self.lbl_volume = QLabel("100%")
        self.lbl_volume.setObjectName("Value")
        vol_col.addWidget(self.slider_volume)
        vol_col.addWidget(self.lbl_volume)
        vol_speed_row.addLayout(vol_col)
        
        # Speed
        speed_col = QVBoxLayout()
        speed_col.addWidget(QLabel("Speed"))
        self.slider_speed = QSlider(Qt.Horizontal)
        self.slider_speed.setRange(50, 200)
        self.slider_speed.setValue(100)
        self.lbl_speed = QLabel("1.0x")
        self.lbl_speed.setObjectName("Value")
        speed_col.addWidget(self.slider_speed)
        speed_col.addWidget(self.lbl_speed)
        vol_speed_row.addLayout(speed_col)
        
        player_layout.addLayout(vol_speed_row)
        
        # A-B Repeat
        ab_row = QHBoxLayout()
        self.btn_set_a = QPushButton("Set A")
        self.btn_set_b = QPushButton("Set B")
        self.btn_clear_ab = QPushButton("Clear A-B")
        self.lbl_ab_status = QLabel("A-B: Off")
        self.lbl_ab_status.setStyleSheet("color: rgba(255,255,255,0.5);")
        ab_row.addWidget(self.btn_set_a)
        ab_row.addWidget(self.btn_set_b)
        ab_row.addWidget(self.btn_clear_ab)
        ab_row.addWidget(self.lbl_ab_status)
        ab_row.addStretch()
        player_layout.addLayout(ab_row)
        
        right_panel.addWidget(player_group, 1)
        
        main_layout.addLayout(right_panel, 5)
        
        # Disable controls initially
        self._set_controls_enabled(False)
        
    def _connect_signals(self):
        # File operations
        self.btn_load.clicked.connect(self._load_file)
        self.btn_process.clicked.connect(self._start_processing)
        self.btn_export.clicked.connect(self._export_file)
        
        # EQ
        self.combo_eq_preset.currentTextChanged.connect(self._load_eq_preset)
        for slider in self.eq_sliders:
            slider.valueChanged.connect(self._on_param_change)
            
        # Dynamics
        self.knob_threshold.valueChanged.connect(self._on_param_change)
        self.knob_ratio.valueChanged.connect(self._on_param_change)
        self.knob_attack.valueChanged.connect(self._on_param_change)
        self.knob_release.valueChanged.connect(self._on_param_change)
        self.toggle_limiter.toggled.connect(self._on_param_change)
        self.knob_ceiling.valueChanged.connect(self._on_param_change)
        self.knob_stereo_width.valueChanged.connect(self._on_param_change)
        self.toggle_reverb.toggled.connect(self._on_param_change)
        self.knob_reverb_mix.valueChanged.connect(self._on_param_change)
        self.knob_reverb_decay.valueChanged.connect(self._on_param_change)
        self.toggle_saturation.toggled.connect(self._on_param_change)
        self.knob_sat_drive.valueChanged.connect(self._on_param_change)
        self.knob_sat_mix.valueChanged.connect(self._on_param_change)
        self.toggle_deesser.toggled.connect(self._on_param_change)
        self.knob_deesser_thresh.valueChanged.connect(self._on_param_change)
        self.knob_deesser_freq.valueChanged.connect(self._on_param_change)
        self.knob_output.valueChanged.connect(self._on_param_change)
        
        # Player
        self.btn_play.clicked.connect(self._toggle_play)
        self.btn_stop.clicked.connect(self._stop_playback)
        self.btn_loop.clicked.connect(lambda: self.player.set_loop(self.btn_loop.isChecked()))
        self.slider_volume.valueChanged.connect(self._on_volume_change)
        self.slider_speed.valueChanged.connect(self._on_speed_change)
        self.seek_slider.sliderMoved.connect(self._on_seek)
        self.btn_set_a.clicked.connect(self._set_point_a)
        self.btn_set_b.clicked.connect(self._set_point_b)
        self.btn_clear_ab.clicked.connect(self._clear_ab)
        
        # Player signals
        self.player.position_changed.connect(self._on_position_change)
        self.player.playback_finished.connect(self._on_playback_finished)
        
        # Waveform clicks
        self.waveform_input.clicked.connect(self._on_waveform_click)
        self.waveform_output.clicked.connect(self._on_waveform_click)

    def _set_controls_enabled(self, enabled: bool):
        for slider in self.eq_sliders:
            slider.slider.setEnabled(enabled)
        self.knob_threshold.slider.setEnabled(enabled)
        self.knob_ratio.slider.setEnabled(enabled)
        self.knob_attack.slider.setEnabled(enabled)
        self.knob_release.slider.setEnabled(enabled)
        self.toggle_limiter.checkbox.setEnabled(enabled)
        self.knob_ceiling.slider.setEnabled(enabled)
        self.knob_stereo_width.slider.setEnabled(enabled)
        self.toggle_reverb.checkbox.setEnabled(enabled)
        self.knob_reverb_mix.slider.setEnabled(enabled)
        self.knob_reverb_decay.slider.setEnabled(enabled)
        self.toggle_saturation.checkbox.setEnabled(enabled)
        self.knob_sat_drive.slider.setEnabled(enabled)
        self.knob_sat_mix.slider.setEnabled(enabled)
        self.toggle_deesser.checkbox.setEnabled(enabled)
        self.knob_deesser_thresh.slider.setEnabled(enabled)
        self.knob_deesser_freq.slider.setEnabled(enabled)
        self.knob_output.slider.setEnabled(enabled)

    def _load_file(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Open Audio File", "",
            "Audio Files (*.mp3 *.wav *.flac *.ogg *.m4a);;All Files (*)"
        )
        if path:
            try:
                self.audio_original, self.sr = librosa.load(path, sr=None, mono=True)
                self.file_label.setText(os.path.basename(path))
                self.waveform_input.plot_audio(self.audio_original)
                self.waveform_output.clear_plot()
                
                self.btn_process.setEnabled(True)
                self.audio_processed = None
                self.audio_final = None
                self.btn_export.setEnabled(False)
                self._set_controls_enabled(False)
                
                # Update time label
                duration = len(self.audio_original) / self.sr
                self.lbl_time_total.setText(self._format_time(duration))
                
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to load file:\n{str(e)}")

    def _start_processing(self):
        if self.audio_original is None:
            return
            
        methods = ["restore", "fb_denoiser", "dsp"]
        method = methods[self.combo_ai_mode.currentIndex()]
        
        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(0)
        self.progress_label.setVisible(True)
        self.progress_label.setText("Starting...")
        self.btn_process.setEnabled(False)
        
        self.worker = WorkerThread(self.ai_engine, self.audio_original, self.sr, method)
        self.worker.progress.connect(self._on_progress)
        self.worker.finished.connect(self._on_processing_done)
        self.worker.start()

    def _on_progress(self, value: int, message: str):
        self.progress_bar.setValue(value)
        self.progress_label.setText(message)

    def _on_processing_done(self, audio, message: str):
        self.progress_bar.setVisible(False)
        self.progress_label.setVisible(False)
        self.btn_process.setEnabled(True)
        
        if audio is not None:
            self.audio_processed = audio
            self._set_controls_enabled(True)
            self._apply_dsp_chain()
            QMessageBox.information(self, "Success", message)
        else:
            QMessageBox.critical(self, "Error", message)

    def _load_eq_preset(self, name: str):
        if name in self.eq_presets:
            values = self.eq_presets[name]
            for i, val in enumerate(values):
                self.eq_sliders[i].setValue(val)
            self._on_param_change()

    def _on_param_change(self, *args):
        """Chiamato quando qualsiasi parametro cambia"""
        if self.audio_processed is not None:
            self._apply_dsp_chain()

    def _apply_dsp_chain(self):
        if self.audio_processed is None:
            return
            
        # Aggiorna settings
        self.dsp_settings.eq_gains = [s.value() for s in self.eq_sliders]
        self.dsp_settings.comp_threshold = self.knob_threshold.value()
        self.dsp_settings.comp_ratio = self.knob_ratio.value()
        self.dsp_settings.comp_attack = self.knob_attack.value()
        self.dsp_settings.comp_release = self.knob_release.value()
        self.dsp_settings.limiter_enabled = self.toggle_limiter.isChecked()
        self.dsp_settings.limiter_ceiling = self.knob_ceiling.value()
        self.dsp_settings.stereo_width = self.knob_stereo_width.value()
        self.dsp_settings.reverb_enabled = self.toggle_reverb.isChecked()
        self.dsp_settings.reverb_mix = self.knob_reverb_mix.value()
        self.dsp_settings.reverb_decay = self.knob_reverb_decay.value()
        self.dsp_settings.deesser_enabled = self.toggle_deesser.isChecked()
        self.dsp_settings.deesser_threshold = self.knob_deesser_thresh.value()
        self.dsp_settings.deesser_freq = self.knob_deesser_freq.value()
        self.dsp_settings.saturation_enabled = self.toggle_saturation.isChecked()
        self.dsp_settings.saturation_drive = self.knob_sat_drive.value()
        self.dsp_settings.saturation_mix = self.knob_sat_mix.value()
        self.dsp_settings.output_gain = self.knob_output.value()
        
        # Processa
        self.audio_final = DSPProcessor.process_chain(
            self.audio_processed, self.sr, self.dsp_settings
        )
        self.audio_final = np.ascontiguousarray(self.audio_final, dtype=np.float32)
        
        # Aggiorna UI
        self.waveform_output.plot_audio(self.audio_final)
        self.btn_export.setEnabled(True)
        
        # Aggiorna player se in riproduzione
        if self.player.is_playing and self.rb_processed.isChecked():
            self.player.load(self.audio_final, self.sr)
            self.player.play()

    def _toggle_play(self):
        if self.player.is_playing and not self.player.is_paused:
            self.player.pause()
            self.btn_play.setText("▶")
        else:
            audio = self._get_playback_audio()
            if audio is None:
                return
            
            if not self.player.is_paused:
                self.player.load(audio, self.sr)
            self.player.play()
            self.btn_play.setText("⏸")

    def _stop_playback(self):
        self.player.stop()
        self.btn_play.setText("▶")

    def _get_playback_audio(self) -> Optional[np.ndarray]:
        if self.rb_processed.isChecked() and self.audio_final is not None:
            return self.audio_final
        elif self.audio_original is not None:
            return self.audio_original
        return None

    def _on_volume_change(self, value: int):
        volume = value / 100.0
        self.player.set_volume(volume)
        self.lbl_volume.setText(f"{value}%")

    def _on_speed_change(self, value: int):
        speed = value / 100.0
        self.player.set_speed(speed)
        self.lbl_speed.setText(f"{speed:.2f}x")

    def _on_seek(self, value: int):
        position = value / 1000.0
        self.player.seek(position)

    def _on_position_change(self, position: float):
        self.seek_slider.blockSignals(True)
        self.seek_slider.setValue(int(position * 1000))
        self.seek_slider.blockSignals(False)
        
        current_time = self.player.get_current_time()
        self.lbl_time_current.setText(self._format_time(current_time))
        
        # Update waveform position
        self.waveform_output.set_position(position)
        
        # Update spectrum
        audio = self._get_playback_audio()
        if audio is not None:
            pos_samples = int(position * len(audio))
            chunk = audio[max(0, pos_samples-2048):pos_samples]
            if len(chunk) > 0:
                self.spectrum.update_spectrum(chunk, self.sr)

    def _on_playback_finished(self):
        self.btn_play.setText("▶")

    def _on_waveform_click(self, position: float):
        self.player.seek(position)
        if not self.player.is_playing:
            audio = self._get_playback_audio()
            if audio is not None:
                self.player.load(audio, self.sr)

    def _set_point_a(self):
        self.player.set_a_point()
        self._update_ab_status()

    def _set_point_b(self):
        self.player.set_b_point()
        self._update_ab_status()

    def _clear_ab(self):
        self.player.clear_ab()
        self._update_ab_status()

    def _update_ab_status(self):
        if self.player.a_point is not None or self.player.b_point is not None:
            a_time = self._format_time(self.player.a_point / self.sr) if self.player.a_point else "--"
            b_time = self._format_time(self.player.b_point / self.sr) if self.player.b_point else "--"
            self.lbl_ab_status.setText(f"A: {a_time} | B: {b_time}")
            self.lbl_ab_status.setStyleSheet("color: #00d4ff;")
        else:
            self.lbl_ab_status.setText("A-B: Off")
            self.lbl_ab_status.setStyleSheet("color: rgba(255,255,255,0.5);")

    def _format_time(self, seconds: float) -> str:
        mins = int(seconds // 60)
        secs = int(seconds % 60)
        return f"{mins}:{secs:02d}"

    def _export_file(self):
        if self.audio_final is None:
            return
            
        path, filter_used = QFileDialog.getSaveFileName(
            self, "Export Audio", "",
            "WAV File (*.wav);;MP3 File (*.mp3);;FLAC File (*.flac)"
        )
        
        if path:
            try:
                if path.lower().endswith('.mp3'):
                    import tempfile
                    tmp = tempfile.mktemp(suffix=".wav")
                    sf.write(tmp, self.audio_final, self.sr)
                    from pydub import AudioSegment
                    AudioSegment.from_wav(tmp).export(path, format="mp3", bitrate="320k")
                    os.remove(tmp)
                else:
                                        sf.write(path, self.audio_final, self.sr)
                    
                QMessageBox.information(self, "Success", f"Audio exported successfully!\n{path}")
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Export failed:\n{str(e)}")

    def closeEvent(self, event):
        self.player.stop()
        event.accept()


# ==================== SPLASH SCREEN ====================

class SplashScreen(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowFlags(Qt.FramelessWindowHint | Qt.WindowStaysOnTopHint)
        self.setAttribute(Qt.WA_TranslucentBackground)
        self.setFixedSize(500, 300)
        
        # Center on screen
        screen = QApplication.primaryScreen().geometry()
        self.move((screen.width() - 500) // 2, (screen.height() - 300) // 2)
        
    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        
        # Background gradient
        gradient = QLinearGradient(0, 0, 500, 300)
        gradient.setColorAt(0, QColor(26, 26, 46))
        gradient.setColorAt(0.5, QColor(22, 33, 62))
        gradient.setColorAt(1, QColor(15, 52, 96))
        
        painter.setBrush(gradient)
        painter.setPen(Qt.NoPen)
        painter.drawRoundedRect(0, 0, 500, 300, 20, 20)
        
        # Border
        painter.setPen(QPen(QColor(0, 212, 255, 100), 2))
        painter.setBrush(Qt.NoBrush)
        painter.drawRoundedRect(1, 1, 498, 298, 20, 20)
        
        # Title
        painter.setPen(QColor(255, 255, 255))
        font = painter.font()
        font.setPointSize(28)
        font.setBold(True)
        painter.setFont(font)
        painter.drawText(QRect(0, 80, 500, 50), Qt.AlignCenter, "AUDIO MASTER PRO")
        
        # Version
        font.setPointSize(14)
        font.setBold(False)
        painter.setFont(font)
        painter.setPen(QColor(0, 212, 255))
        painter.drawText(QRect(0, 130, 500, 30), Qt.AlignCenter, "v6.0 Modern Edition")
        
        # Loading text
        font.setPointSize(10)
        painter.setFont(font)
        painter.setPen(QColor(255, 255, 255, 150))
        painter.drawText(QRect(0, 220, 500, 30), Qt.AlignCenter, "Initializing AI Engine...")
        
        # Progress bar background
        painter.setBrush(QColor(255, 255, 255, 30))
        painter.setPen(Qt.NoPen)
        painter.drawRoundedRect(50, 260, 400, 8, 4, 4)
        
        # Progress bar fill (animated)
        progress_width = int(400 * (self.progress / 100)) if hasattr(self, 'progress') else 0
        gradient = QLinearGradient(50, 0, 450, 0)
        gradient.setColorAt(0, QColor(0, 212, 255))
        gradient.setColorAt(1, QColor(0, 255, 136))
        painter.setBrush(gradient)
        painter.drawRoundedRect(50, 260, progress_width, 8, 4, 4)
        
    def setProgress(self, value):
        self.progress = value
        self.update()


# ==================== APPLICATION ENTRY ====================

def main():
    # High DPI support
    QApplication.setAttribute(Qt.AA_EnableHighDpiScaling, True)
    QApplication.setAttribute(Qt.AA_UseHighDpiPixmaps, True)
    
    app = QApplication(sys.argv)
    app.setStyle('Fusion')
    
    # Set application info
    app.setApplicationName("Audio Master Pro")
    app.setApplicationVersion("1.0")
    app.setOrganizationName("AudioMasterPro")
    
    # Show splash screen
    splash = SplashScreen()
    splash.show()
    splash.setProgress(0)
    app.processEvents()
    
    # Simulate loading
    for i in range(101):
        splash.setProgress(i)
        app.processEvents()
        if i < 30:
            time.sleep(0.01)
        elif i < 60:
            time.sleep(0.005)
        else:
            time.sleep(0.002)
    
    # Create and show main window
    window = AudioMasterPro()
    window.show()
    
    # Close splash
    splash.close()
    
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()