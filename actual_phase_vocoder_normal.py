import numpy as np
import os
from scipy.signal import stft, istft
import wave
import librosa
import matplotlib.pyplot as plt
import librosa.display
from scipy.signal import spectrogram
from scipy.io import wavfile


def phase_vocoder(signal, tsm_rate, psm_rate, window_size=1024, hop_size=256):
    # Compute STFT
    f, t, Zxx = stft(signal, nperseg=window_size, noverlap=window_size - hop_size)

    # Time-Scale Modification
    new_t = np.arange(0, len(t) * tsm_rate) / tsm_rate
    new_Zxx_tsm = np.zeros((Zxx.shape[0], len(new_t)), dtype=Zxx.dtype)

    for k in range(Zxx.shape[0]):
        for u in range(1, len(new_t)):
            u_original = int(u / tsm_rate)  # Map to original frame index
            if u_original < len(t) - 1:  # Ensure index is within bounds
                omega_k = (np.angle(Zxx[k, u_original + 1]) - np.angle(Zxx[k, u_original])) / hop_size
                new_Zxx_tsm[k, u] = np.abs(Zxx[k, u_original]) * np.exp(
                    1j * (np.angle(new_Zxx_tsm[k, u - 1]) + omega_k * hop_size * tsm_rate))

    # Pitch-Scale Modification
    new_Zxx_psm = np.zeros_like(new_Zxx_tsm, dtype=new_Zxx_tsm.dtype)

    for k in range(1, new_Zxx_tsm.shape[0]):
        for u in range(len(new_t)):
            k_psm = int(k / psm_rate)  # Adjust frequency bin index for pitch scaling
            if k_psm < new_Zxx_tsm.shape[0] - 1:
                rho_k = k / psm_rate - k_psm
                new_Zxx_psm[k, u] = (1 - rho_k) * new_Zxx_tsm[k_psm, u] + rho_k * new_Zxx_tsm[k_psm + 1, u]

    # Inverse STFT to reconstruct the signal
    _, modified_signal = istft(new_Zxx_psm, nperseg=window_size, noverlap=window_size - hop_size)

    return modified_signal
# Function to read a WAV file using wave module

def phase_vocoder_2(signal, tsm_rate, psm_rate, window_size=1024, hop_size=256):
    # Compute STFT
    f, t, Zxx = stft(signal, nperseg=window_size, noverlap=window_size - hop_size)

    # Time-Scale Modification
    new_t_length = int(len(t) * tsm_rate)
    new_Zxx_tsm = np.zeros((Zxx.shape[0], new_t_length), dtype=Zxx.dtype)

    # Initial phase for each frequency bin
    last_phase = np.angle(Zxx[:, 0])
    hop_phase_advance = 2 * np.pi * hop_size * np.arange(Zxx.shape[0]) / window_size

    for u in range(1, new_t_length):
        u_original = min(int(u / tsm_rate), len(t) - 1)
        phase_advance = np.angle(Zxx[:, u_original]) - last_phase - hop_phase_advance
        # Phase unwrapping
        phase_advance -= 2 * np.pi * np.round(phase_advance / (2 * np.pi))
        last_phase += phase_advance + hop_phase_advance
        new_Zxx_tsm[:, u] = np.abs(Zxx[:, u_original]) * np.exp(1j * last_phase)

    # Pitch-Scale Modification
    new_Zxx_psm = np.zeros_like(new_Zxx_tsm, dtype=new_Zxx_tsm.dtype)

    for k in range(Zxx.shape[0]):
        k_psm = int(k * psm_rate)
        if k_psm < Zxx.shape[0]:
            new_Zxx_psm[k, :] = new_Zxx_tsm[k_psm, :]

    # Inverse STFT to reconstruct the signal
    _, modified_signal = istft(new_Zxx_psm, nperseg=window_size, noverlap=window_size - hop_size)

    return modified_signal

def read_wave(filename):
    with wave.open(filename, 'rb') as wave_file:
        params = wave_file.getparams()
        n_channels, sample_width, frame_rate, n_frames = params[:4]

        frames = wave_file.readframes(n_frames)
        if sample_width == 2:
            dtype = np.int16
        elif sample_width == 4:
            dtype = np.int32
        else:
            raise ValueError("Unsupported sample width")

        audio_signal = np.frombuffer(frames, dtype)

        if n_channels > 1:
            # Converting stereo to mono by averaging the channels
            audio_signal = audio_signal.reshape(-1, n_channels)
            audio_signal = audio_signal.mean(1)

        return frame_rate, audio_signal

# Function to write a WAV file using wave module
def write_wave(filename, rate, signal):
    with wave.open(filename, 'wb') as wave_file:
        n_channels, sample_width = 1, 2  # Mono, 16-bit samples
        wave_file.setnchannels(n_channels)
        wave_file.setsampwidth(sample_width)
        wave_file.setframerate(rate)
        wave_file.writeframes(signal.tobytes())

# Example usage
if __name__ == "__main__":
    # Read the input file

    sample_rate, audio_signal = read_wave('audio_samples/person talking.wav')
    # Ensure audio_signal is a 1-D array
    if audio_signal.ndim != 1:
        raise ValueError("Audio signal must be a 1-D array")

    # Time-stretch the audio (1.5 times longer)
    stretched_signal = phase_vocoder(audio_signal, 1.5, 1)

    # Normalize and clip the signal to the 16-bit range
    stretched_signal /= np.max(np.abs(stretched_signal), 0)
    stretched_signal = np.clip(stretched_signal, -1, 1)
    stretched_signal = (stretched_signal * 32767).astype(np.int16)

    # Save the result
    write_wave('stretched_audio.wav', sample_rate, stretched_signal)


    # Time-stretch the audio (1.5 times longer)
    stretched_signal = phase_vocoder_2(audio_signal, 1.5, 1)

    # Normalize and clip the signal to the 16-bit range
    stretched_signal /= np.max(np.abs(stretched_signal), 0)
    stretched_signal = np.clip(stretched_signal, -1, 1)
    stretched_signal = (stretched_signal * 32767).astype(np.int16)

    # Save the result
    write_wave('stretched_audio_article.wav', sample_rate, stretched_signal)

