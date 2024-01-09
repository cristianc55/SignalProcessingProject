import numpy as np
import os
from scipy.signal import stft, istft
import wave

def phase_vocoder_article(signal, rate, window_size=1024, hop_size=256):
    """
    Modified phase vocoder for time-stretching an audio signal based on the new method.

    :param signal: Input audio signal (1D numpy array).
    :param rate: Time-stretching factor (greater than 1 for stretching, less than 1 for shrinking).
    :param window_size: Size of the STFT window.
    :param hop_size: Hop size for STFT.
    :return: Time-stretched audio signal.
    """
    if len(signal.shape) > 1:
        raise ValueError("Input signal must be a 1-D array")

    frequencies, times, stft_matrix = stft(signal, window_size)

    columns = int(times.size * rate)
    stretched_stft_matrix = np.zeros((frequencies.size, columns), np.complex_)

    time_stride = times[1] - times[0]
    for t in range(columns-1):
        col_idx = int(t / rate)
        if col_idx < times.size - 1:
            # Direct phase difference calculation
            phase_diff = np.angle(stft_matrix[:, col_idx+1]) - np.angle(stft_matrix[:, col_idx])
            phase = np.angle(stft_matrix[:, col_idx]) + phase_diff * rate
            stretched_stft_matrix[:, t+1] = np.abs(stft_matrix[:, col_idx]) * np.exp(1j * phase)

    _, stretched_signal = istft(stretched_stft_matrix, window_size-hop_size)

    return stretched_signal

# Function to read a WAV file using wave module
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
    stretched_signal_article = phase_vocoder_article(audio_signal, 1.25)

    # Normalize and clip the signal to the 16-bit range
    stretched_signal_article /= np.max(np.abs(stretched_signal_article), 0)
    stretched_signal_article = np.clip(stretched_signal_article, -1, 1)
    stretched_signal_article = (stretched_signal_article * 32767).astype(np.int16)

    # Save the result
    write_wave('stretched_audio_2.wav', sample_rate, stretched_signal_article)

    os.system("afplay " + "person_talking.wav")
    # os.system("afplay " + "stretched_audio.wav")

