import numpy as np
import os
from scipy.signal import stft, istft
import wave
from scipy.signal import spectrogram
from scipy.io import wavfile
import librosa
import matplotlib.pyplot as plt
import librosa.display

def phase_vocoder_article(signal, tsm_rate, psm_rate, window_size=1024, hop_size=256):
    if len(signal.shape) > 1:
        raise ValueError("Input signal must be a 1-D array")

    frequencies, times, stft_matrix = stft(signal, nperseg=window_size, noverlap=window_size - hop_size)

    # Time-Scale Modification
    tsm_columns = int(times.size * tsm_rate)
    stretched_stft_matrix_tsm = np.zeros((frequencies.size, tsm_columns), np.complex_)

    for t in range(1, tsm_columns):
        col_idx = min(int(t / tsm_rate), times.size - 1)
        phase_diff = np.angle(stft_matrix[:, col_idx]) - np.angle(stft_matrix[:, col_idx - 1])
        stretched_stft_matrix_tsm[:, t] = np.abs(stft_matrix[:, col_idx]) * np.exp(
            1j * (np.angle(stretched_stft_matrix_tsm[:, t - 1]) + phase_diff))

    # Pitch-Scale Modification
    psm_columns = stretched_stft_matrix_tsm.shape[1]
    stretched_stft_matrix_psm = np.zeros((frequencies.size, psm_columns), np.complex_)

    for k in range(frequencies.size):
        k_psm = int(k * psm_rate)
        if k_psm < frequencies.size:
            stretched_stft_matrix_psm[k, :] = stretched_stft_matrix_tsm[k_psm, :]

    _, modified_signal = istft(stretched_stft_matrix_psm, nperseg=window_size, noverlap=window_size - hop_size)

    return modified_signal


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


if __name__ == "__main__":
    # Read the input file

    sample_rate, audio_signal = read_wave('../audio_samples/123_PS_sample1.wav')
    # Ensure audio_signal is a 1-D array
    if audio_signal.ndim != 1:
        raise ValueError("Audio signal must be a 1-D array")

    # Time-stretch the audio (1.5 times longer)
    stretched_signal = phase_vocoder_article(audio_signal, 1.5, 1)

    # Normalize and clip the signal to the 16-bit range
    stretched_signal /= np.max(np.abs(stretched_signal), 0)
    stretched_signal = np.clip(stretched_signal, -1, 1)
    stretched_signal = (stretched_signal * 32767).astype(np.int16)

    # Save the result
    write_wave('123_PS_sample1_vocoder_articol.wav', sample_rate, stretched_signal)

    sample_rate, audio = read_wave('123_PS_sample1_vocoder_articol.wav')

    frequencies, times, Sxx = spectrogram(audio, sample_rate)

    plt.pcolormesh(times, frequencies, 10 * np.log10(Sxx), shading='gouraud')
    plt.ylabel('Frequency [Hz]')
    plt.xlabel('Time [sec]')
    plt.title('Spectrograma 123 sample 1 vocoder articol')
    plt.colorbar(label='Intensitate (dB)')
    plt.show()

    sample_rate, audio = read_wave('../audio_samples/123_PS_sample1.wav')

    frequencies, times, Sxx = spectrogram(audio, sample_rate)

    plt.pcolormesh(times, frequencies, 10 * np.log10(Sxx), shading='gouraud')
    plt.ylabel('Frequency [Hz]')
    plt.xlabel('Time [sec]')
    plt.title('Spectrograma 123 sample 1')
    plt.colorbar(label='Intensitate (dB)')
    plt.show()

    sampling_rate, data = wavfile.read('../audio_samples/123_PS_sample1.wav')

    # Step 2: Handle stereo audio by converting to mono
    if len(data.shape) > 1:
        data = data.mean(axis=1)

    # Step 3: Create a time array
    time = np.arange(0, len(data)) / sampling_rate

    # Step 4: Plot the waveform
    plt.figure(figsize=(12, 4))
    plt.plot(time, data)
    plt.xlabel('Time [s]')
    plt.ylabel('Amplitude')
    plt.title('Waveform original 123 Sample 1')
    plt.show()

    sampling_rate, data = wavfile.read('123_PS_sample1_vocoder_articol.wav')

    # Step 2: Handle stereo audio by converting to mono
    if len(data.shape) > 1:
        data = data.mean(axis=1)

    # Step 3: Create a time array
    time = np.arange(0, len(data)) / sampling_rate

    # Step 4: Plot the waveform
    plt.figure(figsize=(12, 4))
    plt.plot(time, data)
    plt.xlabel('Time [s]')
    plt.ylabel('Amplitude')
    plt.title('Waveform vocoder articol 123 Sample 1')
    plt.show()

    # Sample 2
    ######################################
    ######################################

    sample_rate, audio_signal = read_wave('../audio_samples/123_Robot_PS_sample2.wav')
    # Ensure audio_signal is a 1-D array
    if audio_signal.ndim != 1:
        raise ValueError("Audio signal must be a 1-D array")

    # Time-stretch the audio (1.5 times longer)
    stretched_signal = phase_vocoder_article(audio_signal, 1.5, 1)

    # Normalize and clip the signal to the 16-bit range
    stretched_signal /= np.max(np.abs(stretched_signal), 0)
    stretched_signal = np.clip(stretched_signal, -1, 1)
    stretched_signal = (stretched_signal * 32767).astype(np.int16)

    # Save the result
    write_wave('123_Robot_PS_sample2_vocoder_articol.wav', sample_rate, stretched_signal)

    sample_rate, audio = read_wave('123_Robot_PS_sample2_vocoder_articol.wav')


    frequencies, times, Sxx = spectrogram(audio, sample_rate)

    plt.pcolormesh(times, frequencies, 10 * np.log10(Sxx), shading='gouraud')
    plt.ylabel('Frequency [Hz]')
    plt.xlabel('Time [sec]')
    plt.title('Spectrograma 123 Robot sample 2 vocoder articol')
    plt.colorbar(label='Intensitate (dB)')
    plt.show()

    sample_rate, audio = read_wave('../audio_samples/123_Robot_PS_sample2.wav')

    frequencies, times, Sxx = spectrogram(audio, sample_rate)

    plt.pcolormesh(times, frequencies, 10 * np.log10(Sxx), shading='gouraud')
    plt.ylabel('Frequency [Hz]')
    plt.xlabel('Time [sec]')
    plt.title('Spectrograma 123 Robot sample 2')
    plt.colorbar(label='Intensitate (dB)')
    plt.show()

    sampling_rate, data = wavfile.read('../audio_samples/123_Robot_PS_sample2.wav')

    # Step 2: Handle stereo audio by converting to mono
    if len(data.shape) > 1:
        data = data.mean(axis=1)

    # Step 3: Create a time array
    time = np.arange(0, len(data)) / sampling_rate

    # Step 4: Plot the waveform
    plt.figure(figsize=(12, 4))
    plt.plot(time, data)
    plt.xlabel('Time [s]')
    plt.ylabel('Amplitude')
    plt.title('Waveform original 123 Robot Sample 2')
    plt.show()

    sampling_rate, data = wavfile.read('123_Robot_PS_sample2_vocoder_articol.wav')

    # Step 2: Handle stereo audio by converting to mono
    if len(data.shape) > 1:
        data = data.mean(axis=1)

    # Step 3: Create a time array
    time = np.arange(0, len(data)) / sampling_rate

    # Step 4: Plot the waveform
    plt.figure(figsize=(12, 4))
    plt.plot(time, data)
    plt.xlabel('Time [s]')
    plt.ylabel('Amplitude')
    plt.title('Waveform vocoder articol 123 Robot Sample 2')
    plt.show()

    # Sample 5
    ######################################
    ######################################

    sample_rate, audio_signal = read_wave('../audio_samples/Birds_Chirping_PS_sample5.wav')
    # Ensure audio_signal is a 1-D array
    if audio_signal.ndim != 1:
        raise ValueError("Audio signal must be a 1-D array")

    # Time-stretch the audio (1.5 times longer)
    stretched_signal = phase_vocoder_article(audio_signal, 1.5, 3)

    # Normalize and clip the signal to the 16-bit range
    stretched_signal /= np.max(np.abs(stretched_signal), 0)
    stretched_signal = np.clip(stretched_signal, -1, 1)
    stretched_signal = (stretched_signal * 32767).astype(np.int16)

    # Save the result
    write_wave('Birds_Chirping_PS_sample5_vocoder_articol.wav', sample_rate, stretched_signal)

    sample_rate, audio = read_wave('Birds_Chirping_PS_sample5_vocoder_articol.wav')

    frequencies, times, Sxx = spectrogram(audio, sample_rate)

    plt.pcolormesh(times, frequencies, 10 * np.log10(Sxx), shading='gouraud')
    plt.ylabel('Frequency [Hz]')
    plt.xlabel('Time [sec]')
    plt.title('Spectrograma Birds_Chirping_PS_sample5_vocoder_articol')
    plt.colorbar(label='Intensitate (dB)')
    plt.show()

    sample_rate, audio = read_wave('../audio_samples/Birds_Chirping_PS_sample5.wav')

    frequencies, times, Sxx = spectrogram(audio, sample_rate)

    plt.pcolormesh(times, frequencies, 10 * np.log10(Sxx), shading='gouraud')
    plt.ylabel('Frequency [Hz]')
    plt.xlabel('Time [sec]')
    plt.title('Spectrograma Birds_Chirping_PS_sample5')
    plt.colorbar(label='Intensitate (dB)')
    plt.show()

    sampling_rate, data = wavfile.read('../audio_samples/Birds_Chirping_PS_sample5.wav')

    # Step 2: Handle stereo audio by converting to mono
    if len(data.shape) > 1:
        data = data.mean(axis=1)

    # Step 3: Create a time array
    time = np.arange(0, len(data)) / sampling_rate

    # Step 4: Plot the waveform
    plt.figure(figsize=(12, 4))
    plt.plot(time, data)
    plt.xlabel('Time [s]')
    plt.ylabel('Amplitude')
    plt.title('Waveform original Birds_Chirping_PS_sample5.wav')
    plt.show()

    sampling_rate, data = wavfile.read('Birds_Chirping_PS_sample5_vocoder_articol.wav')

    # Step 2: Handle stereo audio by converting to mono
    if len(data.shape) > 1:
        data = data.mean(axis=1)

    # Step 3: Create a time array
    time = np.arange(0, len(data)) / sampling_rate

    # Step 4: Plot the waveform
    plt.figure(figsize=(12, 4))
    plt.plot(time, data)
    plt.xlabel('Time [s]')
    plt.ylabel('Amplitude')
    plt.title('Waveform vocoder normal Birds_Chirping_PS_sample5_vocoder_articol')
    plt.show()

    # C major

    sample_rate, audio_signal = read_wave('../audio_samples/C major scale.wav')
    # Ensure audio_signal is a 1-D array
    if audio_signal.ndim != 1:
        raise ValueError("Audio signal must be a 1-D array")

    # Time-stretch the audio (1.5 times longer)
    stretched_signal = phase_vocoder_article(audio_signal, 1.5, 1)

    # Normalize and clip the signal to the 16-bit range
    stretched_signal /= np.max(np.abs(stretched_signal), 0)
    stretched_signal = np.clip(stretched_signal, -1, 1)
    stretched_signal = (stretched_signal * 32767).astype(np.int16)

    # Save the result
    write_wave('C major scale article.wav', sample_rate, stretched_signal)

    sample_rate, audio = read_wave('C major scale article.wav')

    frequencies, times, Sxx = spectrogram(audio, sample_rate)

    plt.pcolormesh(times, frequencies, 10 * np.log10(Sxx), shading='gouraud')
    plt.ylabel('Frequency [Hz]')
    plt.xlabel('Time [sec]')
    plt.title('Spectrograma C Major articol')
    plt.colorbar(label='Intensitate (dB)')
    plt.show()

    sampling_rate, data = wavfile.read('C major scale article.wav')

    # Step 2: Handle stereo audio by converting to mono
    if len(data.shape) > 1:
        data = data.mean(axis=1)

    # Step 3: Create a time array
    time = np.arange(0, len(data)) / sampling_rate

    # Step 4: Plot the waveform
    plt.figure(figsize=(12, 4))
    plt.plot(time, data)
    plt.xlabel('Time [s]')
    plt.ylabel('Amplitude')
    plt.title('Waveform vocoder C Major articol')
    plt.show()
