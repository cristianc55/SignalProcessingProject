import numpy as np
from scipy.signal import stft, istft

def phase_vocoder(audio, stretch_factor):
    # Parameters
    hop_length = 512
    window_length = 1024

    # STFT
    f, t, Zxx = stft(audio, window_length, window_length-hop_length)

    # Number of time steps in the stretched signal
    num_timesteps = int(np.ceil(len(t) * stretch_factor))

    # Initialize the output array
    stretched = np.zeros((len(f), num_timesteps), Zxx.dtype)

    # Initial phase
    last_phase = np.angle(Zxx[:, 0])
    cum_phase = last_phase.copy()

    for i in range(1, num_timesteps):
        # Determine the corresponding time index in the original signal
        orig_idx = int(i / stretch_factor)

        if orig_idx >= len(t) - 1:
            break

        # Phase difference
        delta_phi = np.angle(Zxx[:, orig_idx + 1]) - np.angle(Zxx[:, orig_idx])
        delta_phi = np.angle(np.exp(1j * delta_phi))  # Map to [-pi, pi] range

        # Phase prediction
        cum_phase += delta_phi

        # Magnitude (unchanged)
        magnitude = np.abs(Zxx[:, orig_idx])

        # Reconstruct the stretched signal
        stretched[:, i] = magnitude * np.exp(1j * cum_phase)

    # ISTFT
    _, audio_stretched = istft(stretched, window_length-hop_length)
    return audio_stretched

if __name__ == "__main__":
    import scipy.io.wavfile as wav

    # Load an audio file (replace 'audio_file.wav' with your file)
    sample_rate, audio_signal = wav.read('audio_samples/piano A4.wav')

    # Ensure mono audio
    if len(audio_signal.shape) > 1:
        audio_signal = audio_signal[:,0]

    # Time-stretch the audio (1.5 times longer)
    stretched_signal = phase_vocoder(audio_signal, 1.5)

    # Save the result
    wav.write('stretched_audio.wav', sample_rate, stretched_signal.astype(np.int16))