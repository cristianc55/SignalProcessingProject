from scipy.io import wavfile
import numpy as np


def calculate_snr(signal, reference):
    power_signal = np.sum(reference ** 2) / len(reference)
    power_noise = np.sum((reference - signal) ** 2) / len(signal)

    snr = 10 * np.log10(power_signal / power_noise)
    return snr


rate, original = wavfile.read('123_PS_sample1_vocoder_normal.wav')
rate, processed = wavfile.read('123_PS_sample1_vocoder_articol.wav')

min_length = min(len(original), len(processed))
original = original[:min_length]
processed = processed[:min_length]

snr_value = calculate_snr(processed, original)
print(f"SNR Sample1: {snr_value} dB")

rate, original = wavfile.read('123_Robot_PS_sample2_vocoder_normal.wav')
rate, processed = wavfile.read('123_Robot_PS_sample2_vocoder_articol.wav')

min_length = min(len(original), len(processed))
original = original[:min_length]
processed = processed[:min_length]

snr_value = calculate_snr(processed, original)
print(f"SNR Sample2: {snr_value} dB")

rate, original = wavfile.read('Birds_Chirping_PS_sample5_vocoder_normal.wav')
rate, processed = wavfile.read('Birds_Chirping_PS_sample5_vocoder_articol.wav')

min_length = min(len(original), len(processed))
original = original[:min_length]
processed = processed[:min_length]

snr_value = calculate_snr(processed, original)
print(f"SNR Sample5: {snr_value} dB")

rate, original = wavfile.read('C major scale normal.wav')
rate, processed = wavfile.read('C major scale article.wav')

min_length = min(len(original), len(processed))
original = original[:min_length]
processed = processed[:min_length]

snr_value = calculate_snr(processed, original)
print(f"SNR Sample C major: {snr_value} dB")
