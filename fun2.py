import soundfile as sf
import cv2
import numpy as np
import librosa
import matplotlib.pyplot as plt

def image_to_audio_convert(img_filename, audio_file):
    print("Converting image to a WAV file...")
    filepath = "./" + img_filename
    image = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)  # read a given image file
    image_size = [image.shape[0], image.shape[1]]  # extra list with image's shape values

    # soundfile parameters
    max_freq = 20000
    min_freq = 20
    sample_rate = 44100  # number of samples per second
    duration = 3  # duration of the entire WAV file in seconds

    freq_list = np.linspace(max_freq, min_freq, num=image_size[0])  # list of generated frequencies
    sound_data = np.zeros(sample_rate*duration)

    # samples_per_pixel = (sample_rate*duration)//image_size[1]  # number of samples appended to the end of an array

    # Sound generation using Coagula algorithm
    for y in range(image_size[0]):
        #samples_array = np.zeros(sample_rate*duration)
        #for x in range(image_size[1]):
        #    if image[y, x] == 0x01:  # if the pixel is white set tone_duration to a set value
        #        np.insert(samples_array, obj=x*samples_per_pixel, values=[librosa.tone(freq_list[y], sr=sample_rate, length=samples_per_pixel)])

        # if tone_duration != 0:  # vectors cannot be added together if one of them has a length of zero
        sound_data += librosa.tone(freq_list[y], sr=sample_rate, length=sample_rate*duration)
        #sound_data += samples_array

    sf.write(audio_file,  data=sound_data, samplerate=sample_rate, subtype='PCM_24')  # write to a 24bit PCM WAV file


# Calculate and display a spectrogram of a file
def plot_spectrogram(audio_filename):
    print("Calculating a waveform spectrogram")
    # Calculate a spectrogram of an example file
    signal, sample_rate = librosa.load("./" + audio_filename)
    stft = librosa.stft(signal)  # an STFT of an example file
    scale_db = librosa.amplitude_to_db(np.abs(stft), ref=np.max)

    # Display a spectrogram of a file
    plt.figure()
    librosa.display.specshow(scale_db)
    plt.colorbar()
    plt.show()
