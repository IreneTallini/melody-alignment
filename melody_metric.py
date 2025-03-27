import librosa
from matplotlib import pyplot as plt
import note_seq
import numpy as np
from librosa.sequence import dtw
import soundfile as sf
import logging
from termcolor import colored

logging.basicConfig(level=logging.INFO)

class ChromaSimilarityCalculator:
    def __init__(self, sr=32000, hop_length=1024):
        logging.info("Initializing ChromaSimilarityCalculator with sr=%d, hop_length=%d", sr, hop_length)
        self.sr = sr
        self.hop_length = hop_length

    def calculate_similarity(self, wav1, wav2, exp_name="midi"):
        logging.info("Calculating similarity")
        # Extract chroma features
        chroma_fast = librosa.feature.chroma_cqt(y=wav1, sr=self.sr, hop_length=self.hop_length)
        chroma_slow = librosa.feature.chroma_cqt(y=wav2, sr=self.sr, hop_length=self.hop_length)
        chroma_fast = chroma_fast[:, ~np.all(chroma_fast == 0, axis=0)]
        chroma_slow = chroma_slow[:, ~np.all(chroma_slow == 0, axis=0)]
        
        fig, ax = plt.subplots(nrows=2, sharey=True, gridspec_kw={'hspace': 0.5})
        img = librosa.display.specshow(chroma_fast, x_axis='time',
                                    y_axis='chroma',
                                    hop_length=self.hop_length, ax=ax[0])
        ax[0].set(title='Chroma Representation of $X_1$')
        librosa.display.specshow(chroma_slow, x_axis='time',
                                y_axis='chroma',
                                hop_length=self.hop_length, ax=ax[1])
        ax[1].set(title='Chroma Representation of $X_2$')
        fig.colorbar(img, ax=ax)
        fig.savefig(exp_name + "_chroma.png")

        logging.info("Chroma features extracted. Computing DTW...")
        # Compute DTW on Chroma Features
        D_chroma, wp_chroma = dtw(X=chroma_fast, Y=chroma_slow, metric='cosine')
        alignment_cost_chroma = D_chroma[-1, -1]
        similarity_chroma = 1 / (1 + alignment_cost_chroma)
        logging.info("DTW computed. Alignment cost: %f, similarity: %f", alignment_cost_chroma, similarity_chroma)

        # Plot and save the warping path
        wp_s = librosa.frames_to_time(wp_chroma, sr=self.sr, hop_length=self.hop_length)
        fig, ax = plt.subplots()
        img = librosa.display.specshow(D_chroma, x_axis='time', y_axis='time', sr=self.sr,
                                       cmap='gray_r', ax=ax, hop_length=self.hop_length)
        ax.plot(wp_s[:, 1], wp_s[:, 0], marker='o', color='r')
        ax.set(title='Warping Path on Acc. Cost Matrix $D$',
               xlabel='Time $(X_2)$', ylabel='Time $(X_1)$')
        fig.colorbar(img, ax=ax)
        logging.info("Saving warping path figure to %s", exp_name + "_chroma_dtw.png")
        fig.savefig(exp_name + "_chroma_dtw.png")

        return alignment_cost_chroma, similarity_chroma
    

HOP_LENGTH = 1024
SR = 32000
STRETCH_COEFF = 0.8
NOTE_REPLACEMENT_COEFF = 0.
ADDITIVE_NOISE_STD = 0.0

ns_slow = note_seq.midi_file_to_note_sequence("slakh_0.mid")
ns_fast = note_seq.midi_file_to_note_sequence("slakh_0.mid")

# stretch the second sequence
logging.info("Stretching second sequence by factor %f", STRETCH_COEFF)
c_inv = 1 / STRETCH_COEFF
ns_fast = note_seq.note_seq.sequences_lib.stretch_note_sequence(ns_fast, STRETCH_COEFF)

# estract subsequence of 10 seconds
ns_slow = note_seq.note_seq.sequences_lib.extract_subsequence(ns_slow, 10, 20)
ns_fast = note_seq.note_seq.sequences_lib.extract_subsequence(ns_fast, int(10 * STRETCH_COEFF), int(20 * STRETCH_COEFF))

# replace some of the pitches with a random pitch
count = 0
for note in ns_fast.notes:
    if np.random.rand() < NOTE_REPLACEMENT_COEFF:
        note.pitch = np.random.randint(0, 128)
        count += 1
logging.info("Replaced %d notes with random pitches", count)

# synthesize audio from the MIDI files
y_slow = note_seq.fluidsynth(ns_slow, sample_rate=SR)
y_fast = note_seq.fluidsynth(ns_fast, sample_rate=SR)
y_slow /= np.max(np.abs(y_slow))
y_fast /= np.max(np.abs(y_fast))

# load real audio WAVs 
y_real_slow, _ = librosa.load("slakh_0.wav", sr=SR)
y_real_slow = y_real_slow[SR * 10 : SR * 20]
y_real_fast = librosa.effects.time_stretch(y_real_slow, rate=c_inv)
y_real_slow /= np.max(np.abs(y_real_slow))
y_real_fast /= np.max(np.abs(y_real_fast))

# add noise
y_fast += np.random.normal(0, ADDITIVE_NOISE_STD, y_fast.shape)
y_real_fast += np.random.normal(0, ADDITIVE_NOISE_STD, y_real_fast.shape)
logging.info("Added noise with std: %f", ADDITIVE_NOISE_STD)

# save the audio files
sf.write("audio_slow.wav", y_slow, SR)
sf.write("audio_fast.wav", y_fast, SR)
sf.write("audio_real_fast.wav", y_real_fast, SR)
sf.write("audio_real_slow.wav", y_real_slow, SR)
logging.info("Saved audio files to audio_slow.wav, audio_fast.wav, audio_real_fast.wav, audio_real_slow.wav")

# test the ChromaSimilarityCalculator
csc = ChromaSimilarityCalculator(hop_length=HOP_LENGTH, sr=SR)
alignment_cost_chroma, similarity_chroma = csc.calculate_similarity(y_fast, y_slow, exp_name="midi")
print(colored("Chroma DTW cost: %.2f, similarity: %.3f" % (alignment_cost_chroma, similarity_chroma), 'green'))

# test the ChromaSimilarityCalculator on the real audio files
csc = ChromaSimilarityCalculator()
alignment_cost_chroma, similarity_chroma = csc.calculate_similarity(y_real_fast, y_real_slow, exp_name="real")
print(colored("Chroma DTW cost real data: %.2f, similarity: %.3f" % (alignment_cost_chroma, similarity_chroma), 'green'))
