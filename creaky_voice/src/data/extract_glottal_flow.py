import importlib
import os
import sys

import librosa
import matplotlib.pyplot as plt
import numpy as np
from scipy.io import wavfile

sys.path.append("creaky_voice/src/external")
from pypevoc.speech.glottal import iaif_ola


if __name__ == "__main__":
    data_root = "creaky_voice/data/processed/phone_L_normal"
    save_dir = "creaky_voice/data/glottal_flows/phone_L_normal"
    ph_seg_lst = [line.rstrip() for line in open("creaky_voice/src/data/phone_segs_L_normal.lst")]

    for wf in ph_seg_lst:
        print(f"Processing {wf}")

        # Read wav
        sample_rate, wav = wavfile.read(os.path.join(data_root, wf))
        # wav = librosa.resample(wav, sample_rate, 16000)  # NOTE: downsample?

        # Extract glottal flow
        g, d_g, vt_coef, g_coef = iaif_ola(
            wav,
            Fs=sample_rate,
            tract_order=2 * int(np.round(sample_rate / 2000)) + 4,
            glottal_order=2 * int(np.round(sample_rate / 4000)),
        )

        # Plot
        # t = np.arange(len(wav)) / sample_rate
        # fig = plt.figure()
        # ax = fig.add_subplot(111)
        # ax.plot(t, wav, "c")
        # ax.plot(t, np.linalg.norm(wav) * g / np.linalg.norm(g), "r")
        # plt.show()

        # Save
        np.save(os.path.join(save_dir, wf.replace(".wav", ".npy", 1)), g)
        print("Done")
