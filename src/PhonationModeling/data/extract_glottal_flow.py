import importlib
import os
import sys

import librosa
import matplotlib.pyplot as plt
import numpy as np
from scipy.io import wavfile

# sys.path.append("creaky_voice/src/external")
from PhonationModeling.external.pypevoc.speech.glottal import iaif_ola

if __name__ == "__main__":
    # data_root = "data/creaky_voice/processed/phone_AA1_normal"
    # save_dir = "data/creaky_voice/glottal_flows/phone_AA1_normal"
    save_dir = "data/vocal_cord_paralysis/glottal_flows/4_gordon_boaz"
    data_root = "data/vocal_cord_paralysis/processed/4_gordon_boaz"
    try:
        os.makedirs(save_dir)
    except FileExistsError:
        print(f"folder {save_dir} already exists")

    ph_seg_lst = [
        line.rstrip()
        for line in open(
            # "src/PhonationModeling/data/filelists/phone_segs_AA1_normal.lst"
            # "src/PhonationModeling/data/filelists/vocal_paralysis_patient_4_gordon_boaz.lst"
            "src/PhonationModeling/data/filelists/vocal_paralysis_normal.lst"
        )
    ]

    for wf in ph_seg_lst:
        print(f"Processing {wf}")

        # Read wav
        sample_rate, wav = wavfile.read(os.path.join(data_root, wf))
        # if wav.dtype.name == "int16":
        # Convert from to 16-bit int to 32-bit float
        wav = (wav / pow(2, 15)).astype("float32")
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
