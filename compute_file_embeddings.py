#!/usr/bin/env python3
"""
Copyright 2024 Ciira wa Maina, DeKUT DSAIL

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""
import os
import argparse
import subprocess
import librosa

import numpy as np

from tqdm import tqdm
from scipy import signal

import bvc_helper_funcs as bhf

def main():
    """Extract embeddings from files in input directory
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-i', '--input_dir', help='Input .wav file directory.', required=True, type=str)
    parser.add_argument(
        '-s', '--sep_dir', help='Separated .wav file directory.', required=True, type=str)
    parser.add_argument(
        '-e', '--embedding_dir', help='Embedding directory.', required=True, type=str)
    args = parser.parse_args()

    SAMPLING_RATE = 32000

    # parameters
    window_ms = 40
    hop_ms = 10
    window_samples = int((window_ms * SAMPLING_RATE) / 1000)
    hop_length = int((hop_ms * SAMPLING_RATE) / 1000)
    n_fft = int(2 ** np.ceil((np.log2(window_samples))))

    num_sources = 8
    energy_quantile = .95
    seg_length_ms = 5000 # 5 seconds for use in the model
    max_width = int(seg_length_ms / hop_ms)
    half_seglen_s = (seg_length_ms / 1000) * .5
    spectral_centroid_thresh = 500
    audio_files = os.listdir(args.input_dir)

    # load model
    model, _ = bhf.load_model()
    for audio_file in tqdm(audio_files):
        # run source separation
        cmd = ["python3",
               "sound-separation/models/tools/process_wav.py",
               "--model_dir",
               "bird_mixit_model_checkpoints/output_sources8",
               "--checkpoint",
               "bird_mixit_model_checkpoints/output_sources8/model.ckpt-2178900",
               "--num_sources",
               "8",
               "--input",
               os.path.join(args.input_dir, audio_file),
               "--output",
               os.path.join(args.sep_dir, audio_file)]

        subprocess.run(cmd, check=True)
        file_embeddings = np.array([])
        labels = []
        for file_indx, file_path in enumerate([os.path.join(args.sep_dir,
                                                            audio_file.replace('.wav',
                                                                               '_source{}.wav'.format(i))) for i in range(num_sources)]):

            audio, sampling_rate = librosa.load(file_path, sr=SAMPLING_RATE)
            _, wav_data = bhf.ensure_sample_rate(audio, sampling_rate)

            rms = librosa.feature.rms(y=wav_data,
                                      frame_length=n_fft,
                                      hop_length=hop_length)
            peakind = signal.find_peaks_cwt(rms[0,:], np.arange(1, max_width))
            ul = np.quantile(rms, energy_quantile)

            # to remove noise dominated files, compute the spectral cetroid
            # use audio files where centroid > 500 Hz
            S = librosa.feature.melspectrogram(y=wav_data,
                                               sr=sampling_rate,
                                               n_fft=n_fft,
                                               hop_length=hop_length,
                                               n_mels=128,
                                               fmax=8000)
            cent = librosa.feature.spectral_centroid(S=S)

            for indx in peakind:
                if rms[0, indx] > ul and np.mean(cent) > spectral_centroid_thresh:
                    t = (indx * hop_ms) / 1000
                    if t > half_seglen_s:
                        start_indx = int(np.floor((t - half_seglen_s) * sampling_rate))
                        stop_indx = int(np.ceil((t + half_seglen_s) * sampling_rate))
                        stop_indx = np.min([stop_indx, len(audio)])

                        test_frames = bhf.frame_audio(wav_data[start_indx:stop_indx], hop_size_s=5)
                        segment_embedding = bhf.process_audio_frames(test_frames, model)

                        if file_embeddings.size:
                            file_embeddings = np.vstack((np.array(segment_embedding[0,:]), file_embeddings))
                            labels.append(file_indx)
                        else:
                            file_embeddings = np.array(segment_embedding[0,:])
                            labels.append(file_indx)

        np.savez(os.path.join(args.embedding_dir,
                             audio_file.replace('wav','npz')),
                 embeddings=file_embeddings,
                 labels=np.array(labels))

if __name__ == '__main__':
    main()
