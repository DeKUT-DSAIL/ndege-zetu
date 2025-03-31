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
import json
import argparse
import librosa

import numpy as np


from tqdm import tqdm

import bvc_helper_funcs as bhf

def main():
    """Extract embeddings from files in input directory
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-i', '--input_dir', help='Input .mp3 file directory.', required=True, type=str)
    parser.add_argument(
        '-e', '--embeddings_dir', help='Embeddings directory.', required=True, type=str)
    parser.add_argument(
        '-a', '--annotation_file', help='Annotation file (full path).', required=False, type=str)
    args = parser.parse_args()

    if args.annotation_file is not None:
        with open(args.annotation_file, 'r') as fp:
            single_species_filenames = json.load(fp)
        file_list = list(single_species_filenames.keys())
    else:
        file_list = os.listdir(args.input_dir)
        
    model, _ = bhf.load_model()

    for filename in tqdm(file_list):
        print(filename)
        audio, sampling_rate = librosa.load(os.path.join(args.input_dir, filename), sr=32000)
        _, wav_data = bhf.ensure_sample_rate(audio, sampling_rate)
        test_frames = bhf.frame_audio(wav_data, hop_size_s=5)
        file_embeddings = bhf.process_audio_frames(test_frames[:-1,:], model)

        np.savez(os.path.join(args.embeddings_dir,
                              filename.replace('mp3','npz')),
                 embeddings=file_embeddings)


if __name__ == '__main__':
    main()
