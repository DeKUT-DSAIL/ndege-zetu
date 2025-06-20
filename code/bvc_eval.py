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
import pandas as pd

from tqdm import tqdm
import bvc_helper_funcs as bhf

def ebird_species_predictions(logits, ebird_labels, ke_ebird):
    ebird_codes = []

    ke_ebird_codes = list(ke_ebird['ebird code'])
    ke_sn = list(ke_ebird['Scientific Name'])
    ke_cn = list(ke_ebird['Common Name'])

    for i in range(logits.shape[0]):
        ebird_codes.append(ebird_labels[np.argmax(logits[i,:])])

    recording_sn = []
    recording_cn = []
    
    for eb in ebird_codes:
        if eb in ke_ebird_codes:
            recording_sn.append(ke_sn[ke_ebird_codes.index(eb)])
            recording_cn.append(ke_cn[ke_ebird_codes.index(eb)])
            
    
    return recording_sn, recording_cn

def species_from_df(df):
    '''
    This function extracts species from the audio annotation files
    '''
    sps = []
    for i in range(len(df)):
        if not pd.isna(df.iloc[i]['Foreground Species']):
            sps += df.iloc[i]['Foreground Species'].split(';')
        if not pd.isna(df.iloc[i]['Background Species']):
            sps += df.iloc[i]['Background Species'].split(';')
    return list(set(sps))


def main():
    """Evaluate performance of Google Perch model on Kenyan data
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-i', '--input_dir', help='Input .mp3 file directory.', required=True, type=str)
    parser.add_argument(
        '-a', '--annotations_dir', help='Annotation directory (full path).', required=True, type=str)
    args = parser.parse_args()

    annotation_files = os.listdir(args.annotations_dir)
    
    # audio 
    aru_dfs = []
    for filename in annotation_files:
        if 'aru' in filename  and filename.endswith('csv'):
        
            df = pd.read_csv(os.path.join(args.annotations_dir, filename))
            aru_dfs.append(df)
        
    annotations = pd.concat(aru_dfs)
    aru_cns = species_from_df(annotations)
    aru_cns.sort()
    
    # scientific names
    df_sp = pd.read_csv(os.path.join(args.annotations_dir,'Kenya-Species-List.csv'))
    aru_sns = [df_sp[df_sp['Common Name']==cn].iloc[0]['Scientific Name'] for cn in aru_cns]


    # ebird codes for Kenyan species
    df_ebird = pd.read_csv(os.path.join(args.annotations_dir, 'ebird_ke.csv'))

    # get number of files with species
    num_files = 0
    for indx, filename in enumerate(tqdm(list(annotations['Filename']))):

        recording_cn = []
        if not pd.isna(annotations.iloc[indx]['Foreground Species']):
            recording_cn += annotations.iloc[indx]['Foreground Species'].split(';')
        if not pd.isna(annotations.iloc[indx]['Background Species']):
            recording_cn += annotations.iloc[indx]['Background Species'].split(';')
        
        if len(recording_cn):
            num_files += 1

    classification_result = np.zeros((num_files, len(aru_sns)))
    recording_species = np.zeros((num_files, len(aru_sns)))
    print(classification_result.shape)

   
    curr_file = 0
    model, labels_path = bhf.load_model()
    ebird_labels = pd.read_csv(labels_path)
    print(ebird_labels)
    for indx, filename in enumerate(tqdm(list(annotations['Filename']))):

        recording_cn = []
        if not pd.isna(annotations.iloc[indx]['Foreground Species']):
            recording_cn += annotations.iloc[indx]['Foreground Species'].split(';')
        if not pd.isna(annotations.iloc[indx]['Background Species']):
            recording_cn += annotations.iloc[indx]['Background Species'].split(';')

        recording_sn = [df_sp[df_sp['Common Name']==cn].iloc[0]['Scientific Name'] for cn in recording_cn]

        
        if len(recording_cn):
            audio, sampling_rate = librosa.load(os.path.join(args.input_dir, filename), sr=32000)
            _, wav_data = bhf.ensure_sample_rate(audio, sampling_rate)
            test_frames = bhf.frame_audio(wav_data, hop_size_s=5)
           
            all_logits, _ = model.infer_tf(test_frames[:1])
            for window in test_frames[1:-1]:
                logits, _ = model.infer_tf(window[np.newaxis, :])
                all_logits = np.concatenate([all_logits, logits], axis=0)

            pred_recording_sn, pred_recording_cn = ebird_species_predictions(all_logits, ebird_labels['ebird2021'], df_ebird)
            print(filename, recording_sn, pred_recording_sn, pred_recording_cn)


            for sp in recording_sn:
                recording_species[curr_file, aru_sns.index(sp)] = 1

            for sn, cn in zip(pred_recording_sn, pred_recording_cn):
                if sn  in aru_sns:
                    classification_result[curr_file, aru_sns.index(sn)] = 1
                if cn in aru_cns:
                    classification_result[curr_file, aru_cns.index(cn)] = 1

            
            curr_file += 1

    np.save('aru_gt_bvc.npy', recording_species)
    np.save('aru_pred_bvc.npy', classification_result)

        


if __name__ == '__main__':
    main()
