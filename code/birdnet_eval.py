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

from pathlib import Path
from tqdm import tqdm
from birdnet import SpeciesPredictions, predict_species_within_audio_file

def species_predictions(predictions):
    recording_species = []
    for k in predictions.keys():
        if len(predictions[k]):
            recording_species += list(predictions[k].keys())
    
    recording_sn = [sp.split('_')[0] for sp in recording_species]
    recording_cn = [sp.split('_')[1] for sp in recording_species]
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
    """Evaluate performance of BirdNET on Kenyan data
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
    df_sp = pd.read_csv('../annotations/Kenya-Species-List.csv')
    aru_sns = [df_sp[df_sp['Common Name']==cn].iloc[0]['Scientific Name'] for cn in aru_cns]


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
    for indx, filename in enumerate(tqdm(list(annotations['Filename']))):

        recording_cn = []
        if not pd.isna(annotations.iloc[indx]['Foreground Species']):
            recording_cn += annotations.iloc[indx]['Foreground Species'].split(';')
        if not pd.isna(annotations.iloc[indx]['Background Species']):
            recording_cn += annotations.iloc[indx]['Background Species'].split(';')

        recording_sn = [df_sp[df_sp['Common Name']==cn].iloc[0]['Scientific Name'] for cn in recording_cn]

        
        audio_path = Path(os.path.join(args.input_dir, filename))
        if len(recording_cn):
            predictions = SpeciesPredictions(predict_species_within_audio_file(audio_path))
            pred_recording_sn, pred_recording_cn = species_predictions(predictions)

            print(filename, recording_sn, pred_recording_sn, pred_recording_cn)

            for sp in recording_sn:
                recording_species[curr_file, aru_sns.index(sp)] = 1

            for sn, cn in zip(pred_recording_sn, pred_recording_cn):
                if sn  in aru_sns:
                    classification_result[curr_file, aru_sns.index(sn)] = 1
                if cn in aru_cns:
                    classification_result[curr_file, aru_cns.index(cn)] = 1

                
            curr_file += 1

    np.save('aru_gt.npy', recording_species)
    np.save('aru_pred.npy', classification_result)

        


if __name__ == '__main__':
    main()
