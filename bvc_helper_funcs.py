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
import librosa
import numpy as np
import tensorflow_io as tfio
import tensorflow as tf
import tensorflow_hub as hub


def frame_audio(
      audio_array: np.ndarray,
      window_size_s: float = 5.0,
      hop_size_s: float = 5.0,
      sample_rate = 32000,
      ) -> np.ndarray:
    """This function frames audio for inference

    Args:
        audio_array: input audio array to be framed
        window_size_s: Frame length
        hop_size_s: Time between frame starting points
        sample_rate: sampling rate
        nil:
    """
    if window_size_s is None or window_size_s < 0:
        return audio_array[np.newaxis, :]

    frame_length = int(window_size_s * sample_rate)
    hop_length = int(hop_size_s * sample_rate)
    framed_audio = tf.signal.frame(audio_array, frame_length, hop_length, pad_end=True)

    return framed_audio

def ensure_sample_rate(waveform, original_sample_rate,
                       desired_sample_rate=32000):
    """Ensure samplerate of audio is 32kHz and resample if not

    Args:
        waveform: Input audio
        original_sample_rate: Original sampling rate
        desired_sample_rate: Default of 32kHz
    """
    if original_sample_rate != desired_sample_rate:
        waveform = tfio.audio.resample(waveform, original_sample_rate, desired_sample_rate)
    return desired_sample_rate, waveform

def load_model():
    """Load the model and get labels
    """
    model_path = 'https://kaggle.com/models/google/bird-vocalization-classifier/frameworks/tensorFlow2/variations/bird-vocalization-classifier/versions/1'
    model = hub.load(model_path)
    labels_path = hub.resolve(model_path) + "/assets/label.csv"

    return model, labels_path

def process_audio(path, filename, model):
    """Open audio files and process 

    Args:
        path: File location
        filename: Filename
    """
    audio, sr = librosa.load(os.path.join(path, filename))
    _, wav_data = ensure_sample_rate(audio, sr)
    test_frames = frame_audio(wav_data, hop_size_s=5)

    _, all_embeddings = model.infer_tf(test_frames[:1])
    for window in test_frames[1:]:
        _, embeddings = model.infer_tf(window[np.newaxis, :])
        all_embeddings = np.concatenate([all_embeddings, embeddings], axis=0)

    return all_embeddings

def process_audio_frames(frames, model):
    """Generate frame embeddings

    Args:
        frames: input frames
        model: model
    """
    _, all_embeddings = model.infer_tf(frames[:1])

    for window in frames[1:]:
        _, embeddings = model.infer_tf(window[np.newaxis, :])
        all_embeddings = np.concatenate([all_embeddings, embeddings], axis=0)

    return all_embeddings
