# Ndege Zetu (Our Birds)

This repository contains code to conduct experiments on the bioacoustic analysis of recordings drawn from the Mt Kenya Ecosystem. These recordings were obtained using custom made Raspberry Pi based recorders and prototypes of the AudioMoth.

The signal processing pipeline makes use of [source-separation](https://github.com/google-research/sound-separation/tree/master/models/bird_mixit) to separate mixtures of several birds into recordings with single species and then obtains embeddings from these recordings using Google's [Bird Vocalization Classifier](https://www.kaggle.com/models/google/bird-vocalization-classifier) for downstream classification.


# Examples
See the notebook to see the output of the  pipeline on an example recording. 


## Requirements
See the `requirements.txt` file

## Installation procedure

Create a virtual environment

```
python3.10 -m venv bird-env
```

The update `pip`

```
pip install --upgrade pip
```
Clone the repository and install the requirements
```
pip install -r requirements.txt
```

### Source Separation
We make use of the unsupervised source separation models trained using mixture invariant training. To access the models and processing scripts, clone this repo

```
git clone https://github.com/google-research/sound-separation.git
```

Then download models and checkpoints for source separation models trained on environmental recordings of bird calls with mixture invariant training (MixIT) by running. See [this link](https://github.com/google-research/sound-separation/tree/master/models/bird_mixit).

```
gsutil -m cp -r gs://gresearch/sound_separation/bird_mixit_model_checkpoints .

```