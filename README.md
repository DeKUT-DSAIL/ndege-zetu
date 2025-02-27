# Ndege Zetu (Our Birds)

We present a dataset of over 3000 minute long audio recordings obtained from two sites within the Mt Kenya ecosystem as well as observations from point counts conducted at the recorder locations and observations from the Kenya Bird Map. The audio recordings and point counts were obtained at the Dedan Kimathi University of Technology Wildlife Conservancy (DeKUWC) and the Mt Kenya National Park (MKNP) between 2016 and 2018 with the recordings obtained using custom made Raspberry Pi based recorders and early versions of the AudioMoth. 

This repository also contains code to analyse these data and reproduce results in an accompanying paper ``Comparing point counts, passive acoustic monitoring, citizen science and machine learning for bird species monitoring in the Mt Kenya ecosystem''  submitted to Philosophical Transactions of the Royal Society B.

We demonstrate the use of embeddings obtained from Google's [Bird Vocalization Classifier (Perch)](https://www.kaggle.com/models/google/bird-vocalization-classifier) to train classifiers for the species observed.

The data are available on Dryad as the dataset

[```wa Maina, Ciira; Njoroge, Peter (2025). Ndege Zetu: A dataset to compare bird species monitoring approaches in the Mt Kenya ecosystem [Dataset]. Dryad. https://doi.org/10.5061/dryad.d51c5b0c7```](https://datadryad.org/dataset/doi:10.5061/dryad.d51c5b0c7)



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

## Data Download
Ndege Zetu is available on Dryad at [https://doi.org/10.5061/dryad.d51c5b0c7](https://doi.org/10.5061/dryad.d51c5b0c7). Download the data and place the recordings in the audio directory.

Part of the data were available in an older dataset  [https://datadryad.org/stash/dataset/doi:10.5061/dryad.69g60](https://datadryad.org/stash/dataset/doi:10.5061/dryad.69g60).


## Acknowledgements
We thank the African Bird Club, ARM and Google for financial support that enabled us to complete the study.