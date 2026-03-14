# Data

The CheXpert dataset is not included in this repository due to its size and licensing terms.

## How to download

The easiest way is via Kaggle:

1. Go to: https://www.kaggle.com/datasets/ashery/chexpert
2. Download the dataset (you need a free Kaggle account)
3. Extract the archive and place the contents inside this `data/` folder so the structure looks like this:

        data/
        ├── archive/
        │   ├── train/
        │   ├── valid/
        │   ├── train.csv
        │   └── valid.csv
        └── sample/

## Notes

- The dataset contains 224,316 chest X-rays from 65,240 patients
- Labels cover 5 pathologies: Atelectasis, Cardiomegaly, Consolidation, Edema, Pleural Effusion
- Labels include uncertainty values (-1) — see notebooks/01_exploration.ipynb for how this is handled