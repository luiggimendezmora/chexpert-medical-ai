import torch
from torch.utils.data import Dataset
from PIL import Image
import pandas as pd
import numpy as np


class CheXpertDataset(Dataset):
    def __init__(self, csv_path, transform=None, uncertainty_policy='zeros'):
        """
        uncertainty_policy:
          'zeros'  — treat uncertain labels as negative (conservative)
          'ones'   — treat uncertain labels as positive (sensitive)
        """
        self.df = pd.read_csv(csv_path)
        self.transform = transform
        self.pathologies = [
            'Atelectasis', 'Cardiomegaly',
            'Consolidation', 'Edema', 'Pleural Effusion'
        ]

        # Handle uncertainty
        labels = self.df[self.pathologies].fillna(0)
        if uncertainty_policy == 'zeros':
            labels = labels.replace(-1, 0)
        elif uncertainty_policy == 'ones':
            labels = labels.replace(-1, 1)

        self.labels = labels.values.astype(np.float32)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img_path = self.df.iloc[idx]['Path']
        image = Image.open(img_path).convert('RGB')

        if self.transform:
            image = self.transform(image)
        label = torch.tensor(self.labels[idx])

        return image, label