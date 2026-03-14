import torch
import numpy as np
from sklearn.metrics import roc_auc_score

PATHOLOGIES = ['Atelectasis', 'Cardiomegaly', 'Consolidation', 'Edema', 'Pleural Effusion']

def evaluate(model, dataloader, device):
    model.eval()
    all_labels = []
    all_preds = []

    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            outputs = torch.sigmoid(model(images))
            all_preds.append(outputs.cpu().numpy())
            all_labels.append(labels.numpy())

    all_preds = np.concatenate(all_preds, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)

    results = {}
    for i, pathology in enumerate(PATHOLOGIES):
        if all_labels[:, i].sum() > 0:
            auc = roc_auc_score(all_labels[:, i], all_preds[:, i])
            results[pathology] = round(auc, 4)
        else:
            results[pathology] = None

    return results