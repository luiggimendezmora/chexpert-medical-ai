# CheXpert Medical Imaging Portfolio Project
## Complete Guide — AI for Healthcare Portfolio

---

## Project Goal

Build a **chest X-ray pathology detection system** that goes beyond a basic classifier — demonstrating clinical awareness, interpretability, and fairness analysis. This is what separates a junior Kaggle notebook from a senior data scientist's portfolio piece.

**Target audience when recruiters read this:** healthtech teams in Ireland, Denmark, and the UK.

---

## Repository Structure

```
chexpert-medical-ai/
│
├── README.md                  ← The most important file
├── requirements.txt
├── .gitignore
│
├── data/
│   ├── README.md              ← Instructions to download CheXpert (don't commit data)
│   └── sample/                ← 5-10 anonymized sample images for demo
│
├── notebooks/
│   ├── 01_exploration.ipynb   ← EDA + dataset understanding
│   ├── 02_training.ipynb      ← Model training
│   ├── 03_interpretability.ipynb  ← Grad-CAM visualizations
│   └── 04_fairness.ipynb      ← Bias analysis by age/sex
│
├── src/
│   ├── __init__.py
│   ├── dataset.py             ← PyTorch Dataset class
│   ├── model.py               ← Model definition + fine-tuning logic
│   ├── train.py               ← Training loop
│   ├── evaluate.py            ← Metrics + fairness evaluation
│   └── gradcam.py             ← Grad-CAM implementation
│
├── outputs/
│   ├── models/                ← Saved checkpoints (.gitignore large files)
│   ├── gradcam/               ← Grad-CAM visualization images
│   └── metrics/               ← CSV/JSON results
│
└── docs/
    └── clinical_limitations.md  ← Written as if presenting to a clinician
```

---

## Step-by-Step Development Plan

### PHASE 1 — Setup (Day 1)

**1. Create the GitHub repository**
- Public repo, name: `chexpert-medical-ai`
- Add description: "Chest pathology detection with interpretability and fairness analysis — PyTorch, EfficientNet, Grad-CAM"
- Add topics/tags: `medical-imaging`, `deep-learning`, `computer-vision`, `healthcare-ai`, `pytorch`

**2. Install dependencies**

```bash
pip install torch torchvision torchaudio
pip install grad-cam
pip install scikit-learn pandas numpy matplotlib seaborn
pip install Pillow tqdm
pip install jupyter notebook
```

**3. Verify MPS (Metal) acceleration on your Mac M-series**

```python
import torch
print(torch.backends.mps.is_available())  # Should print True
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
```

**4. Download CheXpert Small**
- Register at: https://stanfordmlgroup.github.io/competitions/chexpert/
- Download the small version (~11GB)
- Never commit the data to GitHub — only commit the data/README.md with download instructions

---

### PHASE 2 — Exploration (Days 2–3)

**Notebook: `01_exploration.ipynb`**

Goals:
- Understand the label structure (5 pathologies: Atelectasis, Cardiomegaly, Consolidation, Edema, Pleural Effusion)
- Understand the uncertainty labels (positive=1, negative=0, uncertain=-1, not mentioned=blank)
- Visualize class distribution — it's heavily imbalanced, which is clinically realistic
- Visualize patient demographics (age, sex) — needed later for fairness analysis
- Display sample X-rays with their labels

Key insight to document: **Label uncertainty is not a bug, it's clinically realistic.** Even radiologists disagree. Your notebook should explain this.

```python
import pandas as pd
import matplotlib.pyplot as plt

train_df = pd.read_csv('data/CheXpert-v1.0-small/train.csv')

# Check label distribution
pathologies = ['Atelectasis', 'Cardiomegaly', 'Consolidation', 'Edema', 'Pleural Effusion']
print(train_df[pathologies].value_counts())

# Check demographics
print(train_df['Sex'].value_counts())
print(train_df['Age'].describe())
```

---

### PHASE 3 — Model Training (Days 4–6)

**File: `src/dataset.py`**

```python
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
```

**File: `src/model.py`**

```python
import torch
import torch.nn as nn
import torchvision.models as models

def build_model(num_classes=5, freeze_backbone=False):
    """
    EfficientNet-B0 fine-tuned for multi-label chest pathology detection.
    Using transfer learning from ImageNet — standard practice in medical imaging
    where labeled data is scarce.
    """
    model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
    
    if freeze_backbone:
        for param in model.parameters():
            param.requires_grad = False
    
    # Replace classifier head for 5-class multi-label output
    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.3),
        nn.Linear(in_features, num_classes)
    )
    
    return model
```

**File: `src/train.py`**

```python
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from dataset import CheXpertDataset
from model import build_model
from tqdm import tqdm

def train(epochs=10, batch_size=32, lr=1e-4):
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Training on: {device}")
    
    transform_train = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    transform_val = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    train_dataset = CheXpertDataset('data/CheXpert-v1.0-small/train.csv', transform_train)
    val_dataset = CheXpertDataset('data/CheXpert-v1.0-small/valid.csv', transform_val)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    
    model = build_model().to(device)
    
    # BCEWithLogitsLoss for multi-label classification
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    
    best_val_loss = float('inf')
    
    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0
        for images, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        
        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
        
        avg_train = train_loss / len(train_loader)
        avg_val = val_loss / len(val_loader)
        print(f"Epoch {epoch+1}: Train Loss={avg_train:.4f} | Val Loss={avg_val:.4f}")
        
        scheduler.step()
        
        if avg_val < best_val_loss:
            best_val_loss = avg_val
            torch.save(model.state_dict(), 'outputs/models/best_model.pth')
            print("  → Best model saved")

if __name__ == "__main__":
    train()
```

---

### PHASE 4 — Interpretability with Grad-CAM (Days 7–8)

**Notebook: `03_interpretability.ipynb`**

This is the section that will impress clinical AI teams most. Grad-CAM shows *where* in the image the model is looking — essential for clinical trust.

```python
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import torch
import torchvision.transforms as transforms
from src.model import build_model

def visualize_gradcam(image_path, class_idx, class_name, model, device):
    # Load and preprocess image
    img = Image.open(image_path).convert('RGB').resize((224, 224))
    img_array = np.array(img) / 255.0
    
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    input_tensor = transform(img).unsqueeze(0).to(device)
    
    # Target layer — last conv block of EfficientNet
    target_layers = [model.features[-1]]
    targets = [ClassifierOutputTarget(class_idx)]
    
    with GradCAM(model=model, target_layers=target_layers) as cam:
        grayscale_cam = cam(input_tensor=input_tensor, targets=targets)[0]
    
    visualization = show_cam_on_image(img_array.astype(np.float32), grayscale_cam, use_rgb=True)
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].imshow(img_array)
    axes[0].set_title("Original X-Ray", fontsize=13)
    axes[0].axis('off')
    axes[1].imshow(visualization)
    axes[1].set_title(f"Grad-CAM — {class_name}", fontsize=13)
    axes[1].axis('off')
    plt.tight_layout()
    plt.savefig(f'outputs/gradcam/{class_name.lower().replace(" ", "_")}.png', dpi=150)
    plt.show()
```

---

### PHASE 5 — Fairness Analysis (Days 9–10)

**Notebook: `04_fairness.ipynb`**

This is what truly differentiates your project. Most portfolios skip this entirely.

```python
import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns

# Load predictions with demographic info
def fairness_analysis(predictions_df):
    """
    predictions_df columns: 
      Sex, Age_group, true_label, pred_prob (per pathology)
    """
    pathologies = ['Atelectasis', 'Cardiomegaly', 'Consolidation', 
                   'Edema', 'Pleural Effusion']
    
    results = []
    
    for pathology in pathologies:
        # By sex
        for sex in ['Male', 'Female']:
            subset = predictions_df[predictions_df['Sex'] == sex]
            if subset[f'true_{pathology}'].sum() > 0:
                auc = roc_auc_score(
                    subset[f'true_{pathology}'], 
                    subset[f'pred_{pathology}']
                )
                results.append({
                    'Pathology': pathology,
                    'Group': sex,
                    'Subgroup_type': 'Sex',
                    'AUC-ROC': auc
                })
        
        # By age group
        for age_group in predictions_df['Age_group'].unique():
            subset = predictions_df[predictions_df['Age_group'] == age_group]
            if subset[f'true_{pathology}'].sum() > 0:
                auc = roc_auc_score(
                    subset[f'true_{pathology}'], 
                    subset[f'pred_{pathology}']
                )
                results.append({
                    'Pathology': pathology,
                    'Group': age_group,
                    'Subgroup_type': 'Age',
                    'AUC-ROC': auc
                })
    
    results_df = pd.DataFrame(results)
    
    # Visualize
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    sex_data = results_df[results_df['Subgroup_type'] == 'Sex']
    sex_pivot = sex_data.pivot(index='Pathology', columns='Group', values='AUC-ROC')
    sex_pivot.plot(kind='bar', ax=axes[0], colormap='coolwarm')
    axes[0].set_title('AUC-ROC by Sex per Pathology', fontsize=13)
    axes[0].set_ylim(0.5, 1.0)
    axes[0].tick_params(axis='x', rotation=45)
    
    age_data = results_df[results_df['Subgroup_type'] == 'Age']
    age_pivot = age_data.pivot(index='Pathology', columns='Group', values='AUC-ROC')
    age_pivot.plot(kind='bar', ax=axes[1], colormap='viridis')
    axes[1].set_title('AUC-ROC by Age Group per Pathology', fontsize=13)
    axes[1].set_ylim(0.5, 1.0)
    axes[1].tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.savefig('outputs/metrics/fairness_analysis.png', dpi=150)
    plt.show()
    
    return results_df
```

---

### PHASE 6 — Clinical Limitations Document (Day 11)

**File: `docs/clinical_limitations.md`**

Este documento es lo que demuestra madurez clínica. Escríbelo en inglés, en tono sobrio, como si se lo presentaras a un médico antes de un piloto clínico.

Estructura sugerida:

```markdown
# Clinical Limitations and Deployment Considerations

## What this model does
Brief, honest description of the task.

## What this model does NOT do
- It is not a diagnostic tool. It is a decision-support aid.
- It does not replace radiologist review under any circumstance.
- It was not trained or validated on data from [your target hospital/region].

## Known limitations

### Label quality
CheXpert labels were extracted via NLP from radiology reports, 
not manually annotated by radiologists. Label noise is inherent.

### Distribution shift
The model was trained on data from Stanford Medical Center. 
Performance may degrade on X-rays from different equipment, 
patient populations, or imaging protocols.

### Uncertainty handling
Uncertain labels were mapped to negative during training (U-zeros policy). 
This introduces a conservative bias — the model may underdetect 
pathologies in ambiguous cases.

### Fairness gaps
[Insert your findings from the fairness notebook here]

## Regulatory context
Any clinical deployment in the EU would require CE marking under MDR 2017/745. 
In the UK, registration with the MHRA would be required. 
This prototype has not undergone clinical validation.

## Recommended next steps before any clinical use
1. External validation on an independent dataset
2. Prospective study with radiologist oversight
3. Formal bias audit across patient demographics
4. Regulatory pathway assessment
```

---

### PHASE 7 — README (Day 12)

El README es lo primero que lee un reclutador. Tiene que ser impecable.

**Estructura:**

1. **One-line description** — qué hace el proyecto y por qué importa clínicamente
2. **Demo image** — un Grad-CAM bonito, visual, que enganche en 3 segundos
3. **Motivation** — 2-3 frases sobre por qué hiciste esto (conecta con tu LinkedIn)
4. **What makes this different** — interpretability + fairness + clinical limitations
5. **Technical stack** — PyTorch, EfficientNet, Grad-CAM, MPS acceleration
6. **Results** — tabla con AUC-ROC por patología y por subgrupo demográfico
7. **How to run** — instrucciones claras para reproducir
8. **Clinical limitations** — enlace al doc completo
9. **Dataset** — instrucciones para descargar CheXpert

---

## Timeline Estimado

| Semana | Fases |
|--------|-------|
| Semana 1 | Setup + Exploración + Dataset |
| Semana 2 | Entrenamiento + Debugging |
| Semana 3 | Grad-CAM + Fairness analysis |
| Semana 4 | README + Clinical doc + Limpieza del repo |

4 semanas en total trabajando a ritmo razonable.

---

## Métricas a reportar

Para cada una de las 5 patologías:
- **AUC-ROC** — métrica estándar en medical imaging
- **AUC-ROC por sexo** (Male vs Female)
- **AUC-ROC por grupo de edad** (< 40, 40-60, 60+)

No reportes accuracy — es una métrica pobre para datos desbalanceados y un reclutador en healthtech lo sabrá.

---

## Lo que este proyecto demuestra a un reclutador

| Lo que ven | Lo que entienden |
|------------|-----------------|
| EfficientNet fine-tuning | Transfer learning en dominio médico |
| Grad-CAM | Conciencia de interpretabilidad clínica |
| Fairness analysis | Comprensión de sesgos y ética en IA médica |
| Clinical limitations doc | Madurez — sabe que un modelo no es un producto médico |
| README bien escrito | Comunicación técnica a audiencias mixtas |

---

*Guía preparada como parte de un plan de reinserción profesional orientado a AI for Healthcare — Ireland · Denmark · UK*
```
