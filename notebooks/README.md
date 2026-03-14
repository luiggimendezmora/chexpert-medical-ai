# Notebooks

Run the notebooks in order. Each one builds on the previous.

| Notebook | Description |
|----------|-------------|
| 01_exploration.ipynb | Dataset overview — label distribution, demographics, sample images |
| 02_training.ipynb | EfficientNet-B0 fine-tuning — transfer learning from ImageNet |
| 03_interpretability.ipynb | Grad-CAM visualizations — where the model looks in each X-ray |
| 04_fairness.ipynb | Fairness analysis — AUC-ROC by sex and age group per pathology |

## Requirements

All dependencies are in `requirements.txt`. Activate the virtual environment before launching Jupyter:

    source .venv/bin/activate
    jupyter notebook