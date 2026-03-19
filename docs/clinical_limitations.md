# Clinical Limitations and Deployment Considerations

## What this model does
Brief, honest description of the task.

## What this model does NOT do
- It is not a diagnostic tool. It is a decision-support aid.
- It does not replace radiologist review under any circumstance.
- It was not trained or validated on data from European hospitals or non-US patient populations.

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

Analysis on the validation set revealed performance disparities across demographic groups:

**By sex:**
- Atelectasis: AUC-ROC 0.81 (Male) vs 0.65 (Female) — largest gap observed
- Consolidation: AUC-ROC 0.96 (Male) vs 0.89 (Female)
- Edema and Pleural Effusion: minimal gap, performance comparable across sexes

**By age group:**
- Atelectasis: AUC-ROC 0.87 (<40) vs 0.65 (60+) — significant degradation in older patients
- Edema: AUC-ROC 1.00 (<40) — interpret with caution, only 34 patients in this group
- Other pathologies relatively stable across age groups

These gaps should be investigated before any clinical deployment.

## Regulatory context
Any clinical deployment in the EU would require CE marking under MDR 2017/745. 
In the UK, registration with the MHRA would be required. 
This prototype has not undergone clinical validation.

## Recommended next steps before any clinical use
1. External validation on an independent dataset
2. Prospective study with radiologist oversight
3. Formal bias audit across patient demographics
4. Regulatory pathway assessment