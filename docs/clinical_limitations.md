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

### Fairness gaps

Analysis on the validation set (234 patients) revealed performance disparities 
across demographic groups:

**By sex:**

| Pathology | Male (n=128) | Female (n=106) | Gap |
|-----------|-------------|----------------|-----|
| Atelectasis | 0.8066 | 0.6537 | 0.153 |
| Cardiomegaly | 0.8441 | 0.8194 | 0.025 |
| Consolidation | 0.9632 | 0.8855 | 0.078 |
| Edema | 0.9307 | 0.9345 | 0.004 |
| Pleural Effusion | 0.9080 | 0.9452 | -0.037 |

**By age group:**

| Pathology | <40 (n=34) | 40-60 (n=75) | 60+ (n=125) |
|-----------|-----------|--------------|-------------|
| Atelectasis | 0.8667 | 0.8104 | 0.6494 |
| Cardiomegaly | 0.8894 | 0.8459 | 0.8107 |
| Consolidation | 0.9172 | 0.9508 | 0.9190 |
| Edema | 1.0000* | 0.9015 | 0.9231 |
| Pleural Effusion | 0.9103 | 0.9442 | 0.9017 |

*Interpret with caution — only 34 patients in this age group.

**Key findings:**
- Atelectasis shows the largest fairness gap: 15 points lower AUC-ROC in women 
  vs men, and 22 points lower in patients over 60 vs under 40.
- Edema and Pleural Effusion are the most equitable across both sex and age.
- These gaps should be investigated and mitigated before any clinical deployment.

## Regulatory context
Any clinical deployment in the EU would require CE marking under MDR 2017/745. 
In the UK, registration with the MHRA would be required. 
This prototype has not undergone clinical validation.

## Recommended next steps before any clinical use
1. External validation on an independent dataset
2. Prospective study with radiologist oversight
3. Formal bias audit across patient demographics
4. Regulatory pathway assessment