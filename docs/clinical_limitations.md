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