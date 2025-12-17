# Fraud Risk Analytics for Health Insurance Claims

## Course
DSCI 510: Principles of Programming for Data Science

## Team
Shivam Kumar  
USC Email: shivamku@usc.edu  
USC ID: 9153656592  
GitHub: shivam8764  

## Project Summary
This project develops an end to end fraud risk analytics workflow for health insurance claims. The pipeline covers data ingestion, exploratory data analysis, preprocessing, feature engineering, and supervised learning to assign a fraud propensity score to each claim. Model performance is evaluated using recall focused metrics and decile based ranking analysis to reflect real world investigation workflows, where analysts prioritize the highest risk claims first.

## Problem Statement
Health insurance fraud leads to direct financial leakage through suspicious billing behavior such as phantom services and ghost enrollee claims. The objective is to build a claim level scoring model that identifies high risk claims early, enabling insurers to prioritize manual review and reduce fraudulent payouts while maintaining fair premiums for legitimate policyholders.

## Data
The dataset consists of claim level records containing:
- Patient demographics (for example age, gender)
- Claim event fields (encounter and discharge dates)
- Financial fields (billed amount)
- Clinical text fields (diagnosis information)
- Fraud type labels used to define a binary fraud target

Raw data is not included in this repository to avoid redistribution and to respect dataset licensing and privacy constraints. The notebook expects the dataset to be available locally in the runtime environment.

## Target Definition
Fraud is modeled as a binary classification task using the fraud type column:
- Fraud (1): phantom billing, ghost enrollee
- Non fraud (0): no fraud, wrong diagnosis

This mapping prioritizes financially malicious scenarios while treating coding or diagnostic issues as non fraud for modeling consistency.

## Approach
### Exploratory Analysis and Preprocessing
- Distribution analysis of claim amounts and key attributes
- Missing value handling and text normalization for categorical fields
- Class imbalance assessment and fraud prevalence estimation

### Feature Engineering
- Date parsing and temporal features such as length of stay and admission month
- High cardinality handling for diagnosis text via grouping and normalization
- Additional derived signals that improve separability between fraud and non fraud

### Modeling
Models were trained in increasing complexity:
- Logistic Regression as a baseline for interpretability and calibrated scoring
- Random Forest to capture non linear relationships and feature interactions
- XGBoost for high performance on structured tabular data

### Evaluation
Primary metrics:
- Recall, to minimize missed fraud cases
- Precision, to measure investigation efficiency
- F1 score, as a balanced summary metric

Operational metrics:
- Decile based lift and cumulative recall tables using predicted probabilities
- Threshold tuning to select an operating point targeting recall around 0.80 to 0.90 while controlling false positives

## Repository Structure
```text
.
├── notebooks/
│   └── fraud_risk_analytics.ipynb
├── outputs/
│   ├── figures/
│   └── tables/
├── final_report.pdf
├── requirements.txt
└── README.md
