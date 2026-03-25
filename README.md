# Deep-Learning
## Advanced machine Learning


# Diabetes Prediction Project

## Overview
This project aims to build a **binary classification model** to predict diabetes risk using health indicator data from the CDC's Behavioral Risk Factor Surveillance System (BRFSS) 2015 survey. The model identifies individuals at risk for diabetes, enabling earlier intervention and better public health outcomes.

## Context
Diabetes affects over **34 million Americans**, with **88 million** more living with prediabetes and most unaware of their condition. The disease leads to severe complications (cardiovascular disease, kidney failure, vision loss) and represents a **$327 billion annual financial burden**. Early detection is critical for effective treatment and prevention.

## Dataset
- **Source:** BRFSS 2015 survey (CDC)
- **Records:** 253,680 individuals
- **Features:** 22 health indicators

### Key Features:
| Variable | Description |
|:---------|:------------|
| `Diabetes_012` | Target: 0 = no diabetes, 1 = prediabetes, 2 = diabetes |
| `BMI` | Body Mass Index |
| `HighBP` | High blood pressure |
| `HighChol` | High cholesterol |
| `GenHlth` | General health (1-5 scale) |
| `PhysActivity` | Physical activity in past 30 days |
| `Age` | Age category (1-13) |
| `Income` | Income level (1-8) |
| `Education` | Education level (1-6) |
| `Smoker` | History of smoking |

## Objective
Develop a **binary classification model** to predict diabetes probability:
- **Target:** `Diabetes_binary` (0 = no diabetes/prediabetes, 1 = diabetes)
- **Goal:** Maximize **ROC AUC** on unlabeled test data

## Methodology
- **MLOps Approach:** Data versioning, process traceability, automated pipelines
- **Agile (Scrum):** Three sprints with iterative deliverables
- **Feature Importance Analysis:** SHAP values for model interpretability

## Technologies
- Python, Pandas, Scikit-learn
- Random Forest / Neural Networks
- SHAP for explainability
- MLflow / DVC for versioning
