# Pulmonary-Function-Decline-Prediction-Using-LightGBM-with-Patient-Level-Feature-Engineering
# Pulmonary Function Decline Prediction with LightGBM

## Overview
This project predicts **Forced Vital Capacity (FVC)** decline over time in patients with pulmonary fibrosis using **LightGBM regression**.  
The solution applies **patient-level baseline extraction**, **polynomial time modeling**, and **feature interactions**, combined with **5-fold cross-validation** and model ensembling.

The goal is to accurately forecast lung function progression for each patient over future weeks.

---

## Problem Description
Given:
- Patient demographic data
- Clinical baseline measurements
- Time (weeks since baseline)

Predict:
- Future **FVC values**
- With associated confidence scores

This is a **longitudinal regression problem** with repeated measurements per patient.

---

## Dataset
