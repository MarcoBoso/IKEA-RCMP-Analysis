# IKEA RCMP Analysis

**Author**: Marco Boso  
**Date**: August 9, 2025  
**Project Type**: Customer Analytics / KPI Modeling

## Overview

This project simulates and analyzes key performance metrics for IKEA’s **Remote Customer Meeting Point (RCMP)** operations within an omnichannel retail context.  
The dataset is synthetic, inspired by public sources including *Development of Key Performance Indicators for the Product Launch Process at IKEA Industry* by Dennis Widmark & Rasmus Axenram, and other materials on IKEA’s customer service processes.  
It does not contain real customer data but is designed to reflect realistic patterns, distributions, and operational scenarios.

The analysis focuses on two main KPIs:
- **Customer Satisfaction (CSAT)**
- **First Contact Resolution (FCR)**

## Dataset

600 simulated customer interaction records including:
- **Interaction context**: date, country, channel type, specific channel, contact reason, product category, language
- **Operational metrics**: queue length, wait time, handle time, backlog index, SLA breach
- **Customer profile**: order value, promo use, returning customer status
- **Contextual features**: weekday, hour, pre-interaction sentiment
- **Outcomes**: CSAT (1–5 scale), FCR (binary)

## Libraries

- `tidyverse` – data manipulation & visualization  
- `caret` – modeling & train/test split  
- `MASS` – stepwise regression  
- `glmnet` – regularized regression  
- `pROC` – ROC curve analysis  
- `broom` – model output tidying  
- `ggplot2`, `forcats`, `data.table` – additional processing & plotting

## Methods

- Baseline models for CSAT and FCR  
- Multiple linear regression & stepwise AIC selection (CSAT)  
- Logistic regression (FCR)  
- Regularized regression with glmnet (Elastic Net for both CSAT and FCR)  
- Performance evaluation (RMSE, R², Accuracy, AUC)  
- Feature importance analysis (coefficients, non-zero weights)  
- Visual diagnostics (Observed vs Predicted, ROC curves)

## Key Results

- **CSAT**:  
  - Best model: glmnet (Elastic Net)  
  - RMSE ≈ 0.65, R² ≈ 0.24  
  - Key drivers: pre-interaction sentiment (+), wait time (–), channel type, wait × channel interactions

- **FCR**:  
  - Both logistic regression and glmnet yielded AUC ≈ 0.50 (no better than random guessing)  
  - Only meaningful predictors: returning customer (+), queue length (–)

## Outputs

- **ikea_models.rds** – saved R models (linear, stepwise, glmnet, logistic) for reproducibility  
- **ikea_csat_predictions.csv** – predicted vs observed CSAT values from the best model

## Visual Outputs

Includes:
- CSAT distribution histogram  
- CSAT vs Wait Time by channel type  
- Observed vs Predicted CSAT scatter plot (glmnet)  
- ROC curve for FCR
- Dashboard that explores the evolution of CSAT and FCR in realtions with the variables 'date' and 'channel_type'

## How to Run

```bash
git clone git@github.com:MarcoBoso/IKEA-RCMP-Analysis.git
cd IKEA-RCMP-Analysis
