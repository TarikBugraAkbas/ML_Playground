
# Energy Consumption Prediction using Linear Regression

## Overview
This project implements **Linear Regression** to predict energy consumption based on various features. Two versions of the model are implemented:
- **From Scratch Implementation** by creating Gradient Descent and Cost Functions from scratch
- **Scikit-Learn Implementation** using Scikit-Learn model Linear Regression


### Dataset Attribution
The dataset is sourced from [Energy Consumption Dataset - Linear Regression](https://www.kaggle.com/datasets/govindaramsriram/energy-consumption-dataset-linear-regression) by Govindaram Sriram and is licensed under the [CC BY 4.0 License](https://creativecommons.org/licenses/by/4.0/). 

## Dataset
The dataset contains features such as:
- Square Footage
- Number of Occupants
- Appliances Used
- Average Temperature
- Building Type
- Day of Week

**Target Variable:** `Energy Consumption`

## Visualization

The **Residuals Plot** below highlights the difference between the actual data and the predicted values for both implementations:

- **Green dots**: Represent predictions from the **Scikit-Learn implementation**.
- **Blue dots**: Represent predictions from the **From-Scratch implementation**.

This comparison demonstrates the accuracy and consistency of both approaches in predicting energy consumption.

![Residuals Comparison](Figure_1.png)



