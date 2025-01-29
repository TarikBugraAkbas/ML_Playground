# Energy Consumption Prediction using Linear Regression

## Overview
This project implements **Linear Regression** to predict energy consumption based on various features. Two versions of the model are implemented:
- **From Scratch Implementation** by creating Gradient Descent and Cost Functions from scratch
- **Scikit-Learn Implementation** using Scikit-Learn model Linear Regression

## Dataset
The dataset is sourced from **Kaggle**, containing features such as:
- Square Footage
- Number of Occupants
- Appliances Used
- Average Temperature
- Building Type
- Day of Week

**Target Variable:** `Energy Consumption`

## Implementation Details
### From Scratch (Manual Gradient Descent)
- Implements **Gradient Descent** for optimization.
- Uses **Mean Squared Error (MSE)** as the cost function.
- Normalizes input features to improve convergence.
- Implements automatic **convergence detection**.

### Scikit-Learn Version
- Utilizes `LinearRegression()` for training and prediction.
- Requires no manual gradient updates.
- Faster and optimized compared to the scratch implementation.


## Results & Comparison
### Scikit-Learn vs. From-Scratch Performance


Both implementations yield similar **high accuracy**.

## Visualization
Below is a **comparison of actual vs predicted energy consumption**:

![Prediction Results](images/result.png)

## Future Improvements
- Experiment with **polynomial regression** for better accuracy.
- Implement **regularization (L1/L2)** to avoid overfitting.
- Test on **different datasets** for generalization.

## Contributing
If you'd like to contribute, feel free to fork this repository, make changes, and submit a **pull request**.

## License
This project is **open-source** and available under the MIT License.

