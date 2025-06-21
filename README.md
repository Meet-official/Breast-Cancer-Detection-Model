# 🧠 Breast Cancer Detection Model

A machine learning project to classify breast cancer as **Malignant** or **Benign** using a logistic regression model. This project is built using Python and the `sklearn` library and includes a fully functional predictive system.

---

## 🚀 Project Overview

This model uses a medical dataset to predict whether a tumor is malignant or benign based on a variety of features. The classification is performed using Logistic Regression, a simple yet effective algorithm for binary classification tasks.

---

## 📊 Dataset

The dataset used includes features derived from digitized images of breast mass and includes variables like:

- Radius
- Texture
- Perimeter
- Area
- Smoothness
- ...and many more.

### Diagnosis Column:

- `0` → Benign
- `1` → Malignant

---

## 🔍 Features & Techniques

- ✅ Feature Engineering
- 🔄 Data Preprocessing with Stratified Split
- ⚖️ Balanced classes using stratified sampling
- 📈 Model training using `LogisticRegression`
- 🧪 Accuracy evaluation on test data
- 🧠 Built a predictive system for real input

---

## 🛠️ Libraries Used

```python
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
```

---

## 📈 Model Evaluation

- Achieved high accuracy score on the test dataset
   Accuracy: 0.9375

- Predictive system correctly identifies cancer diagnosis from input data

---

## 📬 Output

The model predicts:
- `0` → Benign
- `1` → Malignant

With high accuracy and robustness thanks to stratified data splitting.

---

## 🧪 How to Use

1. Clone the repository:
   ```bash
   git clone https://github.com/Meet-official/BreastCancerClassifier.git
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the Jupyter Notebook:
   ```bash
   jupyter notebook BreastCancer.ipynb
   ```

---

## 📁 Project Structure

```
BreastCancerClassifier/
│
├── data/
│   └── data.csv                  # Dataset
│
├── models/
│   └── BreastCancer.joblib       # Trained ML model
│
├── notebooks/
│   └── BreastCancer.ipynb        # Notebook version
│
├── breastCancer.py               # Main training & prediction script
├── requirements.txt              # Python package dependencies
└── README.md                     # This beautiful file ✨
```

---

## 📌 Conclusion

This project shows how simple logistic regression can be used effectively in medical diagnosis. With the right preprocessing and feature engineering, even linear models can achieve solid results.

---

## 📜 License

This project is licensed under the [MIT License](https://choosealicense.com/licenses/mit/).

---

## ✍️ Author

**Made with ❤ by Meet Patel**

---

## ⭐️ Show Some Love

If you found this useful, consider giving it a ⭐️ on GitHub — it motivates and supports further development!

---
