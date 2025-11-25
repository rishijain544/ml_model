# ml_model

# 🚀 Model Hub Pro — Machine Learning Playground

Model Hub Pro is a **fully interactive ML web application** built with **Streamlit**, designed for data exploration, preprocessing, model training, and visualization — all in one place.

Whether you're a beginner learning ML or someone who wants quick insights on any dataset, this tool gives you a smooth, powerful, and premium experience.

---

## ⭐ Features

### 📂 **Dataset Handling**

* Upload CSV files
* Load Iris demo dataset
* Automatic data type detection
* Missing value handling
* Auto label-encoding for categoricals
* Optional one-touch scaling (Standard / MinMax)

---

### 🧠 **Machine Learning Models**

Supports both **Regression** and **Classification**:

#### **Regression Models**

* Linear Regression
* Decision Tree Regressor

#### **Classification Models**

* Logistic Regression
* Decision Tree Classifier

The app automatically detects task type based on the target column, with manual override available.

---

### 📊 **Advanced Visualizations**

* Confusion Matrix
* Classification Report
* ROC Curve (for binary classification)
* Residual Plot (for regression)
* Feature Importance (Tree-based models)
* Permutation Importance (optional)

---

### 🛠 **Model Utilities**

* Train-test split control
* Random seed configuration
* Feature selection control
* Preview processed dataset
* Download cleaned + transformed CSV
* Sample prediction table for each trained model

---

### 🎨 **Premium UI & Theme System**

* Light / Dark mode toggle
* Modern card layout
* Sleek typography
* Sidebar controls
* Responsive, minimal design

---

## 🧪 Technologies Used

| Category       | Tools                 |
| -------------- | --------------------- |
| Framework      | Streamlit             |
| ML Models      | Scikit-Learn          |
| Data Handling  | Pandas, NumPy         |
| Visuals        | Matplotlib            |
| Preprocessing  | LabelEncoder, Scalers |
| UI Enhancement | Custom CSS            |

---

## 🧩 Project Structure

```
├── app.py
├── requirements.txt
└── README.md
```

---

## 🚀 Live Demo

🔗 **Streamlit App:** https://mlmodel-gyaydbmuxrvkxpzfqcdw4v.streamlit.app/


## 🛠 Installation (Local Setup)

```bash
pip install -r requirements.txt
streamlit run app.py
```

---

## 📄 Requirements

Your `requirements.txt` should contain:

```
streamlit
pandas
numpy
matplotlib
scikit-learn
```

---

## 🤝 Contributing

Pull requests are welcome!
You can also open issues for bugs, feature requests, or improvements.

---

## 📝 License

This project is **open-source** and free to use.

---

## ❤️ Acknowledgements

Thanks to the open-source community & the Streamlit ecosystem for making ML accessible for everyone.

---

