# 🧪 Breast Cancer Diagnosis with SVM and Streamlit

A complete machine learning pipeline for breast cancer diagnosis using the UCI dataset and a web-based prediction app built with Streamlit. This project includes data cleaning, exploratory data analysis (EDA), SVM modeling, and a user-friendly diagnostic tool.

---

## 📌 Project Objective

> Build a machine learning model that classifies tumors as **malignant** or **benign**, and deploy it as an interactive web form to help demonstrate AI-assisted diagnostics.

---

## 📂 Folder Structure

├── data/ # Cleaned breast cancer dataset (CSV)                                                       
├── notebooks/ # EDA, training, SHAP, and cleaning notebooks                                                       
├── outputs/ # Visualizations, exported PDF plots                                                          
├── streamlit_app.py # Final deployed diagnostic tool                                                     
├── requirements.txt # Python dependencies                                                           
└── README.md # Project documentation                                                     


---

## 🧠 Features Used

We used the top 7 most influential features from SHAP analysis:

- `radius_mean`
- `texture_mean`
- `perimeter_mean`
- `area_mean`
- `smoothness_mean`
- `concave_points_mean`
- `compactness_mean`

---

## 📊 Model Used

- **Support Vector Classifier (SVC)**
- `class_weight='balanced'` to handle class imbalance
- Feature scaling via `StandardScaler`
- Evaluated using **recall score** (focus on malignant detection)

---

## 🖥️ Streamlit Web App

### Features:
- 📥 Input tumor features manually or use sample presets
- 🧠 Predicts if the tumor is **Benign** or **Malignant**
- 📈 Displays prediction confidence and probability breakdown
- 🎯 Clean and simple UI

### 📦 Run Locally

bash
git clone https://github.com/your-username/breast-cancer-diagnosis.git                             
cd breast-cancer-diagnosis                                              
pip install -r requirements.txt                                             
streamlit run streamlit_app.py                                            



## 🔬 Dataset

- Source: UCI Machine Learning Repository

- Breast Cancer Wisconsin (Diagnostic) Data Set



## ✅ Status

- ✔ Data cleaning (UCI .data + sklearn)

- ✔ EDA + feature comparison plots (PDF export)

- ✔ SVM model training + recall optimization

- ✔ Streamlit app with sample inputs

- ✔ requirements.txt added


## 🚀 Optional Enhancements

- 🔲 Deploy to Streamlit Cloud

- 🔲 Add SHAP explanations back in future

- 🔲 Export clinical report PDF



## 🤝 License
This project is licensed under the MIT License.


## 👨‍💻 Author
Nitin Kandpal
Feel free to connect on https://www.linkedin.com/in/nitinkandpal/            


