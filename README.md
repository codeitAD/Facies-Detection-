# Facies Detection using Machine Learning

This project implements a machine learning–based system to classify geological facies (Shale and Sandstone) using well log data. It integrates data preprocessing, multiple ML models, and an interactive Streamlit dashboard for visualization and analysis.

---

## Project Overview

Geological facies classification is an important task in reservoir characterization. This project automates the classification process using machine learning techniques applied to well log data.

The system allows users to:
- Upload well log datasets
- Select machine learning models
- Generate facies predictions
- Visualize results interactively

---

## Tech Stack

- Python
- Streamlit (Web Application)
- Scikit-learn (Machine Learning)
- Pandas & NumPy (Data Processing)
- Matplotlib (Visualization)
- Orange & Tableau (External Visualization)

---

## Models Implemented

Supervised Learning:
- Support Vector Machine (SVM)
- Logistic Regression
- Random Forest

Unsupervised Learning:
- K-Means Clustering
- Gaussian Mixture Model (GMM)
- Hierarchical Clustering

---

## Features Used

- GR (Gamma Ray)
- RHOB (Bulk Density)
- DPOR (Porosity)
- RILD (Resistivity)
- SP (Spontaneous Potential)
- Depth

---

## Project Structure

facies-detection-ml/
│
├── app.py                  # Streamlit application
├── requirements.txt       # Dependencies
├── README.md
│
├── models/                # Pre-trained models
│   ├── kmeans.pkl
│   ├── gmm.pkl
│   ├── hierarchical.pkl
│   ├── scaler.pkl
│
├── data/                  # Dataset files
│   
│   ├── log.csv
│
├── notebooks/             # Training & experimentation
│   ├── data_preprocessing.ipynb
│   ├── model_training.ipynb

---

## How to Run

1. Install dependencies:

pip install -r requirements.txt

2. Run the application:

streamlit run app.py

3. Open in browser:

http://localhost:8501

---

## Workflow

1. Load dataset (CSV)
2. Preprocess data
3. Select model
4. Generate predictions
5. Visualize results

---

## Dataset

Source: Kansas Geological Survey  
Type: Well log dataset  

---

## Output

- Facies classification (Shale / Sandstone)
- Statistical analysis (count, percentage)
- Model comparison
- Visualizations

---

## Model Training

The training process is available in the notebooks/ folder.  
It includes preprocessing, feature selection, and model training.

Trained models are saved as .pkl files and used in the application.

---

## Note

The .pkl files are pre-trained machine learning models.  
They are loaded automatically by the application and do not need to be opened manually.

---

## Future Scope

- Integration of deep learning models
- Multi-well dataset support
- Advanced visualization dashboards
- Explainable AI techniques

---

## Author

Ayush Dhiman
B.Tech Computer Science Engineering  

---

## License

This project is for academic use only.
