import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import os

from sklearn.cluster import AgglomerativeClustering

st.set_page_config(page_title="FACIES DETECTION", layout="wide")

st.title("FACIES DETECTION")
st.sidebar.header("Upload Dataset")

uploaded_file = st.sidebar.file_uploader(
    "Upload Well Log CSV",
    type=["csv"]
)

if uploaded_file is None:
    st.info("Upload a dataset to begin.")
    st.stop()



# ---------------------------------
# Load Dataset (Fresh Input)
# ---------------------------------

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.success("Dataset uploaded successfully.")
else:
    df = pd.read_csv("C:\\Users\\HP-PC\\Downloads\\log.csv")


# Cleaning (same as training)
df = df.drop_duplicates()
df = df.fillna(df.mean(numeric_only=True))
df = df[(df['DPOR'] > 0)]
df = df[(df['RHOB'] > 0)]
df = df[(df['CNLS'] > 0)]

# Depth Range Slider
min_depth = float(df["Depth"].min())
max_depth = float(df["Depth"].max())

depth_range = st.sidebar.slider(
    "Select Depth Range",
    min_value=min_depth,
    max_value=max_depth,
    value=(min_depth, max_depth)
)
df = df[(df["Depth"] >= depth_range[0]) &
        (df["Depth"] <= depth_range[1])]

features = ['Depth','GR', 'DPOR', 'RHOB', 'RILD', 'SP']
X = df[features]

model_folder = "saved_models"
tab1, tab2 = st.tabs(["Prediction", "Model Comparison"])

# ---------------------------------
# Sidebar Model Selection
# ---------------------------------
with tab1:
    
    
    model_choice = st.sidebar.selectbox(
        "Select Model",
        [
            "SVM",
            "Logistic Regression",
            "Random Forest",
            "KMeans Clustering",
            "Gaussian Mixture Model",
            "Hierarchical Clustering"
        ]
    )

# ==========================
# RUN PREDICTION BUTTON
# ==========================

    run_model = st.sidebar.button("Run Prediction")

    if run_model:

    # ==========================
    # SUPERVISED MODELS
    # ==========================

        if model_choice == "SVM":

            model = joblib.load(os.path.join(model_folder, "svm.pkl"))
            scaler = joblib.load(os.path.join(model_folder, "svm_scaler.pkl"))
            pca = joblib.load(os.path.join(model_folder, "svm_pca.pkl"))
            le = joblib.load(os.path.join(model_folder, "label_encoder.pkl"))

            X_scaled = scaler.transform(X)
            X_pca = pca.transform(X_scaled)
            preds = model.predict(X_pca)

            df["Prediction"] = le.inverse_transform(preds)


        elif model_choice == "Logistic Regression":

            model = joblib.load(os.path.join(model_folder, "lr.pkl"))
            scaler = joblib.load(os.path.join(model_folder, "lr_scaler.pkl"))
            pca = joblib.load(os.path.join(model_folder, "lr_pca.pkl"))
            le = joblib.load(os.path.join(model_folder, "label_encoder.pkl"))

            X_scaled = scaler.transform(X)
            X_pca = pca.transform(X_scaled)
            preds = model.predict(X_pca)

            df["Prediction"] = le.inverse_transform(preds)


        elif model_choice == "Random Forest":

            model = joblib.load(os.path.join(model_folder, "rf.pkl"))
            le = joblib.load(os.path.join(model_folder, "label_encoder.pkl"))

            preds = model.predict(X)

            df["Prediction"] = le.inverse_transform(preds)


    # ==========================
    # UNSUPERVISED MODELS
    # ==========================

        elif model_choice == "KMeans Clustering":

            model = joblib.load(os.path.join(model_folder, "kmeans.pkl"))
            scaler = joblib.load(os.path.join(model_folder, "kmeans_scaler.pkl"))

            X_scaled = scaler.transform(X)
            clusters = model.predict(X_scaled)
        
            cluster_interpretation = {
                
                    0: "Shale-like",
                    1: "Sandstone-like"
                }
    
            df["Prediction"] = [
                f"Cluster {c} ({cluster_interpretation.get(c, 'Unknown')})"
                for c in clusters
                ]

    

        elif model_choice == "Gaussian Mixture Model":

            model = joblib.load(os.path.join(model_folder, "gmm.pkl"))
            scaler = joblib.load(os.path.join(model_folder, "gmm_scaler.pkl"))

            X_scaled = scaler.transform(X)
            clusters = model.predict(X_scaled)
            cluster_interpretation = {
                
                    0: "Shale-like",
                    1: "Sandstone-like"
                }
    
            df["Prediction"] = [
                f"Cluster {c} ({cluster_interpretation.get(c, 'Unknown')})"
                for c in clusters
                ]
       


        elif model_choice == "Hierarchical Clustering":

            scaler = joblib.load(os.path.join(model_folder, "hc_scaler.pkl"))

            X_scaled = scaler.transform(X)

            from sklearn.cluster import AgglomerativeClustering
            hc = AgglomerativeClustering(n_clusters=2, linkage="ward")
            clusters = hc.fit_predict(X_scaled)
            cluster_interpretation = {
                
                    0: "Sandstone-like",
                    1: "Shale-like"
                }
    
            df["Prediction"] = [
                f"Cluster {c} ({cluster_interpretation.get(c, 'Unknown')})"
                for c in clusters
                ]


    # ==========================
    # DISPLAY RESULTS
    # ==========================

        st.subheader("Predicted Facies (Top 100 Rows)")
        st.dataframe(df[features + ["Prediction"]].head(100))


    # ==========================
    # FACIES STATISTICS PANEL
    # ==========================

        st.subheader("Facies Statistics")

        facies_counts = df["Prediction"].value_counts()
        facies_percent = (facies_counts / len(df)) * 100

        stats_df = pd.DataFrame({
            "Count": facies_counts,
            "Percentage (%)": facies_percent.round(2)
        })

        st.dataframe(stats_df)

        total_thickness = df["Depth"].max() - df["Depth"].min()
        st.write(f"Total Interval Thickness: {total_thickness:.2f} m")

        avg_spacing = df["Depth"].diff().mean()
        facies_thickness = facies_counts * avg_spacing

        thickness_df = pd.DataFrame({
            "Estimated Thickness (m)": facies_thickness.round(2)
        })

        st.dataframe(thickness_df)

        st.subheader("Depth vs Predicted Facies")

        fig, ax = plt.subplots(figsize=(4, 8))

# Case 1: Cluster-based labels (unsupervised)
        if df["Prediction"].str.contains("Cluster").any():
            # Extract cluster number
            df["Cluster_Num"] = df["Prediction"].str.extract(r'Cluster (\d+)').astype(int)

            ax.scatter(df["Cluster_Num"], df["Depth"],
               c=df["Cluster_Num"], s=5, cmap="viridis")

            ax.set_xticks(sorted(df["Cluster_Num"].unique()))
            ax.set_xticklabels(sorted(df["Prediction"].unique()))

# Case 2: Supervised labels (Shale/Sandstone)
        else:
            
            labels = ["Shale", "Sandstone"]  # fixed order
            label_map = {label: i for i, label in enumerate(labels)}

            df["Plot_Num"] = df["Prediction"].map(label_map)

            ax.scatter(df["Plot_Num"], df["Depth"],
               c=df["Plot_Num"], s=5, cmap="viridis")

            ax.set_xticks(list(label_map.values()))
            ax.set_xticklabels(list(label_map.keys()))

# Common settings
        ax.invert_yaxis()
        ax.set_xlabel("Facies")
        ax.set_ylabel("Depth")

        st.pyplot(fig)
# ---------------------------------
# Download Results
# ---------------------------------
with tab2:

    st.subheader("Model Performance Comparison")

    results = []

    # =========================
    # Supervised Models
    # =========================

    from sklearn.metrics import accuracy_score
    from sklearn.metrics import silhouette_score

    # IMPORTANT: you must have true labels for supervised comparison
    # Only run if "Facies" column exists in uploaded dataset

    if "Facies" in df.columns:

        y_true = df["Facies"]

        # SVM
        try:
            model = joblib.load(os.path.join(model_folder, "svm.pkl"))
            scaler = joblib.load(os.path.join(model_folder, "svm_scaler.pkl"))
            pca = joblib.load(os.path.join(model_folder, "svm_pca.pkl"))
            le = joblib.load(os.path.join(model_folder, "label_encoder.pkl"))

            X_scaled = scaler.transform(X)
            X_pca = pca.transform(X_scaled)

            preds = le.inverse_transform(model.predict(X_pca))
            acc = accuracy_score(y_true, preds)

            results.append(["SVM", "Accuracy", round(acc, 4)])

        except:
            pass


        # Logistic Regression
        try:
            model = joblib.load(os.path.join(model_folder, "lr.pkl"))
            scaler = joblib.load(os.path.join(model_folder, "lr_scaler.pkl"))
            pca = joblib.load(os.path.join(model_folder, "lr_pca.pkl"))
            le = joblib.load(os.path.join(model_folder, "label_encoder.pkl"))

            X_scaled = scaler.transform(X)
            X_pca = pca.transform(X_scaled)

            preds = le.inverse_transform(model.predict(X_pca))
            acc = accuracy_score(y_true, preds)

            results.append(["Logistic Regression", "Accuracy", round(acc, 4)])

        except:
            pass


        # Random Forest
        try:
            model = joblib.load(os.path.join(model_folder, "rf.pkl"))
            le = joblib.load(os.path.join(model_folder, "label_encoder.pkl"))

            preds = le.inverse_transform(model.predict(X))
            acc = accuracy_score(y_true, preds)

            results.append(["Random Forest", "Accuracy", round(acc, 4)])

        except:
            pass

    else:
        st.info("No true labels found. Accuracy comparison skipped.")


    # =========================
    # Unsupervised Models
    # =========================

    try:
        # KMeans
        model = joblib.load(os.path.join(model_folder, "kmeans.pkl"))
        scaler = joblib.load(os.path.join(model_folder, "kmeans_scaler.pkl"))

        X_scaled = scaler.transform(X)
        clusters = model.predict(X_scaled)

        score = silhouette_score(X_scaled, clusters)

        results.append(["KMeans", "Silhouette Score", round(score, 4)])

    except:
        pass


    try:
        # GMM
        model = joblib.load(os.path.join(model_folder, "gmm.pkl"))
        scaler = joblib.load(os.path.join(model_folder, "gmm_scaler.pkl"))

        X_scaled = scaler.transform(X)
        clusters = model.predict(X_scaled)

        score = silhouette_score(X_scaled, clusters)

        results.append(["Gaussian Mixture", "Silhouette Score", round(score, 4)])

    except:
        pass


    try:
        # Hierarchical
        scaler = joblib.load(os.path.join(model_folder, "hc_scaler.pkl"))
        X_scaled = scaler.transform(X)

        from sklearn.cluster import AgglomerativeClustering
        hc = AgglomerativeClustering(n_clusters=2, linkage="ward")
        clusters = hc.fit_predict(X_scaled)

        score = silhouette_score(X_scaled, clusters)

        results.append(["Hierarchical", "Silhouette Score", round(score, 4)])

    except:
        pass


    # =========================
    # Display Comparison Table
    # =========================

    if results:
        comparison_df = pd.DataFrame(results, columns=["Model", "Metric", "Score"])
        st.dataframe(comparison_df)
    else:
        st.warning("No models evaluated.")
        
    csv = df.to_csv(index=False).encode("utf-8")

    st.download_button(
        label="Download Prediction Results",
        data=csv,
        file_name="facies_predictions.csv",
        mime="text/csv"
    )
