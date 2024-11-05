import streamlit as st
import pandas as pd 
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import nbformat
from pathlib import Path
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix,  recall_score, f1_score, precision_score
from sklearn import neighbors
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import OrdinalEncoder
from sklearn.inspection import permutation_importance
from sklearn.ensemble import RandomForestClassifier  
from sklearn.datasets import make_classification
from scipy.stats import chi2_contingency
import optuna



# Define a function for loading data
@st.cache_data
def load_data():
    # Modify this path based on your actual data file location
    df_satisfaction = pd.read_csv("C:\\Users\\nourn\\Desktop\\ML\\train.csv")


    return df_satisfaction

# Sidebar for page navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Exploratory Data Analysis (EDA)", "Model and Evaluation"])

# Page 1: EDA
if page == "Exploratory Data Analysis (EDA)":
    st.title("Exploratory Data Analysis (EDA)")

    # Load data
    data = load_data()
    
    # Display dataset overview
    st.write("### Dataset Overview")
    st.write(data.head())

    # Display dataset statistics
    st.write("### Dataset Summary")
    st.write(data.describe())
    st.write(data.describe(include='object'))

    # Check for missing values
    st.write("### Missing Values")
    st.write(data.isnull().sum())

    data= data.dropna()
    
    # Feature distribution
    numerical_cols = data.select_dtypes(include='number').columns.tolist()

    # Selectbox for choosing a numerical column
    selected_column = st.selectbox("Select a numerical column to view its distribution", numerical_cols)

    # Plot the distribution for the selected column
    st.write(f"### Distribution of {selected_column}")
    fig, ax = plt.subplots(figsize=(8, 4))
    sns.histplot(data[selected_column], bins=30, kde=True, ax=ax)
    ax.set_title(f'Distribution of {selected_column}')
    ax.set_xlabel(selected_column)
    ax.set_ylabel('Frequency')
    st.pyplot(fig)
    
    
    
    
    label_encoder = LabelEncoder()
    data['satisfaction'] = label_encoder.fit_transform(data['satisfaction'])
    #from sklearn.preprocessing import LabelEncoder
    numerical_cols = data.select_dtypes(include=['number']).columns
    df_numerical = data[numerical_cols] 
    

    corr_matrix = df_numerical.corr()

    # Display the heatmap in Streamlit
    st.write("### Heatmap de corrélation - Variables numériques")
    fig, ax = plt.subplots(figsize=(12, 10))
    sns.set(font_scale=1.1)

    # Create the heatmap
    sns.heatmap(
        corr_matrix, 
        annot=True,               # Display correlation values
        fmt=".2f",                # Format annotation to 2 decimal places
        cmap='viridis',           # Color map
        linewidths=0.5,           # Line width between squares
        square=True,              # Make cells square-shaped
        cbar_kws={"shrink": .75}, # Adjust color bar size
        annot_kws={"size": 10}    # Annotation font size
    )

    # Aesthetics
    plt.title('Heatmap de corrélation - Variables numériques', fontsize=18, pad=20)
    plt.xticks(rotation=45, ha='right', fontsize=12) 
    plt.yticks(fontsize=12)
    plt.tight_layout()  # Adjust layout to fit labels

    # Display the plot in Streamlit
    st.pyplot(fig)
    
    numerical_cols = data.select_dtypes(include='number').columns.tolist()

    # Calculate correlations with the 'satisfaction' column
    correlation_results = {}
    for col in numerical_cols:
        correlation = data[col].corr(data['satisfaction'])
        correlation_results[col] = correlation



    
    # Convert results to DataFrame for visualization
    correlation_df = pd.DataFrame(list(correlation_results.items()), columns=['Variable', 'Corrélation de Pearson'])

    # Display the barplot in Streamlit
    st.write("### Corrélation de Pearson pour les variables numériques par rapport à la satisfaction")
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(data=correlation_df, x='Variable', y='Corrélation de Pearson', palette='coolwarm', ax=ax)

    # Add plot details
    ax.set_title("Corrélation de Pearson pour les variables numériques par rapport à la satisfaction")
    ax.set_xlabel("Variables")
    ax.set_ylabel("Corrélation de Pearson")
    ax.axhline(0, color='black', linestyle='--')  # Reference line at 0
    ax.axhline(0.1, color='red', linestyle='--', label='Corrélation faible')
    ax.axhline(0.3, color='orange', linestyle='--', label='Corrélation modérée')
    ax.axhline(0.5, color='green', linestyle='--', label='Corrélation forte')
    ax.axhline(-0.1, color='red', linestyle='--')
    ax.axhline(-0.3, color='orange', linestyle='--')  # Moderate negative correlation
    ax.axhline(-0.5, color='green', linestyle='--')   # Strong negative correlation
    ax.legend()
    plt.xticks(rotation=45, ha='right')  # Rotate x labels
    plt.tight_layout()

    # Display the plot in Streamlit
    st.pyplot(fig)
    
    
    
    categorical_cols = data.select_dtypes(exclude='number').columns.tolist()

    # Function to calculate Cramér's V
    def cramers_v(contingency_table):
        chi2, p, dof, expected = chi2_contingency(contingency_table)
        n = contingency_table.sum().sum()
        return np.sqrt(chi2 / (n * (min(contingency_table.shape) - 1)))

    # Calculate Cramér's V for each categorical variable in relation to 'satisfaction'
    results = {}
    for col in categorical_cols:
        contingency_table = pd.crosstab(data[col], data['satisfaction'])
        v = cramers_v(contingency_table)
        results[col] = v

    # Convert results to DataFrame for visualization
    results_df = pd.DataFrame(list(results.items()), columns=['Variable', "Cramér's V"])

    # Display the Cramér's V barplot in Streamlit
    st.write("### Cramér's V pour les variables catégoriques par rapport à la satisfaction")
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(data=results_df, x='Variable', y="Cramér's V", palette='coolwarm', ax=ax)

    # Add plot details
    ax.set_title("Cramér's V pour les variables catégoriques par rapport à la satisfaction")
    ax.set_xlabel("Variables")
    ax.set_ylabel("Cramér's V")
    ax.axhline(0.1, color='red', linestyle='--', label='Association faible')
    ax.axhline(0.3, color='orange', linestyle='--', label='Association modérée')
    ax.axhline(0.5, color='green', linestyle='--', label='Association forte')
    plt.xticks(rotation=45)
    ax.legend()
    plt.tight_layout()

    # Display the plot in Streamlit
    st.pyplot(fig)
        
    
    
    
    
    
    
    

# Page 2: Model and Evaluation
elif page == "Model and Evaluation":
    st.title("Model Training and Evaluation")

    # Load data
    data = load_data()

    # Data preparation (modify based on your notebook)
    # Assuming the last column is the target for demonstration; adjust as needed
    X = data.iloc[:, :-1]  # Features
    y = data.iloc[:, -1]   # Target

    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    # Model training (using RandomForest as an example, modify as per your model)
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # Display classification report
    st.write("### Classification Report")
    report = classification_report(y_test, y_pred, output_dict=True)
    report_df = pd.DataFrame(report).transpose()
    st.write(report_df)

    # Display confusion matrix
    st.write("### Confusion Matrix")
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
    st.pyplot(fig)

    # Feature importance (for models that support it)
    if hasattr(model, "feature_importances_"):
        st.write("### Feature Importances")
        feature_importances = pd.DataFrame({
            'feature': X.columns,
            'importance': model.feature_importances_
        }).sort_values(by="importance", ascending=False)

        st.write(feature_importances)

# Run the app with `streamlit run app.py`
