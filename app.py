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
import plotly
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, classification_report
from xgboost import XGBClassifier
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import uniform




# Define a function for loading data

def load_data():
    # Modify this path based on your actual data file location
    
    url = "https://raw.githubusercontent.com/Nourben7/machine_learning/main/df_satisfaction.csv"
    data = pd.read_csv(url)


    return data
if "page" not in st.session_state:
    st.session_state.page = "Exploratory Data Analysis (EDA)"
    

# Sidebar for page navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Exploratory Data Analysis (EDA)", "Baseline : KNN Classifier","Iterations","Random forest","Gradient boosting"])

st.session_state.page = page
# Page 1: EDA
if st.session_state.page == "Exploratory Data Analysis (EDA)":
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
elif st.session_state.page == "Baseline : KNN Classifier":
    
    
   
        data = load_data()
        data= data.dropna()
        label_encoder = LabelEncoder()
        data['satisfaction'] = label_encoder.fit_transform(data['satisfaction'])
    
        # Data preparation (modify based on your notebook)
        # Assuming the last column is the target for demonstration; adjust as needed
        X = data.drop(columns=['satisfaction']) #'Unnamed: 0','id' 
        X_numeric = X.select_dtypes(include = ['number'])  # Caractéristiques
        y = data['satisfaction']   # Cible

        st.title("Baseline : KNN Classifier")
        X, y = X_numeric, y
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        # Pipeline setup with KNeighborsClassifier
        pipeline = Pipeline(steps=[('model', KNeighborsClassifier())])

        # Model training
        pipeline.fit(X_train, y_train)

        # Predictions and evaluation
        y_pred = pipeline.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        st.write(f"Accuracy: {accuracy:.4f}")
        st.write("Classification Report:")
        st.text(classification_report(y_test, y_pred))
        
    
    
    
        accuracy = accuracy_score(y_test, y_pred) * 100  # Pourcentage d'accuracy
        st.write(f'Accuracy (Exactitude): {accuracy:.2f}%')


        # Générer la matrice de confusion
        cm = confusion_matrix(y_test, y_pred)

        # Visualiser la matrice de confusion avec Seaborn
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False, 
                    xticklabels=pipeline.classes_, yticklabels=pipeline.classes_)
        plt.xlabel('Prédiction')
        plt.ylabel('Vrai Label')
        plt.title('Matrice de Confusion du KNN')
        st.pyplot(plt)
        
        
        
        
        
        
elif st.session_state.page == "Iterations":
    
        data = load_data()
        data= data.dropna()
        label_encoder = LabelEncoder()
        data['satisfaction'] = label_encoder.fit_transform(data['satisfaction'])
        st.title("Normalisation de la donnée")

        X = data.drop(columns=['satisfaction']) #'Unnamed: 0','id' 
        X_numeric = X.select_dtypes(include = ['number'])  # Caractéristiques
        y = data['satisfaction'] 
        X, y = X_numeric, y

        # Normalisation de la donnée
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_numeric)
        
        knn = KNeighborsClassifier()

        X_train, X_test, y_train, y_test = train_test_split(X_scaled, y,
                                                            test_size=0.2, random_state=42)
        # Entraînement du modèle avec le pipeline
        knn.fit(X_train, y_train)

        # Prédiction sur l'ensemble de test
        y_pred = knn.predict(X_test)

        # Accuracy score (test set)
        accuracy_test = accuracy_score(y_test, y_pred)
        st.write(f'Accuracy on test set: {accuracy_test:.4f}')

        # Affichage du rapport de classification (précision, rappel, F1-score)
        st.write(classification_report(y_test, y_pred))

        # Compute the confusion matrix on the test set
        cm = confusion_matrix(y_test, y_pred)

        # Plot the confusion matrix using Seaborn
        plt.figure(figsize=(6, 4))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title('Confusion Matrix (Test Set)')
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        st.pyplot(plt)
    
    
        y_pred_train = knn.predict(X_train)

        # Rapport de classification et accuracy sur l'ensemble d'entraînement
        st.write("Rapport de classification (ensemble d'entraînement) :\n")
        st.write(classification_report(y_train, y_pred_train))
        accuracy_train = accuracy_score(y_train, y_pred_train)
        st.write(f'Accuracy sur l\'ensemble d\'entraînement : {accuracy_train:.4f}')

        # Matrice de confusion (train)
        cm_train = confusion_matrix(y_train, y_pred_train)
        plt.figure(figsize=(6, 4))
        sns.heatmap(cm_train, annot=True, fmt='d', cmap='Blues')
        plt.title('Matrice de confusion (Ensemble d\'entraînement)')
        plt.xlabel('Label prédit')
        plt.ylabel('Label réel')
        st.pyplot(plt)
 
  
        st.title("Changement 2: Validation des données")
            # Définir le modèle KNN avec k=10 
        k_default = 10
        knn = KNeighborsClassifier(n_neighbors=k_default)

        # Effectuer la validation croisée
        scores = cross_val_score(knn, X_scaled, y, cv=5, scoring='accuracy')  # 5 plis

        # Afficher les scores pour chaque pli
        st.write(f'Scores de validation croisée pour k={k_default}: {scores}')
        st.write(f'Moyenne de l\'accuracy: {np.mean(scores):.4f} ± {np.std(scores):.4f}')

        
        st.title("Changement 3 : Itérations de k ")
        X, y = X_numeric, y

        # Séparer le dataset en utilisant train_test_split
        X_train, X_test, y_train, y_test = train_test_split(X_scaled,
                    y, test_size=0.2, shuffle=True, random_state=42)

        # Create the KNN classifier with k neighbors
        clf = neighbors.KNeighborsClassifier(n_neighbors=10)
        clf.fit(X_train, y_train)

        # Generate predictions on the test set
        y_pred = clf.predict(X_test)

        # Compute the accuracy score on the test set
        accuracy_testk10 = accuracy_score(y_test, y_pred)
        st.write(f'Accuracy on test set: {accuracy_testk10:.4f}')

        # Generate the classification report on the test set
        st.write("Classification Report on test set:")
        st.write(classification_report(y_test, y_pred))

        # Compute the confusion matrix on the test set
        cm = confusion_matrix(y_test, y_pred)

        # Plot the confusion matrix using Seaborn
        plt.figure(figsize=(6, 4))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title('Confusion Matrix (Test Set)')
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        st.pyplot(plt)
        
        pipeline = Pipeline(steps=[('model', KNeighborsClassifier())])

        # Model training
        pipeline.fit(X_train, y_train)

        # --- Evaluation sur l'ensemble d'ENTRAÎNEMENT ---
        # Prédiction sur l'ensemble d'entraînement
        y_pred_train = pipeline.predict(X_train)

        # Rapport de classification et accuracy sur l'ensemble d'entraînement
        st.write("Rapport de classification (ensemble d'entraînement) :\n")
        st.write(classification_report(y_train, y_pred_train))
        accuracy_traink10 = accuracy_score(y_train, y_pred_train)
        st.write(f'Accuracy sur l\'ensemble d\'entraînement : {accuracy_traink10:.4f}')

        # Matrice de confusion (train)
        cm_train = confusion_matrix(y_train, y_pred_train)
        plt.figure(figsize=(6, 4))
        sns.heatmap(cm_train, annot=True, fmt='d', cmap='Blues')
        plt.title('Matrice de confusion (Ensemble d\'entraînement)')
        plt.xlabel('Label prédit')
        plt.ylabel('Label réel')
        st.pyplot(plt)

        st.title("Amélioration des paramètres - Itération des valeurs de k (n_neighbors)")
        
        X, y = X_numeric, y

        # Normalisation des données
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # Diviser le jeu de données en ensembles d'entraînement et de test
        X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, 
                                                            test_size=0.2, 
                                                            shuffle=True, 
                                                            random_state=42)

        # Liste pour stocker les scores de précision pour différentes valeurs de k
        k_values = list(range(1, 21))  # Tester les valeurs de k de 1 à 20
        accuracy_scores = []

        # Itérer sur les valeurs de k
        for k in k_values:
            # Créer le classificateur KNN avec k voisins
            clf = neighbors.KNeighborsClassifier(n_neighbors=k)
            clf.fit(X_train, y_train)  # Entraîner le modèle

            # Générer des prédictions sur l'ensemble de test
            y_pred = clf.predict(X_test)

            # Calculer le score de précision sur l'ensemble de test
            accuracy = accuracy_score(y_test, y_pred)
            accuracy_scores.append(accuracy)

            st.write(f'Accuracy for k={k}: {accuracy:.4f}')

        # Tracer les scores de précision pour différentes valeurs de k
        plt.figure(figsize=(10, 6))
        plt.plot(k_values, accuracy_scores, marker='o')
        plt.title('Accuracy vs. Number of Neighbors (k)')
        plt.xlabel('Number of Neighbors (k)')
        plt.ylabel('Accuracy')
        plt.xticks(k_values)
        plt.grid()
        st.pyplot(plt)

        # Afficher la meilleure valeur de k basée sur la précision
        best_k = k_values[np.argmax(accuracy_scores)]
        st.write(f'Best k value: {best_k} with accuracy: {max(accuracy_scores):.4f}')

        # Évaluer le modèle avec le meilleur k sur l'ensemble de test
        best_clf = neighbors.KNeighborsClassifier(n_neighbors=best_k)
        best_clf.fit(X_train, y_train)
        y_test_pred = best_clf.predict(X_test)

        # Afficher le rapport de classification sur l'ensemble de test
        st.write("Classification Report for Test Set:")
        st.write(classification_report(y_test, y_test_pred))

        # Afficher la matrice de confusion sur l'ensemble de test
        cm_train = confusion_matrix(y_test, y_test_pred)
        plt.figure(figsize=(6, 4))
        sns.heatmap(cm_train, annot=True, fmt='d', cmap='Blues')
        plt.title('Matrice de Confusion (Ensemble de test)')
        plt.xlabel('Label Prédit')
        plt.ylabel('Label Réel')
        st.pyplot(plt)

        # Évaluer le modèle avec le meilleur k sur l'ensemble d'entraînement
        best_clf = neighbors.KNeighborsClassifier(n_neighbors=best_k)
        best_clf.fit(X_train, y_train)
        y_train_pred = best_clf.predict(X_train)

        # Afficher le rapport de classification sur l'ensemble d'entraînement
        st.write("Classification Report for Training Set:")
        st.write(classification_report(y_train, y_train_pred))

        # Afficher la matrice de confusion sur l'ensemble d'entraînement
        cm_train = confusion_matrix(y_train, y_train_pred)
        plt.figure(figsize=(6, 4))
        sns.heatmap(cm_train, annot=True, fmt='d', cmap='Blues')
        plt.title('Matrice de Confusion (Ensemble d\'Entraînement)')
        plt.xlabel('Label Prédit')
        plt.ylabel('Label Réel')
        st.pyplot(plt)
        
        
        # Prédictions de probabilités pour l'ensemble de test
        y_proba = best_clf.predict_proba(X_test)[:, 1]  # Probabilités pour la classe positive (1)

        # Calculer les valeurs de la courbe ROC
        fpr, tpr, thresholds = roc_curve(y_test, y_proba)

        # Calculer l'AUC
        roc_auc = auc(fpr, tpr)

        # Tracer la courbe ROC
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='blue', label=f'ROC curve (AUC = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], color='red', linestyle='--')  # ligne de hasard
        plt.title('Receiver Operating Characteristic (ROC) Curve')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.legend(loc='lower right')
        plt.grid()
        st.pyplot(plt)

        
        st.title("Amélioration des paramètres - Itération des paramètres 'n_neighbors' & 'weight'")
    
        # Définir le classificateur KNN
        knn = neighbors.KNeighborsClassifier()

        # Définir les paramètres à tester
        param_grid = {
            'n_neighbors': range(1, 21),  # Tester les valeurs de k de 1 à 20
            'weights': ['uniform', 'distance'],  # Tester les poids uniformes et basés sur la distance
        }

        # Initialiser GridSearchCV
        grid_search = GridSearchCV(knn, param_grid, cv=5, scoring='accuracy')

        # Ajuster le modèle sur l'ensemble d'entraînement
        grid_search.fit(X_train, y_train)

        # Afficher les meilleurs paramètres
        st.write("Meilleurs paramètres : ", grid_search.best_params_)
        st.write("Meilleure précision : ", grid_search.best_score_)

        # Évaluer le modèle optimisé sur l'ensemble de test
        best_knn = grid_search.best_estimator_
        y_test_pred = best_knn.predict(X_test)

        # Afficher le rapport de classification sur l'ensemble de test
        st.write("Classification Report (Test Set):")
        st.write(classification_report(y_test, y_test_pred))

        # Afficher la matrice de confusion sur l'ensemble de test
        cm_test = confusion_matrix(y_test, y_test_pred)
        plt.figure(figsize=(6, 4))
        sns.heatmap(cm_test, annot=True, fmt='d', cmap='Blues')
        plt.title('Matrice de Confusion (Ensemble de Test)')
        plt.xlabel('Label Prédit')
        plt.ylabel('Label Réel')
        st.pyplot(plt)

    
    
        st.title("Changement 4 : Optuna")
        X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42, shuffle=True)

        # Définir la fonction objective à optimiser
        def objective(trial):
            try:
            # Hyperparamètres à optimiser
                k = trial.suggest_int('n_neighbors', 1, 20)  # Nombre de voisins entre 1 et 20
                weights = trial.suggest_categorical('weights', ['uniform', 'distance'])  # Poids uniformes ou basés sur la distance
                metric = trial.suggest_categorical('metric', ['euclidean', 'manhattan', 'chebyshev'])
                leaf_size = trial.suggest_int('leaf_size', 10, 50)  # Taille des feuilles
                p = trial.suggest_int('p', 1, 2)  # Paramètre pour la distance de Minkowski (1=Manhattan, 2=Euclidean)
                algorithm = trial.suggest_categorical('algorithm', ['auto', 'ball_tree', 'kd_tree', 'brute'])  # Algorithme utilisé
            
                # Création du classificateur KNN avec les paramètres suggérés
                clf = KNeighborsClassifier(n_neighbors=k, weights=weights, metric=metric, 
                                            leaf_size=leaf_size, p=p, algorithm=algorithm)
            
                # Évaluation du modèle avec validation croisée
                scores = cross_val_score(clf, X_train, y_train, cv=5, scoring='accuracy') 
                return scores.mean()  # Retourner la moyenne des scores de validation croisée
            except FloatingPointError:
                return 0 
        # Création d'un objet d'étude
        study = optuna.create_study(direction='maximize')  # Nous voulons maximiser l'accuracy
        study.optimize(objective, n_trials=10)  # Effectuer 10 essais

        # Afficher les meilleurs hyperparamètres trouvés
        st.write("Meilleurs hyperparamètres : ", study.best_params)
        st.write("Meilleure accuracy : ", study.best_value)

        # Évaluer le meilleur modèle sur l'ensemble de test
        best_knn = KNeighborsClassifier(**study.best_params)
        best_knn.fit(X_train, y_train)
        y_test_pred = best_knn.predict(X_test)

        # Afficher le rapport de classification sur l'ensemble de test
        st.write("Classification Report (Test Set):")
        st.write(classification_report(y_test, y_test_pred))

        # Afficher la matrice de confusion sur l'ensemble de test
        cm_test = confusion_matrix(y_test, y_test_pred)
        plt.figure(figsize=(6, 4))
        sns.heatmap(cm_test, annot=True, fmt='d', cmap='coolwarm')
        plt.title('Matrice de Confusion (Ensemble de Test)')
        plt.xlabel('Label Prédit')
        plt.ylabel('Label Réel')
        st.pyplot(plt)
        
        
        
        
        # Visualisation de l'importance des hyperparamètres
        def plot_param_importance(study):
            optuna.visualization.plot_param_importances(study).show()

        # Visualisation de la distribution des hyperparamètres
        def plot_param_distributions(study):
            optuna.visualization.plot_parallel_coordinate(study).show()

        # Tracer la matrice de confusion pour le meilleur modèle
        def plot_confusion_matrix(best_knn, X_test, y_test):
            y_pred = best_knn.predict(X_test)
            cm = confusion_matrix(y_test, y_pred)

            plt.figure(figsize=(6, 4))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
            plt.title('Matrice de Confusion (Ensemble de Test)')
            plt.xlabel('Label Prédit')
            plt.ylabel('Label Réel')
            st.pyplot(plt)

        ## Prédire avec le meilleur modèle et tracer la matrice de confusion
        #best_knn = study.best_trial.user_attrs['model'] 
        #plot_confusion_matrix(best_knn, X_test, y_test)

        # Afficher les visualisations
        # Plot and display parameter importance
        fig1 = plot_param_importance(study)
        st.pyplot(fig1)

        # Plot and display parameter distributions
        fig2 = plot_param_distributions(study)
        st.pyplot(fig2)
    
        st.title("changemnt 5: Sélection de features + Optuna")
        df_clean_satisfaction2=data
        label_encoder = LabelEncoder()

        # Encoder "Type of Travel" avec LabelEncoder
        label_encoder = LabelEncoder()
        df_clean_satisfaction2['Type of Travel'] = label_encoder.fit_transform(df_clean_satisfaction2['Type of Travel'])

        # Encoder "Class" avec OrdinalEncoder
        ordinal_encoder = OrdinalEncoder(categories=[['Eco', 'Eco Plus','Business']])  # Spécifiez l'ordre
        df_clean_satisfaction2['Class'] = ordinal_encoder.fit_transform(df_clean_satisfaction2[['Class']])

        # Afficher le DataFrame encodé
        st.write(df_clean_satisfaction2.head(5))
        
        type_of_travel_classes = label_encoder.classes_

        st.write("Mapping for 'Type of Travel':")
        for index, label in enumerate(type_of_travel_classes):
            st.write(f"Label: '{label}' correspond à l'index: {index}")

        # Accéder aux étiquettes originales et à leurs indices pour "Class"
        class_categories = ordinal_encoder.categories_[0]  # Récupérer les catégories définies pour 'Class'

        st.write("\nMapping for 'Class':")
        for index, label in enumerate(class_categories):
            st.write(f"Label: '{label}' correspond à l'index: {index}")

        X = df_clean_satisfaction2.drop(columns=['satisfaction', 'Unnamed: 0','Age','id','Departure Delay in Minutes','Arrival Delay in Minutes','Departure/Arrival time convenient'  ])
        X_numeric2 = X.select_dtypes(include = ['number'])  # Caractéristiques
    
    
        # Préparer les caractéristiques d'entrée (X) et la variable cible (y)
        X, y = X_numeric2, y

        # Normalisation des données
        scaler = StandardScaler()
        X_scaled2 = scaler.fit_transform(X_numeric2)

        # Diviser le jeu de données une fois avant l'optimisation
        X_train, X_test, y_train, y_test = train_test_split(X_scaled2, y, test_size=0.2, random_state=42, shuffle=True)

        # Définir la fonction objective à optimiser
        def objective(trial):
            try:
                # Hyperparamètres à optimiser
                k = trial.suggest_int('n_neighbors', 5, 20)  # Nombre de voisins entre 5 et 20
                weights = trial.suggest_categorical('weights', ['uniform', 'distance'])  # Poids uniformes ou basés sur la distance
                metric = trial.suggest_categorical('metric', ['euclidean', 'manhattan', 'chebyshev'])
                leaf_size = trial.suggest_int('leaf_size', 10, 20)  # Taille des feuilles
                p = trial.suggest_int('p', 1, 2)  # Paramètre pour la distance de Minkowski (1=Manhattan, 2=Euclidean)
                algorithm = trial.suggest_categorical('algorithm', ['auto', 'ball_tree', 'kd_tree', 'brute'])  # Algorithme utilisé

                # Création du classificateur KNN avec les paramètres suggérés
                clf = KNeighborsClassifier(n_neighbors=k, weights=weights, metric=metric, 
                                        leaf_size=leaf_size, p=p, algorithm=algorithm)

                # Évaluation du modèle avec validation croisée
                scores = cross_val_score(clf, X_train, y_train, cv=5, scoring='accuracy') 
                return scores.mean()  # Retourner la moyenne des scores de validation croisée
            except FloatingPointError:
                return 0
        # Création d'un objet d'étude
        study = optuna.create_study(direction='maximize')  # Nous voulons maximiser l'accuracy
        study.optimize(objective, n_trials=30)  # Effectuer 30 essais

        # Afficher les meilleurs hyperparamètres trouvés
        st.write("Meilleurs hyperparamètres : ", study.best_params)
        st.write("Meilleure accuracy : ", study.best_value)

        # Évaluer le meilleur modèle sur l'ensemble de test
        best_knn = KNeighborsClassifier(**study.best_params)
        best_knn.fit(X_train, y_train)
        y_test_pred = best_knn.predict(X_test)

        # Afficher le rapport de classification sur l'ensemble de test
        st.write("Classification Report (Test Set):")
        st.write(classification_report(y_test, y_test_pred))

        # Afficher la matrice de confusion sur l'ensemble de test
        cm_test = confusion_matrix(y_test, y_test_pred)
        plt.figure(figsize=(6, 4))
        sns.heatmap(cm_test, annot=True, fmt='d', cmap='coolwarm')
        plt.title('Matrice de Confusion (Ensemble de Test)')
        plt.xlabel('Label Prédit')
        plt.ylabel('Label Réel')
        st.pyplot(plt)
        
        # Visualisation de l'importance des hyperparamètres
        def plot_param_importance(study):
            optuna.visualization.plot_param_importances(study).show()

        # Visualisation de la distribution des hyperparamètres
        def plot_param_distributions(study):
            optuna.visualization.plot_parallel_coordinate(study).show()

        # Tracer la matrice de confusion pour le meilleur modèle
        def plot_confusion_matrix(best_knn, X_test, y_test):
            y_pred = best_knn.predict(X_test)
            cm = confusion_matrix(y_test, y_pred)

            plt.figure(figsize=(6, 4))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
            plt.title('Matrice de Confusion (Ensemble de Test)')
            plt.xlabel('Label Prédit')
            plt.ylabel('Label Réel')
            st.pyplot(plt)

        
        fig1 = plot_param_importance(study)
        st.pyplot(fig1)

        # Plot and display parameter distributions
        fig2 = plot_param_distributions(study)
        st.pyplot(fig2)

elif st.session_state.page == "Random forest":
    
        data = load_data()
        data= data.dropna()
        label_encoder = LabelEncoder()
        data['satisfaction'] = label_encoder.fit_transform(data['satisfaction'])
        st.title("Random forest")   

        df_clean_satisfaction= data
        X = df_clean_satisfaction.drop(columns=['satisfaction']) #'Unnamed: 0','id' 
        X_numeric = X.select_dtypes(include = ['number'])  # Caractéristiques
        y = df_clean_satisfaction['satisfaction']  

        # Diviser les données en ensembl d'entraînement et de test
        X_train, X_test, y_train, y_test = train_test_split(X_numeric, y, test_size=0.2, shuffle=True)

        # Créer le classificateur Random Forest avec 100 arbres
        rf_clf = RandomForestClassifier(n_estimators=100,random_state=42)
        rf_clf.fit(X_train, y_train)

        ### Évaluation sur l'ensemble de test

        # Prédire sur l'ensemble de test
        y_pred_test_rf = rf_clf.predict(X_test)

        # Évaluer la précision sur l'ensemble de test
        accuracy_test_rf = accuracy_score(y_test, y_pred_test_rf)
        st.write(f'Random Forest Accuracy on test set: {accuracy_test_rf:.4f}')

        # Afficher le rapport de classification pour l'ensemble de test
        st.write("Random Forest Classification Report on test set:")
        st.write(classification_report(y_test, y_pred_test_rf))

        # Afficher la matrice de confusion pour l'ensemble de test
        cm_test_rf = confusion_matrix(y_test, y_pred_test_rf)
        plt.figure(figsize=(6, 4))
        sns.heatmap(cm_test_rf, annot=True, fmt='d', cmap='Blues')
        plt.title('Random Forest Confusion Matrix (Test Set)')
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        st.pyplot(plt)
        
        
        rf_clf = RandomForestClassifier(n_estimators=100, random_state=42)
        rf_clf.fit(X_train, y_train)

        scores_rf = cross_val_score(rf_clf, X_train, y_train, cv=5, scoring='accuracy')
        st.write(f"Validation croisée, scores: {scores_rf}")
        
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_numeric)

        # Diviser les données en ensemble d'entraînement et de test
        X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, shuffle=True)

        # Créer le classificateur Random Forest avec 100 arbres
        rf_clf = RandomForestClassifier(n_estimators=100, random_state=42)
        rf_clf.fit(X_train, y_train)

        ### Évaluation sur l'ensemble de test

        # Prédire sur l'ensemble de test
        y_pred_test_rf = rf_clf.predict(X_test)

        # Évaluer la précision sur l'ensemble de test
        accuracy_test_rf = accuracy_score(y_test, y_pred_test_rf)
        st.write(f'Random Forest Accuracy on test set: {accuracy_test_rf:.4f}')

        # Afficher le rapport de classification pour l'ensemble de test
        st.write("Random Forest Classification Report on test set:")
        st.write(classification_report(y_test, y_pred_test_rf))

        # Afficher la matrice de confusion pour l'ensemble de test
        cm_test_rf = confusion_matrix(y_test, y_pred_test_rf)
        plt.figure(figsize=(6, 4))
        sns.heatmap(cm_test_rf, annot=True, fmt='d', cmap='Blues')
        plt.title('Random Forest Confusion Matrix (Test Set)')
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        st.pyplot(plt)
        
        st.title("Cross Validation")
        # Créer le classificateur Random Forest avec 100 arbres
        rf_clf = RandomForestClassifier(n_estimators=100, random_state=42)

        # Appliquer la validation croisée avec 5 folds et évaluer la précision
        cv_scores = cross_val_score(rf_clf, X_scaled, y, cv=5, scoring='accuracy')
        st.write(f'Cross-validated accuracy scores: {cv_scores}')
        st.write(f'Mean accuracy: {np.mean(cv_scores):.4f} (+/- {np.std(cv_scores):.4f})')

        # Utiliser la validation croisée pour obtenir les prédictions
        y_pred_cv = cross_val_predict(rf_clf, X_scaled, y, cv=5)

        # Afficher le rapport de classification basé sur les prédictions de la validation croisée
        st.write("Random Forest Classification Report (cross-validation):")
        st.write(classification_report(y, y_pred_cv))

        # Afficher la matrice de confusion pour la validation croisée
        cm_cv_rf = confusion_matrix(y, y_pred_cv)
        plt.figure(figsize=(6, 4))
        sns.heatmap(cm_cv_rf, annot=True, fmt='d', cmap='Blues')
        plt.title('Random Forest Confusion Matrix (Cross-Validation)')
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        st.pyplot(plt)
        
        # Afficher le rapport de classification pour l'ensemble de test
        st.write("Random Forest Classification Report on test set:")
        st.write(classification_report(y_test, y_pred_test_rf))
        
        st.title("Grid search + cross validation")
        
        # Définir les paramètres à tester
        param_grid = {
            'n_estimators': [i for i in range(50, 300, 50)],  # Nombre d'arbres
            'max_depth': [i for i in range(6, 11)],  # Profondeur maximale des arbres
            'min_samples_split': [i for i in range(4, 11, 2)],  # Nombre minimum d'échantillons pour diviser un nœud
            'bootstrap': [True]        # Si on utilise ou non l'échantillonnage Bootstrap
        }

        # Initialiser le classificateur Random Forest
        rf = RandomForestClassifier(random_state=42)

        # Configurer GridSearchCV
        grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, 
                                cv=5, n_jobs=-1, verbose=2, scoring='accuracy')

        # Exécuter GridSearchCV sur les données d'entraînement
        grid_search.fit(X_train, y_train)

        # Afficher les meilleurs paramètres et le meilleur score
        st.write("Best parameters found: ", grid_search.best_params_)
        st.write("Best accuracy found: ", grid_search.best_score_)

        st.title("Optuna")
        # Diviser les données en ensemble d'entraînement et de test
        X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

        # Fonction de l'objectif à minimiser (Optuna va chercher à maximiser cette fonction)
        def objective(trial):
            # Hyperparamètres à optimiser
            try:
                n_estimators = trial.suggest_int('n_estimators', 50, 300, step=50)
                max_depth = trial.suggest_int('max_depth', 5, 20)
                min_samples_split = trial.suggest_int('min_samples_split', 2, 20)
                bootstrap = trial.suggest_categorical('bootstrap', [True, False])
                
                # Initialisation du modèle RandomForest avec les hyperparamètres choisis
                clf = RandomForestClassifier(
                    n_estimators=n_estimators,
                    max_depth=max_depth,
                    min_samples_split=min_samples_split,
                    bootstrap=bootstrap,
                    random_state=42
                )
                
                # Utilisation de la validation croisée pour évaluer la performance du modèle
                score = cross_val_score(clf, X_train, y_train, cv=5, scoring='accuracy').mean()
                
                return score  # Optuna cherche à maximiser cette valeur
            except FloatingPointError:
                return 0

        # Étude avec Optuna
        study = optuna.create_study(direction='maximize')  # On cherche à maximiser l'accuracy
        study.optimize(objective, n_trials=25)  # Effectuer 50 tests 

        # Meilleurs paramètres trouvés
        st.write("Best parameters found: ", study.best_params)
        st.write("Best cross-validation accuracy: ", study.best_value)

        # Entraîner le modèle avec les meilleurs paramètres trouvés via Optuna
        best_params = study.best_params
        rf_clf = RandomForestClassifier(**best_params, random_state=42)

        # Évaluation finale sur l'ensemble de test
        rf_clf.fit(X_train, y_train)  # Entraîner le modèle sur tout l'ensemble d'entraînement
        y_pred_test = rf_clf.predict(X_test)
        accuracy_test = accuracy_score(y_test, y_pred_test)
        st.write(f'Random Forest Accuracy on test set: {accuracy_test:.4f}')

        # Afficher le rapport de classification pour l'ensemble de test
        st.write("Random Forest Classification Report on test set:")
        st.write(classification_report(y_test, y_pred_test))

        # Afficher la matrice de confusion pour l'ensemble de test
        cm_test_rf = confusion_matrix(y_test, y_pred_test)
        plt.figure(figsize=(6, 4))
        sns.heatmap(cm_test_rf, annot=True, fmt='d', cmap='Blues')
        plt.title('Random Forest Confusion Matrix (Test Set)')
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        st.pyplot(plt)
        
        # Visualisation de l'importance des hyperparamètres
        def plot_param_importance(study):
            optuna.visualization.plot_param_importances(study).show()

        # Visualisation de la distribution des hyperparamètres
        def plot_param_distributions(study):
            optuna.visualization.plot_parallel_coordinate(study).show()

       # Plot and display parameter importance
        fig1 = plot_param_importance(study)
        st.pyplot(fig1)

        # Plot and display parameter distributions
        fig2 = plot_param_distributions(study)
        st.pyplot(fig2)

        st.title("Optuna + sélection des features")
        # Diviser les données en ensemble d'entraînement et de test
        X_train, X_test, y_train, y_test = train_test_split(X_scaled2, y, test_size=0.2, random_state=42)

        # Fonction de l'objectif à minimiser (Optuna va chercher à maximiser cette fonction)
        def objective(trial):
            try:# Hyperparamètres à optimiser
                n_estimators = trial.suggest_int('n_estimators', 100, 300, step=50)
                max_depth = trial.suggest_int('max_depth', 5, 20)
                min_samples_split = trial.suggest_int('min_samples_split', 2, 20)
                bootstrap = trial.suggest_categorical('bootstrap', [True, False])
                
                # Initialisation du modèle RandomForest avec les hyperparamètres choisis
                clf = RandomForestClassifier(
                    n_estimators=n_estimators,
                    max_depth=max_depth,
                    min_samples_split=min_samples_split,
                    bootstrap=bootstrap,
                    random_state=42
                )
                
                # Utilisation de la validation croisée pour évaluer la performance du modèle
                score = cross_val_score(clf, X_train, y_train, cv=5, scoring='accuracy').mean()
                
                return score  # Optuna cherche à maximiser cette valeur
            except FloatingPointError:
                return 0
        
        # Étude avec Optuna
        study = optuna.create_study(direction='maximize')  # On cherche à maximiser l'accuracy
        study.optimize(objective, n_trials=30)  # Effectuer 30 tests (trials)

        # Meilleurs paramètres trouvés
        st.write("Best parameters found: ", study.best_params)
        st.write("Best cross-validation accuracy: ", study.best_value)

        # Entraîner le modèle avec les meilleurs paramètres
        best_params = study.best_params
        rf_clf = RandomForestClassifier(**best_params, random_state=42)
        rf_clf.fit(X_train, y_train)

        # Évaluation sur l'ensemble de test
        y_pred = rf_clf.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        st.write(f'Random Forest Accuracy on test set: {accuracy:.4f}')

                # Visualisation de l'importance des hyperparamètres
        def plot_param_importance(study):
            optuna.visualization.plot_param_importances(study).show()

        # Visualisation de la distribution des hyperparamètres
        def plot_param_distributions(study):
            optuna.visualization.plot_parallel_coordinate(study).show()

        # Plot and display parameter importance
        fig1 = plot_param_importance(study)
        st.pyplot(fig1)

        # Plot and display parameter distributions
        fig2 = plot_param_distributions(study)
        st.pyplot(fig2)

elif st.session_state.page == "Gradient boosting":
    
        data = load_data()
        data= data.dropna()
    
        st.title("Gradient boosting ") 
        
        df_train, df_test = train_test_split(data, test_size=0.2, random_state=42)
        # Drop rows with missing values for 'Arrival Delay in Minutes' since it's a minor portion of the data
        train_data_cleaned = df_train.dropna(subset=['Arrival Delay in Minutes'])
        test_data_cleaned = df_test.dropna(subset=['Arrival Delay in Minutes'])

        # Encode satisfaction (target variable) to binary
        label_encoder = LabelEncoder()
        train_data_cleaned['satisfaction'] = label_encoder.fit_transform(train_data_cleaned['satisfaction'])
        test_data_cleaned['satisfaction'] = label_encoder.transform(test_data_cleaned['satisfaction'])

        # Prendre en compte les variables categoriques:
        categorical_columns = ['Gender', 'Customer Type', 'Type of Travel', 'Class']
        train_data_cleaned = pd.get_dummies(train_data_cleaned, columns=categorical_columns)
        test_data_cleaned = pd.get_dummies(test_data_cleaned, columns=categorical_columns)

        # Separate features and target
        X_train = train_data_cleaned.drop(columns=['satisfaction', 'Unnamed: 0', 'id'])
        y_train = train_data_cleaned['satisfaction']
        X_test = test_data_cleaned.drop(columns=['satisfaction', 'Unnamed: 0', 'id'])
        y_test = test_data_cleaned['satisfaction']

        # Train a Gradient Boosting Classifier as a baseline model
        

        # Train a Gradient Boosting Classifier
        gbc = GradientBoostingClassifier()
        gbc.fit(X_train, y_train)

        # Make predictions on the test set
        y_pred = gbc.predict(X_test)

        # Evaluate the model
        accuracy = accuracy_score(y_test, y_pred)
        classification_rep = classification_report(y_test, y_pred)

        st.write("accuracy",accuracy)
        st.write("classification_rep",classification_rep)
        
        st.title("XGBoost Classifier")
        
        xgb = XGBClassifier(use_label_encoder=False,
                    eval_metric='logloss')

        # Fit the XGBoost model on the training data
        xgb.fit(X_train, y_train)

        # Make predictions on the test data
        y_pred_xgb = xgb.predict(X_test)

        # Evaluate the model performance
        accuracy_xgb = accuracy_score(y_test, y_pred_xgb)
        classification_report_xgb = classification_report(y_test, y_pred_xgb)

        st.write(f"Accuracy: {accuracy_xgb}")
        st.write(f"Classification Report:\n {classification_report_xgb}")

        
        # Define the parameter distribution for RandomizedSearchCV
        param_dist = {
            'n_estimators': [100, 200, 300],
            'learning_rate': uniform(0.01, 0.1),
            'max_depth': [3, 5, 7],
            'subsample': uniform(0.7, 0.3),
            'colsample_bytree': uniform(0.7, 0.3)
        }

        # Initialize XGBoost Classifier
        xgb = XGBClassifier(use_label_encoder=False, eval_metric='logloss')

        # Set up RandomizedSearchCV with XGBoost
        random_search_xgb = RandomizedSearchCV(estimator=xgb, param_distributions=param_dist,
                                            n_iter=20, scoring='accuracy', cv=3, n_jobs=-1, verbose=1)

        # Fit the model on the training data
        random_search_xgb.fit(X_train, y_train)

        # Get best parameters and the best score
        best_params_xgb = random_search_xgb.best_params_
        best_score_xgb = random_search_xgb.best_score_

        st.write("best_params_xgb", best_params_xgb)
        st.write ("best_score_xgb", best_score_xgb)
    
    
    
    
    
    
    
    
    
    
    
    
    
    
   
    
    
    
    
    
    
    
    
