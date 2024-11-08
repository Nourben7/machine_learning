### lien vers l'API application :
https://machinelearning-cj2ffzsebiutlb62zcb3jn.streamlit.app/

ps: certaines pages de l'application prennent du temps pour charger

# Prédiction de la Satisfaction des Passagers Aériens

## Contexte et Objectif

### Objectif
L'objectif principal de ce projet est de prédire la satisfaction des passagers aériens en se basant sur leurs caractéristiques démographiques et leurs évaluations des services en vol. L'analyse vise à fournir des insights exploitables pour améliorer l'expérience client et la fidélisation.

## Enjeux

### 1. Identification des Facteurs de Satisfaction
- **But** : Identifier les variables influençant la satisfaction, telles que la ponctualité, le confort et la qualité du service.
- **Impact** : Prioriser les actions stratégiques pour optimiser l'expérience client.

### 2. Prédiction de la Mécontentement
- **But** : Détecter les passagers susceptibles d'être insatisfaits.
- **Impact** : Mettre en place des actions préventives afin de minimiser les avis négatifs et renforcer la fidélisation.

### 3. Segmentation des Passagers selon leur Satisfaction
- **But** : Regrouper les passagers en segments homogènes selon leurs attentes et niveaux de satisfaction.
- **Impact** : Proposer des services personnalisés et optimiser l'expérience client par segment.

## Description du Projet

Le projet utilise un modèle de machine learning pour classer les passagers en tant que satisfaits ou non satisfaits. Le jeu de données comprend **129 880 lignes** et **25 colonnes**, incluant :

- **Caractéristiques démographiques** : `Sexe`, `Type de client`, `Âge`, `Type de voyage`, `Classe`.
- **Évaluations des services** : `Wi-Fi en vol`, `Confort du siège`, `Divertissement`, etc., notés de 1 à 5.
- **Retards de vol** : `Retards au départ` et `à l’arrivée`.
- **Cible** : `Satisfaction` (satisfait ou non).

## Approche Méthodologique

### 1. Installation et Importation des Bibliothèques

Les bibliothèques suivantes ont été utilisées :
- **Manipulation de données** : `pandas`, `numpy`
- **Visualisation** : `seaborn`, `matplotlib`
- **Machine Learning** : `scikit-learn`
- **Optimisation des hyperparamètres** : `optuna`
- **Analyse exploratoire** : `sweetviz`

### 2. Chargement et Exploration des Données

- **Identification des valeurs manquantes** : 393 lignes avec des valeurs manquantes ont été supprimées.
- **Analyse exploratoire** : Réalisation d'analyses des distributions, des corrélations et des relations avec la variable cible. Les variables clés identifiées incluent `Embarquement en ligne`, `Confort du siège`, `Divertissement en vol`, `Type de voyage` et `Classe`.

> Un rapport détaillé de l'analyse exploratoire (`rapport_EDA.html`) a été généré pour un aperçu global.

### 3. Préprocessing des Données

Avant d'entraîner le modèle, un prétraitement des données a été réalisé pour assurer la qualité et la compatibilité des données avec les algorithmes de machine learning.

- **Encodage de la variable cible (`y`)** :
  La variable `satisfaction` a été encodée en utilisant `LabelEncoder`, attribuant 0 pour "insatisfait ou neutre" et 1 pour "satisfait".
    ```python
    from sklearn.preprocessing import LabelEncoder

    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(df_clean_satisfaction['satisfaction'])
    ```

- **Sélection des caractéristiques (`X`)** :
  Initialement, `X` ne comprenait que les variables numériques présentes dans le jeu de données. Cela a permis de simplifier l'entraînement du modèle baseline avant d'intégrer des techniques avancées.


### 4. Modélisation : K-Nearest Neighbors (KNN)

#### a. Modèle Baseline

- **Pourquoi KNN ?**
Le KNN est un modèle de classification intuitif qui classe les observations en fonction de la similarité avec les k voisins les plus proches. Dans le contexte de la satisfaction des passagers, cela permet de prédire la satisfaction en s'appuyant sur les caractéristiques partagées entre les passagers similaires.

- **Implémentation** :
    ```python
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.model_selection import train_test_split

    # Séparation des données
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Initialisation et entraînement du modèle
    knn = KNeighborsClassifier()
    knn.fit(X_train, y_train)
    ```

#### b. Normalisation des Données

- **Pourquoi normaliser ?**
Le scaling est essentiel pour garantir que toutes les caractéristiques contribuent de manière égale aux calculs de distance, évitant ainsi les biais introduits par des échelles différentes.

- **Méthode** : `StandardScaler()` standardise les données en soustrayant la moyenne et en divisant par l'écart-type.
    ```python
    from sklearn.preprocessing import StandardScaler

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_numeric)
    ```

- **Découpage des données** :
    ```python
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
    ```

## Résultats et Conclusion

- **Impact de la normalisation** : La normalisation des données a eu un effet significatif sur la performance du modèle. En effet, le scaling nous a permis de passer d'une accuracy de 60 % avec la baseline à une accuracy de 90 %, améliorant ainsi considérablement la fiabilité des prédictions.
- **Améliorations** : Utilisation de techniques avancées d'optimisation des hyperparamètres, telles que `optuna`, pour ajuster les valeurs de k et améliorer la performance du modèle.


