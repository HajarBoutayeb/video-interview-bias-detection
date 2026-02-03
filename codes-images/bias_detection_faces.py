import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler, RobustScaler
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score, 
                           classification_report, confusion_matrix, roc_auc_score, roc_curve)
from imblearn.over_sampling import SMOTE, ADASYN, BorderlineSMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.combine import SMOTETomek, SMOTEENN
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

# ================= CHEMINS =================
EXCEL_PATH = r"C:\Users\ADmiN\Desktop\video_project\annotations\faces_annotations.xlsx"

# ================= CONFIGURATION AMÉLIORÉE =================
plt.style.use('seaborn-v0_8')
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 10

# Stratégies d'échantillonnage pour données déséquilibrées
SAMPLING_STRATEGIES = {
    'SMOTE': SMOTE(random_state=42),
    'ADASYN': ADASYN(random_state=42),
    'BorderlineSMOTE': BorderlineSMOTE(random_state=42),
    'SMOTETomek': SMOTETomek(random_state=42),
    'SMOTEENN': SMOTEENN(random_state=42)
}

# Classificateurs améliorés avec meilleurs paramètres
ENHANCED_CLASSIFIERS = {
    "RandomForest": RandomForestClassifier(
        n_estimators=200, 
        max_depth=10,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1
    ),
    "GradientBoosting": GradientBoostingClassifier(
        n_estimators=200,
        learning_rate=0.1,
        max_depth=6,
        random_state=42
    ),
    "AdaBoost": AdaBoostClassifier(
        n_estimators=100,
        learning_rate=1.0,
        random_state=42
    ),
    "SVM": SVC(
        probability=True, 
        random_state=42,
        kernel='rbf',
        C=1.0
    ),
    "LogisticRegression": LogisticRegression(
        random_state=42,
        max_iter=1000,
        multi_class='ovr'
    ),
    "MLP": MLPClassifier(
        hidden_layer_sizes=(100, 50),
        max_iter=500,
        random_state=42,
        early_stopping=True
    ),
    "NaiveBayes": GaussianNB()
}

# ================= CHARGER ET PRÉPARER LES DONNÉES =================
print("🔄 Chargement des données...")
df = pd.read_excel(EXCEL_PATH)
print(f"📊 Taille initiale des données : {df.shape}")

# ================= CRÉER CATÉGORIE D'ÂGE =================
def get_age_category(age):
    if pd.isna(age):
        return "Unknown"
    if age < 13: return "Child"
    elif age < 20: return "Teen"
    elif age < 30: return "Young Adult"
    elif age < 45: return "Adult"
    elif age < 65: return "Middle-aged"
    else: return "Senior"

df['age_category'] = df['age'].apply(get_age_category)

# ================= PRÉTRAITEMENT DES DONNÉES =================
def clean_numeric_columns(df, columns):
    """Nettoyer et convertir les colonnes en numériques"""
    cleaned_df = df.copy()
    
    for col in columns:
        if col in cleaned_df.columns:
            # Convertir en string d'abord, puis nettoyer
            cleaned_df[col] = cleaned_df[col].astype(str)
            # Remplacer les virgules par des points et supprimer les points finaux
            cleaned_df[col] = cleaned_df[col].str.replace(',', '.', regex=False).str.rstrip('.')
            # Remplacer 'nan', 'None', chaînes vides avec NaN
            cleaned_df[col] = cleaned_df[col].replace(['nan', 'None', '', 'null'], np.nan)
            # Convertir en numérique
            cleaned_df[col] = pd.to_numeric(cleaned_df[col], errors='coerce')
            # Remplir NaN avec 0 pour les colonnes de confiance
            if 'confidence' in col:
                cleaned_df[col] = cleaned_df[col].fillna(0.0)
            else:
                cleaned_df[col] = cleaned_df[col].fillna(cleaned_df[col].median())
    
    return cleaned_df

# ================= INGÉNIERIE DE CARACTÉRISTIQUES AMÉLIORÉE =================
def create_enhanced_features(df):
    """Créer des caractéristiques supplémentaires pour une meilleure prédiction"""
    print("🔧 Création de caractéristiques améliorées...")
    
    # D'abord, nettoyer toutes les colonnes numériques
    numeric_columns = ['age', 'gender_confidence', 'race_confidence', 'emotion_confidence', 
                      'bbox_w', 'bbox_h', 'bbox_x', 'bbox_y', 'face_area']
    
    enhanced_df = clean_numeric_columns(df, numeric_columns)
    
    # S'assurer d'avoir les colonnes requises pour les calculs
    required_cols = ['gender_confidence', 'race_confidence', 'emotion_confidence', 'age']
    for col in required_cols:
        if col not in enhanced_df.columns:
            enhanced_df[col] = 0.0
        # S'assurer qu'elles sont numériques
        enhanced_df[col] = pd.to_numeric(enhanced_df[col], errors='coerce').fillna(0.0)
    
    try:
        # Ratios et combinaisons de confiance (avec division sécurisée)
        enhanced_df['conf_ratio_gender_race'] = enhanced_df['gender_confidence'] / (enhanced_df['race_confidence'] + 0.001)
        enhanced_df['conf_ratio_emotion_gender'] = enhanced_df['emotion_confidence'] / (enhanced_df['gender_confidence'] + 0.001)
        enhanced_df['conf_product'] = enhanced_df['gender_confidence'] * enhanced_df['race_confidence'] * enhanced_df['emotion_confidence']
        
        # Gérer les valeurs infinies
        enhanced_df['conf_ratio_gender_race'] = enhanced_df['conf_ratio_gender_race'].replace([np.inf, -np.inf], 0)
        enhanced_df['conf_ratio_emotion_gender'] = enhanced_df['conf_ratio_emotion_gender'].replace([np.inf, -np.inf], 0)
        
        print("✅ Ratios de confiance créés avec succès")
        
    except Exception as e:
        print(f"⚠️ Erreur lors de la création des ratios de confiance : {e}")
        enhanced_df['conf_ratio_gender_race'] = 0.0
        enhanced_df['conf_ratio_emotion_gender'] = 0.0
        enhanced_df['conf_product'] = 0.0
    
    try:
        # Caractéristiques géométriques du visage
        if 'bbox_w' in enhanced_df.columns and 'bbox_h' in enhanced_df.columns:
            enhanced_df['bbox_w'] = pd.to_numeric(enhanced_df['bbox_w'], errors='coerce').fillna(100)
            enhanced_df['bbox_h'] = pd.to_numeric(enhanced_df['bbox_h'], errors='coerce').fillna(100)
            
            enhanced_df['face_aspect_ratio'] = enhanced_df['bbox_w'] / (enhanced_df['bbox_h'] + 1)
            enhanced_df['face_area_norm'] = enhanced_df['bbox_w'] * enhanced_df['bbox_h']
            
            print("✅ Caractéristiques géométriques créées avec succès")
        else:
            enhanced_df['face_aspect_ratio'] = 1.0
            enhanced_df['face_area_norm'] = 10000.0
            print("⚠️ Colonnes bbox introuvables, utilisation de valeurs par défaut")
            
    except Exception as e:
        print(f"⚠️ Erreur lors de la création des caractéristiques géométriques : {e}")
        enhanced_df['face_aspect_ratio'] = 1.0
        enhanced_df['face_area_norm'] = 10000.0
    
    try:
        # Caractéristiques basées sur l'âge
        enhanced_df['age'] = pd.to_numeric(enhanced_df['age'], errors='coerce').fillna(25.0)
        enhanced_df['is_young'] = (enhanced_df['age'] < 25).astype(int)
        enhanced_df['is_senior'] = (enhanced_df['age'] > 60).astype(int)
        enhanced_df['age_squared'] = enhanced_df['age'] ** 2
        
        print("✅ Caractéristiques basées sur l'âge créées avec succès")
        
    except Exception as e:
        print(f"⚠️ Erreur lors de la création des caractéristiques d'âge : {e}")
        enhanced_df['is_young'] = 0
        enhanced_df['is_senior'] = 0
        enhanced_df['age_squared'] = 625.0
    
    print(f"📊 Caractéristiques améliorées créées. Nouvelle forme : {enhanced_df.shape}")
    
    return enhanced_df

# ================= FOCUS SUR RACE ET ÉMOTION =================
target_labels = ['race', 'emotion']

# Caractéristiques améliorées
base_features = ['age', 'gender_confidence', 'race_confidence', 'emotion_confidence']
enhanced_features = [
    'conf_ratio_gender_race', 'conf_ratio_emotion_gender', 'conf_product',
    'is_young', 'is_senior', 'age_squared'
]

# Ajouter les caractéristiques géométriques si disponibles
geometry_features = []
if 'bbox_w' in df.columns:
    geometry_features = ['face_aspect_ratio', 'face_area_norm']
    enhanced_features.extend(geometry_features)

all_features = base_features + enhanced_features

df = create_enhanced_features(df)

# Vérifier quelles colonnes nous avons après la création de caractéristiques
print("📋 Colonnes disponibles après l'ingénierie de caractéristiques :")
available_columns = df.columns.tolist()
for i, col in enumerate(available_columns):
    print(f"   {i+1:2d}. {col}")

# Définir les caractéristiques de base
base_features = ['age', 'gender_confidence', 'race_confidence', 'emotion_confidence', 'avg_confidence']

# Caractéristiques améliorées (inclure uniquement celles qui existent)
enhanced_features = [
    'conf_ratio_gender_race', 'conf_ratio_emotion_gender', 'conf_product',
    'is_young', 'is_senior', 'age_squared'
]

# Ajouter les caractéristiques géométriques si elles existent
geometry_features = []
if 'face_aspect_ratio' in df.columns:
    geometry_features = ['face_aspect_ratio', 'face_area_norm']
    enhanced_features.extend(geometry_features)

# Vérifier quelles caractéristiques de base sont disponibles
available_base_features = []
for feature in base_features:
    if feature in df.columns:
        available_base_features.append(feature)
    else:
        print(f"⚠️ Caractéristique de base introuvable : {feature}")

# Vérifier quelles caractéristiques améliorées sont disponibles
available_enhanced_features = []
for feature in enhanced_features:
    if feature in df.columns:
        available_enhanced_features.append(feature)
    else:
        print(f"⚠️ Caractéristique améliorée introuvable : {feature}")

all_features = available_base_features + available_enhanced_features
print(f"✅ Total des caractéristiques disponibles : {len(all_features)} caractéristiques")
print(f"📊 Caractéristiques de base : {available_base_features}")
print(f"🔧 Caractéristiques améliorées : {available_enhanced_features}")


# ================= NETTOYAGE DES DONNÉES ET PRÉPARATION DES CARACTÉRISTIQUES =================
# Supprimer les lignes avec des étiquettes cibles manquantes
print("🧹 Nettoyage des données...")
for label in target_labels:
    initial_count = len(df)
    df = df.dropna(subset=[label])
    removed = initial_count - len(df)
    if removed > 0:
        print(f"🧹 Supprimé {removed} lignes avec {label} manquant")

# Caractéristiques améliorées (seront nettoyées dans la fonction)
all_features = base_features + enhanced_features

# ================= PRÉPARATION FINALE DES DONNÉES =================
print("🔄 Traitement des caractéristiques...")

# Obtenir uniquement les caractéristiques qui existent dans le dataframe
available_features = [f for f in all_features if f in df.columns]
if len(available_features) < len(all_features):
    missing_features = set(all_features) - set(available_features)
    print(f"⚠️ Caractéristiques manquantes : {missing_features}")
    print(f"📊 Utilisation de {len(available_features)} caractéristiques disponibles")

# Créer une matrice de caractéristiques avec un nettoyage robuste
X = df[available_features].copy()

# Nettoyer toutes les caractéristiques systématiquement
for col in X.columns:
    if X[col].dtype == 'object':
        X[col] = X[col].astype(str).str.replace(',', '.', regex=False).str.rstrip('.')
        X[col] = X[col].replace(['nan', 'None', '', 'null'], np.nan)
    
    X[col] = pd.to_numeric(X[col], errors='coerce')

# Remplir les valeurs manquantes avec la médiane pour chaque colonne
X = X.fillna(X.median())

# Gérer tout problème restant
X = X.replace([np.inf, -np.inf], 0)

print(f"📊 Forme finale de la matrice de caractéristiques : {X.shape}")
print(f"🎯 Caractéristiques utilisées : {list(X.columns)}")
print(f"✅ Prétraitement des données terminé avec succès")

# ================= FONCTION D'ÉVALUATION AVANCÉE =================
def advanced_evaluate_classifier(X_train, X_test, y_train, y_test, clf, label_name):
    """Évaluation améliorée avec plusieurs métriques"""
    
    # Entraîner le classificateur
    clf.fit(X_train, y_train)
    
    # Prédictions
    y_pred = clf.predict(X_test)
    y_pred_proba = None
    
    try:
        y_pred_proba = clf.predict_proba(X_test)
    except:
        pass
    
    # Calculer les métriques
    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, average='weighted', zero_division=0),
        "recall": recall_score(y_test, y_pred, average='weighted', zero_division=0),
        "f1": f1_score(y_test, y_pred, average='weighted', zero_division=0),
        "f1_macro": f1_score(y_test, y_pred, average='macro', zero_division=0),
        "f1_micro": f1_score(y_test, y_pred, average='micro', zero_division=0)
    }
    
    # ROC AUC pour multi-classes
    if y_pred_proba is not None and len(np.unique(y_test)) > 2:
        try:
            metrics["roc_auc"] = roc_auc_score(y_test, y_pred_proba, multi_class='ovr', average='weighted')
        except:
            metrics["roc_auc"] = 0
    elif y_pred_proba is not None and len(np.unique(y_test)) == 2:
        try:
            metrics["roc_auc"] = roc_auc_score(y_test, y_pred_proba[:, 1])
        except:
            metrics["roc_auc"] = 0
    else:
        metrics["roc_auc"] = 0
    
    return metrics, y_pred, y_pred_proba

# ================= FONCTIONS DE VISUALISATION =================
def plot_class_distribution(y, title, label_encoder=None):
    """Tracer la distribution des classes"""
    plt.figure(figsize=(12, 6))
    
    if label_encoder:
        labels = label_encoder.inverse_transform(range(len(label_encoder.classes_)))
        counts = pd.Series(y).value_counts().sort_index()
        plt.bar(range(len(labels)), [counts.get(i, 0) for i in range(len(labels))])
        plt.xticks(range(len(labels)), labels, rotation=45)
    else:
        pd.Series(y).value_counts().plot(kind='bar')
        plt.xticks(rotation=45)
    
    plt.title(f'{title} - Distribution des Classes')
    plt.xlabel('Classes')
    plt.ylabel('Nombre')
    plt.tight_layout()
    plt.show()

def plot_confusion_matrix(y_true, y_pred, classes, title):
    """Tracer la matrice de confusion améliorée"""
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=classes, yticklabels=classes)
    plt.title(f'{title} - Matrice de Confusion')
    plt.xlabel('Prédit')
    plt.ylabel('Réel')
    plt.tight_layout()
    plt.show()

def plot_metrics_comparison(results_dict, title):
    """Tracer la comparaison des métriques entre les modèles"""
    models = list(results_dict.keys())
    metrics = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()
    
    for i, metric in enumerate(metrics):
        if i < len(axes):
            values = [results_dict[model][metric] * 100 for model in models]
            bars = axes[i].bar(models, values)
            axes[i].set_title(f'Comparaison {metric.upper()}')
            axes[i].set_ylabel('Pourcentage (%)')
            axes[i].set_ylim(0, 100)
            
            # Ajouter des étiquettes de valeur sur les barres
            for bar, value in zip(bars, values):
                axes[i].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, 
                           f'{value:.1f}%', ha='center', va='bottom')
            
            axes[i].tick_params(axis='x', rotation=45)
    
    # Supprimer le sous-graphique vide
    if len(axes) > len(metrics):
        fig.delaxes(axes[-1])
    
    plt.suptitle(f'{title} - Comparaison des Performances des Modèles', fontsize=16)
    plt.tight_layout()
    plt.show()

# ================= RÉGLAGE DES HYPERPARAMÈTRES =================
def tune_hyperparameters(X_train, y_train, clf_name, clf):
    """Effectuer le réglage des hyperparamètres pour des classificateurs spécifiques"""
    
    param_grids = {
        'RandomForest': {
            'n_estimators': [100, 200],
            'max_depth': [5, 10, None],
            'min_samples_split': [2, 5]
        },
        'GradientBoosting': {
            'n_estimators': [100, 200],
            'learning_rate': [0.05, 0.1, 0.2],
            'max_depth': [3, 6, 9]
        },
        'SVM': {
            'C': [0.1, 1, 10],
            'kernel': ['rbf', 'linear'],
            'gamma': ['scale', 'auto']
        }
    }
    
    if clf_name not in param_grids:
        return clf
    
    print(f"🔧 Réglage des hyperparamètres de {clf_name}...")
    
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    grid_search = GridSearchCV(
        clf, param_grids[clf_name], 
        cv=cv, scoring='f1_weighted', 
        n_jobs=-1, verbose=0
    )
    
    grid_search.fit(X_train, y_train)
    print(f"✅ Meilleurs paramètres pour {clf_name} : {grid_search.best_params_}")
    
    return grid_search.best_estimator_

# ================= BOUCLE D'ÉVALUATION PRINCIPALE =================
def evaluate_target_labels():
    """Fonction d'évaluation principale pour la race et l'émotion"""
    
    all_results = {}
    
    for label in target_labels:
        print(f"\n{'='*60}")
        print(f"🎯 ÉVALUATION DE L'ÉTIQUETTE : {label.upper()}")
        print(f"{'='*60}")
        
        # Préparer la variable cible
        y = df[label].dropna()
        X_filtered = X.loc[y.index]
        
        # Encoder les étiquettes
        le = LabelEncoder()
        y_encoded = le.fit_transform(y.astype(str))
        
        print(f"📊 Distribution des classes pour {label} :")
        class_counts = pd.Series(y).value_counts()
        for cls, count in class_counts.items():
            percentage = (count / len(y)) * 100
            print(f"   - {cls} : {count} ({percentage:.1f}%)")
        
        # Tracer la distribution des classes
        plot_class_distribution(y, f'Distribution de {label.capitalize()}')
        
        # Vérifier la stratification
        min_class_count = min(pd.Series(y_encoded).value_counts())
        stratify = y_encoded if min_class_count >= 2 else None
        
        # Diviser les données
        X_train, X_test, y_train, y_test = train_test_split(
            X_filtered, y_encoded, test_size=0.2, random_state=42, stratify=stratify
        )
        
        print(f"📈 Taille de l'ensemble d'entraînement : {len(X_train)}")
        print(f"📈 Taille de l'ensemble de test : {len(X_test)}")
        
        # Normaliser les caractéristiques
        scaler = RobustScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Essayer différentes stratégies d'échantillonnage
        best_sampling = None
        best_sampling_score = 0
        
        print(f"\n🔍 Test des stratégies d'échantillonnage...")
        
        for sampling_name, sampler in SAMPLING_STRATEGIES.items():
            try:
                X_train_resampled, y_train_resampled = sampler.fit_resample(X_train_scaled, y_train)
                
                # Évaluation rapide avec RandomForest
                rf = RandomForestClassifier(n_estimators=50, random_state=42)
                cv_scores = cross_val_score(rf, X_train_resampled, y_train_resampled, 
                                          cv=3, scoring='f1_weighted')
                avg_score = cv_scores.mean()
                
                print(f"   - {sampling_name} : {avg_score:.3f}")
                
                if avg_score > best_sampling_score:
                    best_sampling_score = avg_score
                    best_sampling = sampling_name
                    
            except Exception as e:
                print(f"   - {sampling_name} : Échec ({str(e)[:50]}...)")
        
        # Appliquer la meilleure stratégie d'échantillonnage
        if best_sampling:
            print(f"✅ Meilleure stratégie d'échantillonnage : {best_sampling}")
            sampler = SAMPLING_STRATEGIES[best_sampling]
            X_train_final, y_train_final = sampler.fit_resample(X_train_scaled, y_train)
            print(f"📊 Après rééchantillonnage : {len(X_train_final)} échantillons")
        else:
            X_train_final, y_train_final = X_train_scaled, y_train
            print("⚠️ Aucun échantillonnage appliqué")
        
        # Évaluer les classificateurs
        results = {}
        predictions = {}
        
        print(f"\n🚀 Évaluation des classificateurs...")
        
        for clf_name, clf in ENHANCED_CLASSIFIERS.items():
            print(f"\n--- {clf_name} ---")
            
            try:
                # Réglage des hyperparamètres pour les modèles sélectionnés
                if clf_name in ['RandomForest', 'GradientBoosting', 'SVM']:
                    tuned_clf = tune_hyperparameters(X_train_final, y_train_final, clf_name, clf)
                else:
                    tuned_clf = clf
                
                # Évaluer
                metrics, y_pred, y_pred_proba = advanced_evaluate_classifier(
                    X_train_final, X_test_scaled, y_train_final, y_test, tuned_clf, label
                )
                
                results[clf_name] = metrics
                predictions[clf_name] = (y_pred, y_pred_proba)
                
                # Afficher les résultats
                print(f"Précision :  {metrics['accuracy']*100:.2f}%")
                print(f"Précision : {metrics['precision']*100:.2f}%")
                print(f"Rappel :    {metrics['recall']*100:.2f}%")
                print(f"F1-Score :  {metrics['f1']*100:.2f}%")
                print(f"F1-Macro :  {metrics['f1_macro']*100:.2f}%")
                print(f"ROC-AUC :   {metrics['roc_auc']*100:.2f}%")
                
                # Validation croisée
                cv_scores = cross_val_score(tuned_clf, X_train_final, y_train_final, 
                                          cv=5, scoring='f1_weighted')
                print(f"CV F1 :     {cv_scores.mean()*100:.2f}% ± {cv_scores.std()*100:.2f}%")
                
            except Exception as e:
                print(f"❌ Erreur avec {clf_name} : {e}")
                results[clf_name] = {
                    'accuracy': 0, 'precision': 0, 'recall': 0, 
                    'f1': 0, 'f1_macro': 0, 'f1_micro': 0, 'roc_auc': 0
                }
        
        # Trouver le meilleur modèle
        best_f1 = max(results.values(), key=lambda x: x['f1'])['f1']
        best_models = [name for name, metrics in results.items() if metrics['f1'] == best_f1]
        
        print(f"\n🏆 MEILLEUR(S) MODÈLE(S) pour {label} : {', '.join(best_models)}")
        print(f"🎯 Meilleur F1-Score : {best_f1*100:.2f}%")
        
        # Analyse détaillée pour le meilleur modèle
        best_model_name = best_models[0]
        best_y_pred, best_y_pred_proba = predictions[best_model_name]
        
        # Rapport de classification
        print(f"\n📋 Rapport de Classification Détaillé pour {best_model_name} :")
        report = classification_report(y_test, best_y_pred, 
                                     target_names=le.classes_, 
                                     zero_division=0)
        print(report)
        
        # Matrice de Confusion
        plot_confusion_matrix(y_test, best_y_pred, le.classes_, 
                            f'{label.capitalize()} - {best_model_name}')
        
        # Graphique de comparaison des métriques
        plot_metrics_comparison(results, f'Classification de {label.capitalize()}')
        
        # Stocker les résultats
        all_results[label] = {
            'best_model': best_model_name,
            'best_score': best_f1,
            'all_results': results,
            'class_distribution': class_counts.to_dict(),
            'sampling_strategy': best_sampling
        }
        
        # Importance des caractéristiques pour les modèles basés sur les arbres
        if best_model_name in ['RandomForest', 'GradientBoosting', 'AdaBoost']:
            best_clf = ENHANCED_CLASSIFIERS[best_model_name]
            if best_model_name in ['RandomForest', 'GradientBoosting', 'SVM']:
                best_clf = tune_hyperparameters(X_train_final, y_train_final, best_model_name, best_clf)
            
            best_clf.fit(X_train_final, y_train_final)
            
            if hasattr(best_clf, 'feature_importances_'):
                feature_importance = pd.DataFrame({
                    'feature': X.columns,
                    'importance': best_clf.feature_importances_
                }).sort_values('importance', ascending=False)
                
                print(f"\n🔍 Importance des Caractéristiques pour {best_model_name} :")
                for idx, row in feature_importance.head(10).iterrows():
                    print(f"   {row['feature']} : {row['importance']:.4f}")
                
                # Tracer l'importance des caractéristiques
                plt.figure(figsize=(12, 8))
                sns.barplot(data=feature_importance.head(15), x='importance', y='feature')
                plt.title(f'{label.capitalize()} - Importance des Caractéristiques ({best_model_name})')
                plt.xlabel('Importance')
                plt.tight_layout()
                plt.show()
    
    return all_results

# ================= RAPPORT RÉCAPITULATIF =================
def generate_summary_report(all_results):
    """Générer un rapport récapitulatif complet"""
    
    print(f"\n{'='*80}")
    print(f"📊 RÉSUMÉ D'ÉVALUATION COMPLET")
    print(f"{'='*80}")
    
    summary_data = []
    
    for label, results in all_results.items():
        best_model = results['best_model']
        best_score = results['best_score']
        sampling = results['sampling_strategy']
        n_classes = len(results['class_distribution'])
        
        summary_data.append({
            'Étiquette': label.capitalize(),
            'Meilleur Modèle': best_model,
            'F1-Score (%)': f"{best_score*100:.2f}%",
            'Classes': n_classes,
            'Stratégie Échantillonnage': sampling or 'Aucune'
        })
        
        print(f"\n🎯 {label.upper()} :")
        print(f"   - Meilleur Modèle : {best_model}")
        print(f"   - F1-Score : {best_score*100:.2f}%")
        print(f"   - Nombre de Classes : {n_classes}")
        print(f"   - Stratégie d'Échantillonnage : {sampling or 'Aucune'}")
        print(f"   - Distribution des Classes : {results['class_distribution']}")
    
    # Créer un DataFrame récapitulatif
    summary_df = pd.DataFrame(summary_data)
    print(f"\n📋 TABLEAU RÉCAPITULATIF :")
    print(summary_df.to_string(index=False))
    
    return summary_df

# ================= EXÉCUTER L'ÉVALUATION =================
if __name__ == "__main__":
    print("🚀 Démarrage du Système d'Évaluation Amélioré Race & Émotion")
    print(f"📊 Taille du dataset : {df.shape}")
    print(f"🎯 Étiquettes cibles : {target_labels}")
    print(f"📈 Caractéristiques : {len(all_features)} caractéristiques")
    print(f"🤖 Classificateurs : {len(ENHANCED_CLASSIFIERS)} modèles")
    print(f"⚖️ Stratégies d'échantillonnage : {len(SAMPLING_STRATEGIES)} méthodes")
    
    # Exécuter l'évaluation
    results = evaluate_target_labels()
    
    # Générer le résumé
    summary = generate_summary_report(results)
    
    print(f"\n✅ Évaluation terminée avec succès !")
    print(f"💡 Consultez les graphiques et rapports générés ci-dessus pour des informations détaillées.")