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

# ================= CONFIGURATION =================
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

# Classificateurs améliorés optimisés pour le genre et l'âge
CLASSIFIERS = {
    "RandomForest": RandomForestClassifier(
        n_estimators=300, 
        max_depth=15,
        min_samples_split=3,
        min_samples_leaf=1,
        random_state=42,
        n_jobs=-1
    ),
    "GradientBoosting": GradientBoostingClassifier(
        n_estimators=300,
        learning_rate=0.08,
        max_depth=8,
        random_state=42
    ),
    "AdaBoost": AdaBoostClassifier(
        n_estimators=150,
        learning_rate=0.8,
        random_state=42
    ),
    "SVM": SVC(
        probability=True, 
        random_state=42,
        kernel='rbf',
        C=10.0,
        gamma='scale'
    ),
    "LogisticRegression": LogisticRegression(
        random_state=42,
        max_iter=2000,
        C=1.0,
        solver='lbfgs'
    ),
    "MLP": MLPClassifier(
        hidden_layer_sizes=(150, 100, 50),
        max_iter=1000,
        random_state=42,
        early_stopping=True,
        validation_fraction=0.15
    ),
    "NaiveBayes": GaussianNB()
}

# ================= CHARGER LES DONNÉES =================
print("🔄 Chargement des données pour l'analyse Genre & Âge...")
df = pd.read_excel(EXCEL_PATH)
print(f"📊 Taille initiale des données : {df.shape}")

# ================= ÉTIQUETTES CIBLES =================
target_labels = ['gender', 'age_category']

print(f"🎯 Étiquettes cibles : {target_labels}")
print("   - Genre : Classification Homme/Femme")
print("   - Catégorie d'Âge : Classification par groupe d'âge")

# ================= CRÉER CATÉGORIES D'ÂGE =================
def get_age_category(age):
    if pd.isna(age):
        return "Unknown"
    if age < 13: return "Child"
    elif age < 20: return "Teen"
    elif age < 30: return "Young Adult"
    elif age < 45: return "Adult"
    elif age < 65: return "Middle-aged"
    else: return "Senior"

# Créer la catégorie d'âge si elle n'existe pas
if 'age_category' not in df.columns:
    df['age_category'] = df['age'].apply(get_age_category)

# Supprimer les catégories d'âge inconnues pour un meilleur entraînement
df = df[df['age_category'] != 'Unknown']

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
            # Remplir NaN avec des valeurs appropriées
            if 'confidence' in col:
                cleaned_df[col] = cleaned_df[col].fillna(0.5)  # Confiance par défaut
            else:
                cleaned_df[col] = cleaned_df[col].fillna(cleaned_df[col].median())
    
    return cleaned_df

# ================= INGÉNIERIE DE CARACTÉRISTIQUES POUR GENRE & ÂGE =================
def create_gender_age_features(df):
    """Créer des caractéristiques optimisées pour la prédiction du genre et de l'âge"""
    print("🔧 Création de caractéristiques spécifiques Genre & Âge...")
    
    # D'abord, nettoyer les colonnes numériques
    numeric_columns = ['age', 'gender_confidence', 'race_confidence', 'emotion_confidence',
                      'bbox_w', 'bbox_h', 'bbox_x', 'bbox_y', 'face_area']
    
    enhanced_df = clean_numeric_columns(df, numeric_columns)
    
    # S'assurer que les colonnes requises existent
    required_cols = ['gender_confidence', 'race_confidence', 'emotion_confidence', 'age']
    for col in required_cols:
        if col not in enhanced_df.columns:
            enhanced_df[col] = 0.5
        enhanced_df[col] = pd.to_numeric(enhanced_df[col], errors='coerce').fillna(0.5)
    
    try:
        # Caractéristiques spécifiques au genre
        enhanced_df['gender_conf_squared'] = enhanced_df['gender_confidence'] ** 2
        enhanced_df['gender_conf_log'] = np.log(enhanced_df['gender_confidence'] + 0.001)
        enhanced_df['high_gender_conf'] = (enhanced_df['gender_confidence'] > 0.8).astype(int)
        
        print("✅ Caractéristiques de genre créées")
        
    except Exception as e:
        print(f"⚠️ Erreur lors de la création des caractéristiques de genre : {e}")
        enhanced_df['gender_conf_squared'] = 0.25
        enhanced_df['gender_conf_log'] = -1.0
        enhanced_df['high_gender_conf'] = 0
    
    try:
        # Caractéristiques spécifiques à l'âge
        enhanced_df['age'] = pd.to_numeric(enhanced_df['age'], errors='coerce').fillna(30.0)
        enhanced_df['age_squared'] = enhanced_df['age'] ** 2
        enhanced_df['age_cubed'] = enhanced_df['age'] ** 3
        enhanced_df['age_log'] = np.log(enhanced_df['age'] + 1)
        enhanced_df['age_sqrt'] = np.sqrt(enhanced_df['age'])
        
        # Indicateurs de groupe d'âge
        enhanced_df['is_child'] = (enhanced_df['age'] < 13).astype(int)
        enhanced_df['is_teen'] = ((enhanced_df['age'] >= 13) & (enhanced_df['age'] < 20)).astype(int)
        enhanced_df['is_young_adult'] = ((enhanced_df['age'] >= 20) & (enhanced_df['age'] < 30)).astype(int)
        enhanced_df['is_adult'] = ((enhanced_df['age'] >= 30) & (enhanced_df['age'] < 45)).astype(int)
        enhanced_df['is_middle_aged'] = ((enhanced_df['age'] >= 45) & (enhanced_df['age'] < 65)).astype(int)
        enhanced_df['is_senior'] = (enhanced_df['age'] >= 65).astype(int)
        
        print("✅ Caractéristiques d'âge créées")
        
    except Exception as e:
        print(f"⚠️ Erreur lors de la création des caractéristiques d'âge : {e}")
        for feature in ['age_squared', 'age_cubed', 'age_log', 'age_sqrt']:
            enhanced_df[feature] = 900.0
        for feature in ['is_child', 'is_teen', 'is_young_adult', 'is_adult', 'is_middle_aged', 'is_senior']:
            enhanced_df[feature] = 0
    
    try:
        # Caractéristiques géométriques du visage (utiles pour genre/âge)
        if 'bbox_w' in enhanced_df.columns and 'bbox_h' in enhanced_df.columns:
            enhanced_df['bbox_w'] = pd.to_numeric(enhanced_df['bbox_w'], errors='coerce').fillna(100)
            enhanced_df['bbox_h'] = pd.to_numeric(enhanced_df['bbox_h'], errors='coerce').fillna(100)
            
            enhanced_df['face_aspect_ratio'] = enhanced_df['bbox_w'] / (enhanced_df['bbox_h'] + 1)
            enhanced_df['face_area_norm'] = enhanced_df['bbox_w'] * enhanced_df['bbox_h']
            enhanced_df['face_perimeter'] = 2 * (enhanced_df['bbox_w'] + enhanced_df['bbox_h'])
            enhanced_df['face_compactness'] = (4 * np.pi * enhanced_df['face_area_norm']) / (enhanced_df['face_perimeter'] ** 2)
            
            print("✅ Caractéristiques géométriques créées")
        else:
            enhanced_df['face_aspect_ratio'] = 1.0
            enhanced_df['face_area_norm'] = 10000.0
            enhanced_df['face_perimeter'] = 400.0
            enhanced_df['face_compactness'] = 0.785
            print("⚠️ Utilisation de valeurs géométriques par défaut")
            
    except Exception as e:
        print(f"⚠️ Erreur lors de la création des caractéristiques géométriques : {e}")
        enhanced_df['face_aspect_ratio'] = 1.0
        enhanced_df['face_area_norm'] = 10000.0
        enhanced_df['face_perimeter'] = 400.0
        enhanced_df['face_compactness'] = 0.785
    
    try:
        # Interactions croisées entre caractéristiques
        enhanced_df['age_gender_conf'] = enhanced_df['age'] * enhanced_df['gender_confidence']
        enhanced_df['conf_diversity'] = enhanced_df['gender_confidence'] * enhanced_df['race_confidence'] * enhanced_df['emotion_confidence']
        enhanced_df['conf_avg'] = (enhanced_df['gender_confidence'] + enhanced_df['race_confidence'] + enhanced_df['emotion_confidence']) / 3
        
        print("✅ Caractéristiques d'interaction créées")
        
    except Exception as e:
        print(f"⚠️ Erreur lors de la création des caractéristiques d'interaction : {e}")
        enhanced_df['age_gender_conf'] = 15.0
        enhanced_df['conf_diversity'] = 0.125
        enhanced_df['conf_avg'] = 0.5
    
    print(f"📊 Caractéristiques améliorées créées. Nouvelle forme : {enhanced_df.shape}")
    return enhanced_df

# Appliquer l'ingénierie de caractéristiques
df = create_gender_age_features(df)

# ================= DÉFINIR LES CARACTÉRISTIQUES =================
# Caractéristiques de base
base_features = ['age', 'gender_confidence', 'race_confidence', 'emotion_confidence']

# Caractéristiques spécifiques au genre
gender_features = ['gender_conf_squared', 'gender_conf_log', 'high_gender_conf']

# Caractéristiques spécifiques à l'âge
age_features = ['age_squared', 'age_cubed', 'age_log', 'age_sqrt', 
               'is_child', 'is_teen', 'is_young_adult', 'is_adult', 'is_middle_aged', 'is_senior']

# Caractéristiques géométriques
geometry_features = ['face_aspect_ratio', 'face_area_norm', 'face_perimeter', 'face_compactness']

# Caractéristiques d'interaction
interaction_features = ['age_gender_conf', 'conf_diversity', 'conf_avg']

# Toutes les caractéristiques
all_features = base_features + gender_features + age_features + geometry_features + interaction_features

# Vérifier quelles caractéristiques sont disponibles
available_features = [f for f in all_features if f in df.columns]
print(f"📊 Total des caractéristiques : {len(available_features)} sur {len(all_features)}")

# ================= NETTOYAGE DES DONNÉES =================
print("🧹 Nettoyage final des données...")

# Supprimer les lignes avec des étiquettes cibles manquantes
for label in target_labels:
    initial_count = len(df)
    df = df.dropna(subset=[label])
    removed = initial_count - len(df)
    if removed > 0:
        print(f"🧹 Supprimé {removed} lignes avec {label} manquant")

# Créer la matrice de caractéristiques
X = df[available_features].copy()

# Nettoyage final
for col in X.columns:
    X[col] = pd.to_numeric(X[col], errors='coerce')

X = X.fillna(X.median())
X = X.replace([np.inf, -np.inf], 0)

print(f"✅ Forme finale du dataset : {X.shape}")
print(f"🎯 Caractéristiques utilisées : {len(X.columns)} caractéristiques")

# ================= FONCTIONS D'ÉVALUATION =================
def evaluate_classifier_detailed(X_train, X_test, y_train, y_test, clf, label_name):
    """Évaluation améliorée avec plusieurs métriques"""
    
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    
    try:
        y_pred_proba = clf.predict_proba(X_test)
    except:
        y_pred_proba = None
    
    # Calculer les métriques
    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, average='weighted', zero_division=0),
        "recall": recall_score(y_test, y_pred, average='weighted', zero_division=0),
        "f1_weighted": f1_score(y_test, y_pred, average='weighted', zero_division=0),
        "f1_macro": f1_score(y_test, y_pred, average='macro', zero_division=0),
    }
    
    # ROC AUC pour binaire/multi-classes
    if y_pred_proba is not None:
        try:
            if len(np.unique(y_test)) == 2:
                metrics["roc_auc"] = roc_auc_score(y_test, y_pred_proba[:, 1])
            else:
                metrics["roc_auc"] = roc_auc_score(y_test, y_pred_proba, multi_class='ovr', average='weighted')
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
        labels = label_encoder.classes_
        counts = pd.Series(y).value_counts()
        label_counts = [counts.get(i, 0) for i in range(len(labels))]
        bars = plt.bar(range(len(labels)), label_counts)
        plt.xticks(range(len(labels)), labels, rotation=45)
        
        # Ajouter des étiquettes de pourcentage sur les barres
        total = sum(label_counts)
        for bar, count in zip(bars, label_counts):
            percentage = (count / total) * 100 if total > 0 else 0
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(label_counts)*0.01, 
                    f'{count}\n({percentage:.1f}%)', ha='center', va='bottom')
    else:
        counts = pd.Series(y).value_counts()
        bars = counts.plot(kind='bar')
        plt.xticks(rotation=45)
        
        # Ajouter des étiquettes de pourcentage
        total = len(y)
        for i, (idx, count) in enumerate(counts.items()):
            percentage = (count / total) * 100
            plt.text(i, count + max(counts)*0.01, f'{count}\n({percentage:.1f}%)', 
                    ha='center', va='bottom')
    
    plt.title(f'{title} - Distribution des Classes')
    plt.xlabel('Classes')
    plt.ylabel('Nombre')
    plt.tight_layout()
    plt.show()

def plot_confusion_matrix_enhanced(y_true, y_pred, classes, title):
    """Tracer la matrice de confusion améliorée avec pourcentages"""
    cm = confusion_matrix(y_true, y_pred)
    cm_percentage = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
    
    # Nombres
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=classes, yticklabels=classes, ax=ax1)
    ax1.set_title(f'{title} - Matrice de Confusion (Nombres)')
    ax1.set_xlabel('Prédit')
    ax1.set_ylabel('Réel')
    
    # Pourcentages
    sns.heatmap(cm_percentage, annot=True, fmt='.1f', cmap='Oranges', 
                xticklabels=classes, yticklabels=classes, ax=ax2)
    ax2.set_title(f'{title} - Matrice de Confusion (Pourcentages)')
    ax2.set_xlabel('Prédit')
    ax2.set_ylabel('Réel')
    
    plt.tight_layout()
    plt.show()

def plot_metrics_comparison(results_dict, title):
    """Tracer la comparaison complète des métriques"""
    models = list(results_dict.keys())
    metrics = ['accuracy', 'precision', 'recall', 'f1_weighted', 'f1_macro', 'roc_auc']
    
    fig, axes = plt.subplots(2, 3, figsize=(20, 12))
    axes = axes.flatten()
    
    colors = plt.cm.Set3(np.linspace(0, 1, len(models)))
    
    for i, metric in enumerate(metrics):
        values = [results_dict[model][metric] * 100 for model in models]
        bars = axes[i].bar(models, values, color=colors)
        axes[i].set_title(f'{metric.upper().replace("_", " ")} (%)', fontsize=14, fontweight='bold')
        axes[i].set_ylabel('Pourcentage (%)')
        axes[i].set_ylim(0, 105)
        
        # Ajouter des étiquettes de valeur sur les barres
        for bar, value in zip(bars, values):
            axes[i].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, 
                        f'{value:.1f}%', ha='center', va='bottom', fontweight='bold')
        
        axes[i].tick_params(axis='x', rotation=45)
        axes[i].grid(True, alpha=0.3)
    
    plt.suptitle(f'{title} - Performance Complète des Modèles', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.show()

# ================= RÉGLAGE DES HYPERPARAMÈTRES =================
def tune_hyperparameters(X_train, y_train, clf_name, clf):
    """Réglage des hyperparamètres pour des classificateurs spécifiques"""
    
    param_grids = {
        'RandomForest': {
            'n_estimators': [200, 300],
            'max_depth': [10, 15, None],
            'min_samples_split': [2, 3]
        },
        'GradientBoosting': {
            'n_estimators': [200, 300],
            'learning_rate': [0.05, 0.08, 0.1],
            'max_depth': [6, 8]
        },
        'SVM': {
            'C': [1, 10, 100],
            'kernel': ['rbf'],
            'gamma': ['scale', 'auto']
        },
        'LogisticRegression': {
            'C': [0.1, 1, 10],
            'solver': ['lbfgs', 'liblinear']
        }
    }
    
    if clf_name not in param_grids:
        return clf
    
    print(f"🔧 Réglage de {clf_name}...")
    
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    grid_search = GridSearchCV(
        clf, param_grids[clf_name], 
        cv=cv, scoring='f1_weighted', 
        n_jobs=-1, verbose=0
    )
    
    grid_search.fit(X_train, y_train)
    print(f"✅ Meilleurs paramètres : {grid_search.best_params_}")
    
    return grid_search.best_estimator_

# ================= ÉVALUATION PRINCIPALE =================
def evaluate_gender_age():
    """Fonction d'évaluation principale pour le genre et l'âge"""
    
    all_results = {}
    
    for label in target_labels:
        print(f"\n{'='*70}")
        print(f"🎯 ÉVALUATION : {label.upper()}")
        print(f"{'='*70}")
        
        # Préparer la cible
        y = df[label].dropna()
        X_filtered = X.loc[y.index]
        
        # Encoder les étiquettes
        le = LabelEncoder()
        y_encoded = le.fit_transform(y.astype(str))
        
        print(f"📊 Classes dans {label} :")
        class_counts = pd.Series(y).value_counts()
        for cls, count in class_counts.items():
            percentage = (count / len(y)) * 100
            print(f"   - {cls} : {count} échantillons ({percentage:.1f}%)")
        
        # Tracer la distribution
        plot_class_distribution(y_encoded, f'{label.capitalize()}', le)
        
        # Division entraînement-test
        min_class_count = min(pd.Series(y_encoded).value_counts())
        stratify = y_encoded if min_class_count >= 2 else None
        
        X_train, X_test, y_train, y_test = train_test_split(
            X_filtered, y_encoded, test_size=0.25, random_state=42, stratify=stratify
        )
        
        print(f"📈 Entraînement : {len(X_train)}, Test : {len(X_test)}")
        
        # Normaliser les caractéristiques
        scaler = RobustScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Trouver la meilleure stratégie d'échantillonnage
        best_sampling = None
        best_score = 0
        
        print("\n🔍 Test des stratégies d'échantillonnage...")
        for sampling_name, sampler in SAMPLING_STRATEGIES.items():
            try:
                X_resampled, y_resampled = sampler.fit_resample(X_train_scaled, y_train)
                rf = RandomForestClassifier(n_estimators=50, random_state=42)
                scores = cross_val_score(rf, X_resampled, y_resampled, cv=3, scoring='f1_weighted')
                avg_score = scores.mean()
                print(f"   - {sampling_name} : {avg_score:.3f}")
                
                if avg_score > best_score:
                    best_score = avg_score
                    best_sampling = sampling_name
            except Exception as e:
                print(f"   - {sampling_name} : Échec")
        
        # Appliquer le meilleur échantillonnage
        if best_sampling:
            print(f"✅ Utilisation : {best_sampling}")
            sampler = SAMPLING_STRATEGIES[best_sampling]
            X_train_final, y_train_final = sampler.fit_resample(X_train_scaled, y_train)
        else:
            X_train_final, y_train_final = X_train_scaled, y_train
            print("⚠️ Aucun échantillonnage appliqué")
        
        # Évaluer tous les classificateurs
        results = {}
        predictions = {}
        
        print(f"\n🚀 Évaluation des classificateurs...")
        
        for clf_name, clf in CLASSIFIERS.items():
            print(f"\n--- {clf_name} ---")
            
            try:
                # Réglage des hyperparamètres
                tuned_clf = tune_hyperparameters(X_train_final, y_train_final, clf_name, clf)
                
                # Évaluer
                metrics, y_pred, y_pred_proba = evaluate_classifier_detailed(
                    X_train_final, X_test_scaled, y_train_final, y_test, tuned_clf, label
                )
                
                results[clf_name] = metrics
                predictions[clf_name] = (y_pred, y_pred_proba)
                
                # Afficher les résultats
                print(f"Précision :    {metrics['accuracy']*100:.2f}%")
                print(f"Précision :   {metrics['precision']*100:.2f}%")
                print(f"Rappel :      {metrics['recall']*100:.2f}%")
                print(f"F1-Pondéré : {metrics['f1_weighted']*100:.2f}%")
                print(f"F1-Macro :    {metrics['f1_macro']*100:.2f}%")
                print(f"ROC-AUC :     {metrics['roc_auc']*100:.2f}%")
                
                # Validation croisée
                cv_scores = cross_val_score(tuned_clf, X_train_final, y_train_final, 
                                          cv=5, scoring='f1_weighted')
                print(f"CV F1 :       {cv_scores.mean()*100:.2f}% ± {cv_scores.std()*100:.2f}%")
                
            except Exception as e:
                print(f"❌ Erreur : {e}")
                results[clf_name] = {
                    'accuracy': 0, 'precision': 0, 'recall': 0, 
                    'f1_weighted': 0, 'f1_macro': 0, 'roc_auc': 0
                }
        
        # Trouver le meilleur modèle
        best_f1 = max(results.values(), key=lambda x: x['f1_weighted'])['f1_weighted']
        best_models = [name for name, metrics in results.items() 
                      if metrics['f1_weighted'] == best_f1]
        
        print(f"\n🏆 MEILLEUR MODÈLE : {', '.join(best_models)}")
        print(f"🎯 Meilleur F1-Score : {best_f1*100:.2f}%")
        
        # Analyse détaillée pour le meilleur modèle
        best_model = best_models[0]
        best_y_pred, best_y_pred_proba = predictions[best_model]
        
        print(f"\n📋 Rapport de Classification ({best_model}) :")
        report = classification_report(y_test, best_y_pred, 
                                     target_names=le.classes_, zero_division=0)
        print(report)
        
        # Visualisations
        plot_confusion_matrix_enhanced(y_test, best_y_pred, le.classes_, 
                                     f'{label.capitalize()} - {best_model}')
        
        plot_metrics_comparison(results, f'{label.capitalize()}')
        
        # Importance des caractéristiques
        if best_model in ['RandomForest', 'GradientBoosting', 'AdaBoost']:
            tuned_clf = tune_hyperparameters(X_train_final, y_train_final, best_model, 
                                           CLASSIFIERS[best_model])
            tuned_clf.fit(X_train_final, y_train_final)
            
            if hasattr(tuned_clf, 'feature_importances_'):
                importance_df = pd.DataFrame({
                    'feature': X.columns,
                    'importance': tuned_clf.feature_importances_
                }).sort_values('importance', ascending=False)
                
                print(f"\n🔍 Top 15 des Caractéristiques Importantes ({best_model}) :")
                for _, row in importance_df.head(15).iterrows():
                    print(f"   {row['feature']} : {row['importance']:.4f}")
                
                # Tracer l'importance des caractéristiques
                plt.figure(figsize=(12, 10))
                top_features = importance_df.head(20)
                sns.barplot(data=top_features, y='feature', x='importance', palette='viridis')
                plt.title(f'{label.capitalize()} - Top 20 Importance des Caractéristiques ({best_model})')
                plt.xlabel('Score d\'Importance')
                plt.tight_layout()
                plt.show()
        
        # Stocker les résultats
        all_results[label] = {
            'best_model': best_model,
            'best_score': best_f1,
            'all_results': results,
            'class_distribution': class_counts.to_dict(),
            'sampling_strategy': best_sampling
        }
    
    return all_results

# ================= RAPPORT RÉCAPITULATIF =================
def generate_final_report(all_results):
    """Générer un rapport final complet"""
    
    print(f"\n{'='*80}")
    print(f"📊 CLASSIFICATION GENRE & ÂGE - RAPPORT FINAL")
    print(f"{'='*80}")
    
    summary_data = []
    
    for label, results in all_results.items():
        summary_data.append({
            'Cible': label.capitalize(),
            'Meilleur Modèle': results['best_model'],
            'F1-Score': f"{results['best_score']*100:.2f}%",
            'Classes': len(results['class_distribution']),
            'Échantillonnage': results['sampling_strategy'] or 'Aucun'
        })
        
        print(f"\n🎯 RÉSULTATS {label.upper()} :")
        print(f"   - Meilleur Modèle : {results['best_model']}")
        print(f"   - F1-Score : {results['best_score']*100:.2f}%")
        print(f"   - Classes : {len(results['class_distribution'])}")
        print(f"   - Échantillonnage : {results['sampling_strategy'] or 'Aucun'}")
        
        # Performance par classe
        print(f"   - Distribution des Classes :")
        for class_name, count in results['class_distribution'].items():
            total = sum(results['class_distribution'].values())
            percentage = (count / total) * 100
            print(f"     * {class_name} : {count} ({percentage:.1f}%)")
        
        # Comparaison des modèles
        print(f"   - Performance de Tous les Modèles :")
        sorted_models = sorted(results['all_results'].items(), 
                             key=lambda x: x[1]['f1_weighted'], reverse=True)
        for model_name, metrics in sorted_models:
            print(f"     * {model_name} : {metrics['f1_weighted']*100:.1f}%")
    
    # Créer le tableau récapitulatif
    summary_df = pd.DataFrame(summary_data)
    print(f"\n📋 TABLEAU RÉCAPITULATIF :")
    print("+" + "-"*70 + "+")
    print(f"| {'Cible':<12} | {'Meilleur Modèle':<15} | {'F1-Score':<10} | {'Classes':<7} | {'Échantillonnage':<12} |")
    print("+" + "-"*70 + "+")
    for _, row in summary_df.iterrows():
        print(f"| {row['Cible']:<12} | {row['Meilleur Modèle']:<15} | {row['F1-Score']:<10} | {row['Classes']:<7} | {row['Échantillonnage']:<12} |")
    print("+" + "-"*70 + "+")
    
    # Insights globaux
    print(f"\n💡 INSIGHTS CLÉS :")
    
    # Meilleurs modèles globalement
    all_models = {}
    for results in all_results.values():
        for model, metrics in results['all_results'].items():
            if model not in all_models:
                all_models[model] = []
            all_models[model].append(metrics['f1_weighted'])
    
    avg_performance = {model: np.mean(scores) for model, scores in all_models.items()}
    best_overall = max(avg_performance, key=avg_performance.get)
    
    print(f"   - Meilleur Modèle Global : {best_overall} (F1 Moyen : {avg_performance[best_overall]*100:.1f}%)")
    
    # Comparaison performance Genre vs Âge
    gender_score = all_results.get('gender', {}).get('best_score', 0) * 100
    age_score = all_results.get('age_category', {}).get('best_score', 0) * 100
    
    if gender_score > 0 and age_score > 0:
        if gender_score > age_score:
            print(f"   - La classification du Genre performe mieux que l'Âge ({gender_score:.1f}% vs {age_score:.1f}%)")
        else:
            print(f"   - La classification de l'Âge performe mieux que le Genre ({age_score:.1f}% vs {gender_score:.1f}%)")
        
        diff = abs(gender_score - age_score)
        if diff < 5:
            print(f"   - Différence de performance minimale ({diff:.1f}%)")
        elif diff < 15:
            print(f"   - Différence de performance modérée ({diff:.1f}%)")
        else:
            print(f"   - Différence de performance significative ({diff:.1f}%)")
    
    # Insights sur les caractéristiques
    print(f"\n🔍 INSIGHTS SUR LES CARACTÉRISTIQUES :")
    print(f"   - Total des Caractéristiques Utilisées : {len(available_features)}")
    print(f"   - Catégories de Caractéristiques :")
    print(f"     * Caractéristiques de Base : {len(base_features)} (âge, confiances)")
    print(f"     * Caractéristiques de Genre : {len(gender_features)} (spécifiques au genre)")
    print(f"     * Caractéristiques d'Âge : {len(age_features)} (transformations & indicateurs d'âge)")
    print(f"     * Caractéristiques Géométriques : {len(geometry_features)} (mesures du visage)")
    print(f"     * Caractéristiques d'Interaction : {len(interaction_features)} (caractéristiques croisées)")
    
    # Recommandations
    print(f"\n🚀 RECOMMANDATIONS :")
    
    for label, results in all_results.items():
        score = results['best_score'] * 100
        if score >= 90:
            print(f"   - {label.capitalize()} : Performance excellente ({score:.1f}%) - Prêt pour la production")
        elif score >= 80:
            print(f"   - {label.capitalize()} : Bonne performance ({score:.1f}%) - Envisager l'optimisation")
        elif score >= 70:
            print(f"   - {label.capitalize()} : Performance modérée ({score:.1f}%) - Nécessite amélioration")
        else:
            print(f"   - {label.capitalize()} : Performance faible ({score:.1f}%) - Nécessite travail données/caractéristiques")
    
    # Recommandations qualité des données
    total_samples = len(df)
    if total_samples < 1000:
        print(f"   - Collecter plus de données : {total_samples} échantillons actuels peuvent être insuffisants")
    
    # Recommandations déséquilibre des classes
    for label, results in all_results.items():
        class_dist = results['class_distribution']
        max_class = max(class_dist.values())
        min_class = min(class_dist.values())
        imbalance_ratio = max_class / min_class if min_class > 0 else float('inf')
        
        if imbalance_ratio > 10:
            print(f"   - {label.capitalize()} : Déséquilibre élevé des classes (ratio : {imbalance_ratio:.1f}) - Considérer l'augmentation de données")
        elif imbalance_ratio > 5:
            print(f"   - {label.capitalize()} : Déséquilibre modéré des classes (ratio : {imbalance_ratio:.1f}) - L'échantillonnage a aidé")
    
    return summary_df

# ================= EXÉCUTION PRINCIPALE =================
if __name__ == "__main__":
    print("🚀 Démarrage du Système d'Évaluation Classification Genre & Âge")
    print(f"📊 Dataset : {EXCEL_PATH}")
    print(f"🎯 Étiquettes Cibles : Genre & Catégorie d'Âge")
    print(f"🤖 Classificateurs : {len(CLASSIFIERS)} modèles")
    print(f"⚖️ Stratégies d'Échantillonnage : {len(SAMPLING_STRATEGIES)} méthodes")
    print(f"📈 Caractéristiques : Caractéristiques complètes spécifiques au genre & âge")
    
    # Aperçu des données
    print(f"\n📋 APERÇU DES DONNÉES :")
    print(f"   - Total d'Échantillons : {len(df)}")
    print(f"   - Total de Caractéristiques : {len(available_features)}")
    
    # Distribution du genre
    if 'gender' in df.columns:
        gender_dist = df['gender'].value_counts()
        print(f"   - Distribution du Genre :")
        for gender, count in gender_dist.items():
            percentage = (count / len(df)) * 100
            print(f"     * {gender} : {count} ({percentage:.1f}%)")
    
    # Distribution de l'âge
    if 'age_category' in df.columns:
        age_dist = df['age_category'].value_counts()
        print(f"   - Distribution des Catégories d'Âge :")
        for age_cat, count in age_dist.items():
            percentage = (count / len(df)) * 100
            print(f"     * {age_cat} : {count} ({percentage:.1f}%)")
    
    print(f"\n{'='*60}")
    print("🏁 DÉMARRAGE DE L'ÉVALUATION...")
    print(f"{'='*60}")
    
    # Exécuter l'évaluation
    results = evaluate_gender_age()
    
    # Générer le rapport final
    summary = generate_final_report(results)
    
    print(f"\n✅ ÉVALUATION TERMINÉE AVEC SUCCÈS !")
    print(f"📊 Consultez les visualisations et rapports générés ci-dessus")
    print(f"💾 Données récapitulatives disponibles dans le DataFrame retourné")
    print(f"\n🎉 Merci d'avoir utilisé le Système de Classification Genre & Âge !")