import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import RobustScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, classification_report
from sklearn.feature_selection import SelectKBest, f_classif, VarianceThreshold
from imblearn.over_sampling import ADASYN
import warnings
warnings.filterwarnings('ignore')

# ===== CHARGEMENT DES DONNÉES =====
file_path = r"C:\Users\ADmiN\Desktop\video_project\dataset_merged.xlsx"
df = pd.read_excel(file_path)

# ===== NETTOYAGE DES DONNÉES =====
df_clean = df.dropna(subset=['severity']).copy()

print("Distribution des classes d'origine:")
print(df_clean['severity'].value_counts())

exclude_cols = ['filename','line_number','text','label','bias_type','severity','video_id','clip_name','clip_text','video','chunk']
numeric_features = df_clean.select_dtypes(include=['float64','int64']).columns
feature_cols = [col for col in numeric_features if col not in exclude_cols]

X = df_clean[feature_cols].copy()

# Remplissage des valeurs manquantes par la médiane
for col in X.columns:
    if X[col].isnull().sum() > 0:
        X[col] = X[col].fillna(X[col].median())

# Filtrage par variance
variance_selector = VarianceThreshold(threshold=0.01)
X_variance_filtered = variance_selector.fit_transform(X)
selected_features = X.columns[variance_selector.get_support()]
X = pd.DataFrame(X_variance_filtered, columns=selected_features, index=X.index)

print(f"Features après filtrage par variance: {X.shape[1]}")

# Binarisation de la cible
y = df_clean['severity'].apply(lambda x: 1 if x in ['HIGH', 'MEDIUM', 'UNKNOWN'] else 0)

print("\nDistribution de la cible:")
print(f"Classe 0 (LOW/POSITIVE): {(y==0).sum()}")
print(f"Classe 1 (HIGH/MEDIUM/UNKNOWN): {(y==1).sum()}")

# ===== DIVISION =====
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# RobustScaler
scaler = RobustScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Sélection des K meilleures features
k_best = SelectKBest(score_func=f_classif, k=min(20, X_train_scaled.shape[1]))
X_train_selected = k_best.fit_transform(X_train_scaled, y_train)
X_test_selected = k_best.transform(X_test_scaled)

print(f"Features après sélection: {X_train_selected.shape[1]}")

# ===== ÉQUILIBRAGE ADASYN =====
adasyn = ADASYN(random_state=42, sampling_strategy='auto', n_neighbors=5)
X_train_balanced, y_train_balanced = adasyn.fit_resample(X_train_selected, y_train)

print("\nAprès ADASYN:")
print(f"Classe 0: {(y_train_balanced==0).sum()}")
print(f"Classe 1: {(y_train_balanced==1).sum()}")

# ===== MODÈLES =====
def get_best_models():
    models = {}

    # Random Forest
    rf_params = {
        'n_estimators': [200],
        'max_depth': [10, 20, None],
        'min_samples_split': [2, 5],
        'min_samples_leaf': [1, 2],
        'class_weight': ['balanced']
    }
    rf_grid = GridSearchCV(
        RandomForestClassifier(random_state=42),
        rf_params,
        cv=3,
        scoring='f1',
        n_jobs=-1
    )
    rf_grid.fit(X_train_balanced, y_train_balanced)
    models['RandomForest'] = rf_grid.best_estimator_

    # Gradient Boosting
    gb_params = {
        'n_estimators': [100, 200],
        'learning_rate': [0.01, 0.05, 0.1],
        'max_depth': [3, 5, 7]
    }
    gb_grid = GridSearchCV(
        GradientBoostingClassifier(random_state=42),
        gb_params,
        cv=3,
        scoring='f1',
        n_jobs=-1
    )
    gb_grid.fit(X_train_balanced, y_train_balanced)
    models['GradientBoosting'] = gb_grid.best_estimator_

    # Régression Logistique
    lr_params = {
        'C': [0.1, 1, 10],
        'penalty': ['l1', 'l2'],
        'solver': ['liblinear', 'saga'],
        'class_weight': ['balanced']
    }
    lr_grid = GridSearchCV(
        LogisticRegression(max_iter=1000, random_state=42),
        lr_params,
        cv=3,
        scoring='f1',
        n_jobs=-1
    )
    lr_grid.fit(X_train_balanced, y_train_balanced)
    models['LogisticRegression'] = lr_grid.best_estimator_

    # SVM
    svm_params = {
        'C': [0.1, 1, 10],
        'kernel': ['rbf', 'linear'],
        'class_weight': ['balanced']
    }
    svm_grid = GridSearchCV(
        SVC(random_state=42, probability=True),
        svm_params,
        cv=3,
        scoring='f1',
        n_jobs=-1
    )
    svm_grid.fit(X_train_balanced, y_train_balanced)
    models['SVM'] = svm_grid.best_estimator_

    return models

print("Entraînement des modèles optimisés...")
models = get_best_models()

# Ensemble Voting
voting_clf = VotingClassifier(
    estimators=[
        ('rf', models['RandomForest']),
        ('gb', models['GradientBoosting']),
        ('lr', models['LogisticRegression'])
    ],
    voting='soft'
)
voting_clf.fit(X_train_balanced, y_train_balanced)
models['VotingClassifier'] = voting_clf

# ===== ÉVALUATION =====
def evaluate_model(name, model, X_train, X_test, y_train, y_test):
    cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='f1')
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:,1] if hasattr(model, "predict_proba") else y_pred

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc_score = roc_auc_score(y_test, y_pred_proba)

    print(f"\n===== {name} =====")
    print(f"CV F1-score   = {cv_scores.mean()*100:.1f}% ± {cv_scores.std()*100:.1f}%")
    print(f"Test Accuracy = {accuracy*100:.1f}%")
    print(f"Test Precision= {precision*100:.1f}%")
    print(f"Test Recall   = {recall*100:.1f}%")
    print(f"Test F1-score = {f1*100:.1f}%")
    print(f"Test AUC      = {auc_score*100:.1f}%")

    return f1, cv_scores.mean()

best_f1 = 0
best_model_name = None
best_cv_f1 = 0

results = {}
for name, model in models.items():
    test_f1, cv_f1 = evaluate_model(name, model, X_train_balanced, X_test_selected, y_train_balanced, y_test)
    results[name] = {'test_f1': test_f1, 'cv_f1': cv_f1}
    if test_f1 > best_f1:
        best_f1 = test_f1
        best_model_name = name
        best_cv_f1 = cv_f1

print(f"\n>>> MEILLEUR MODÈLE: {best_model_name}")
print(f">>> CV F1-score: {best_cv_f1*100:.1f}%")
print(f">>> Test F1-score: {best_f1*100:.1f}%")

# Importance des features
if hasattr(models[best_model_name], 'feature_importances_'):
    selected_feature_names = X.columns[k_best.get_support()]
    importances = models[best_model_name].feature_importances_
    feature_importance_df = pd.DataFrame({
        'feature': selected_feature_names[:len(importances)],
        'importance': importances
    }).sort_values('importance', ascending=False)

    print(f"\n===== TOP 10 FEATURES POUR {best_model_name} =====")
    print(feature_importance_df.head(10))

# Rapport de classification
best_model = models[best_model_name]
y_pred_final = best_model.predict(X_test_selected)
print(f"\n===== RAPPORT DE CLASSIFICATION DÉTAILLÉ POUR {best_model_name} =====")
print(classification_report(y_test, y_pred_final, target_names=['LOW/POSITIVE', 'HIGH/MEDIUM/UNKNOWN']))