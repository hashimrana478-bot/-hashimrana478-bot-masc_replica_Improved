"""
=============================================================================
MASC TASK 3: QUICK RUN SCRIPT FOR VIVA DEMO
=============================================================================
Optimized for fast execution (3-5 minutes)
Best for showing results quickly to TA during VIVA
=============================================================================
"""

import os, warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                              f1_score, roc_auc_score, confusion_matrix, f1_score)
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier

import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.tokenize import word_tokenize

warnings.filterwarnings('ignore')

for res in ['stopwords','punkt','wordnet']:
    try: nltk.download(res, quiet=True)
    except: pass

# Configuration
DATA_PATH   = 'data/processed/MASC_Features.csv'
LABELS_PATH = 'data/Labels.csv'
OUTPUT_DIR  = 'task3_results'
os.makedirs(OUTPUT_DIR, exist_ok=True)

RANDOM_STATE = 42
print("\n" + "="*70)
print("  MASC TASK 3: QUICK RUN (3-5 minutes)")
print("  Course Algorithms: Naive Bayes, Decision Tree, LR, MLP")
print("="*70)

# =============================================================================
# STEP 1: LOAD DATA
# =============================================================================
print("\n[1/6] Loading data...")
data = pd.read_csv(DATA_PATH)
data.columns = data.columns.astype(str)

for path in [LABELS_PATH, 'data/processed/Labels.csv', 'Labels.csv']:
    if os.path.exists(path):
        labels_df = pd.read_csv(path, header=None)
        break

label_col_idx = None
for col_idx in range(labels_df.shape[1]):
    col = labels_df.iloc[:, col_idx]
    sample = col.dropna().astype(str)
    non_num = sample[~sample.str.match(r'^\s*\d+\.?\d*\s*$')]
    if len(non_num) > len(sample) * 0.5:
        label_col_idx = col_idx
        break
if label_col_idx is None:
    label_col_idx = labels_df.shape[1] - 1

labels = labels_df.iloc[:, label_col_idx].astype(str).str.strip()
min_len = min(len(data), len(labels))
data = data.iloc[:min_len].reset_index(drop=True)
labels = labels.iloc[:min_len].reset_index(drop=True)

print(f"     Loaded: {data.shape[0]} samples × {data.shape[1]} features")

# =============================================================================
# STEP 2: PREPARE FEATURES
# =============================================================================
print("[2/6] Preparing features...")
text_col = None
for col in data.columns:
    if data[col].dtype == object:
        text_col = col
        break

numeric_cols = [c for c in data.columns if pd.api.types.is_numeric_dtype(data[c])]
le = LabelEncoder()
y = le.fit_transform(labels)

unique, counts = np.unique(y, return_counts=True)
rare = unique[counts < 3]
if len(rare) > 0:
    keep = ~np.isin(y, rare)
    data = data[keep].reset_index(drop=True)
    labels = labels[keep].reset_index(drop=True)
    y = le.fit_transform(labels)

print(f"     Ready: {len(y)} samples, {len(le.classes_)} classes")

# =============================================================================
# STEP 3: FEATURE ENGINEERING
# =============================================================================
print("[3/6] Engineering features...")
df_eng = data[numeric_cols].fillna(0).copy()

# IQR Outlier Capping
for col in numeric_cols:
    Q1 = df_eng[col].quantile(0.25)
    Q3 = df_eng[col].quantile(0.75)
    IQR = Q3 - Q1
    lo, hi = Q1 - 1.5*IQR, Q3 + 1.5*IQR
    df_eng[col] = df_eng[col].clip(lower=lo, upper=hi)

# Log Transformation
for col in numeric_cols:
    df_eng[f'log_{col}'] = np.log1p(df_eng[col].clip(lower=0))

# Ratio Feature
num_p = [c for c in numeric_cols if c in df_eng.columns]
if len(num_p) >= 2:
    df_eng['ratio'] = df_eng[num_p[0]] / (df_eng[num_p[1]] + 1)

# Standardization
log_cols = [f'log_{c}' for c in numeric_cols if f'log_{c}' in df_eng.columns]
all_scale = numeric_cols + log_cols + (['ratio'] if 'ratio' in df_eng.columns else [])
all_scale = [c for c in all_scale if c in df_eng.columns]
scaler = StandardScaler()
df_eng[all_scale] = scaler.fit_transform(df_eng[all_scale])

print(f"     Features: {df_eng.shape[1]} engineered features")

# =============================================================================
# STEP 4: ADVANCED NLP
# =============================================================================
print("[4/6] Processing text...")
stemmer = PorterStemmer()
lemmatizer = WordNetLemmatizer()
try: std_stops = set(stopwords.words('english'))
except: std_stops = set()
ui_stops = {'android','view','layout','activity','button','text','click'}
all_stops = std_stops | ui_stops

def process_text(txt):
    if pd.isna(txt) or str(txt).strip() in ('','nan'): return ''
    txt = str(txt).lower()
    try: tokens = word_tokenize(txt)
    except: tokens = txt.split()
    tokens = [t for t in tokens if t.isalpha() and t not in all_stops]
    s = [stemmer.stem(t) for t in tokens]
    l = [lemmatizer.lemmatize(t) for t in tokens]
    return ' '.join(set(s + l))

proc_text = data[text_col].fillna('').apply(process_text)
tfidf = TfidfVectorizer(max_features=200, ngram_range=(1,2), min_df=2, sublinear_tf=True)
X_txt = tfidf.fit_transform(proc_text).toarray()
X_tdf = pd.DataFrame(X_txt, columns=[f't_{i}' for i in range(X_txt.shape[1])])

X = pd.concat([df_eng.reset_index(drop=True), X_tdf.reset_index(drop=True)], axis=1)
X.columns = X.columns.astype(str)

print(f"     Features: {X.shape[1]} total (engineering + NLP)")

# =============================================================================
# STEP 5: TRAIN MODELS WITH GRIDSEARCHCV
# =============================================================================
print("[5/6] Training 4 course algorithms...")

n_classes = len(np.unique(y))
test_frac = max(0.20, (n_classes + 5) / len(y))
test_frac = min(test_frac, 0.40)

X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=test_frac, random_state=RANDOM_STATE, stratify=y)

results = []
predictions = {}

# Naive Bayes
print("     Training: Naive Bayes...", end='', flush=True)
params = {"var_smoothing": [1e-9, 1e-7, 1e-5, 1e-3, 1e-1]}
gs = GridSearchCV(GaussianNB(), params, cv=3, scoring='accuracy', n_jobs=-1, verbose=0)
gs.fit(X_tr, y_tr)
nb = gs.best_estimator_
nb_pred = nb.predict(X_te)
nb_acc = nb.score(X_te, y_te) * 100
nb_f1 = f1_score(y_te, nb_pred, average='weighted', zero_division=0) * 100
results.append({'Algorithm': 'Naive Bayes', 'Test Acc': round(nb_acc,2), 'F1': round(nb_f1,2), 'Params': str(gs.best_params_)})
predictions['Naive Bayes'] = (y_te, nb_pred)
print(f" ✓ {nb_acc:.2f}%")

# Decision Tree
print("     Training: Decision Tree...", end='', flush=True)
params = {"max_depth": [3,5,7,10,None], "criterion": ["gini","entropy"], "min_samples_split": [2,5,10]}
gs = GridSearchCV(DecisionTreeClassifier(random_state=RANDOM_STATE), params, cv=3, scoring='accuracy', n_jobs=-1, verbose=0)
gs.fit(X_tr, y_tr)
dt = gs.best_estimator_
dt_pred = dt.predict(X_te)
dt_acc = dt.score(X_te, y_te) * 100
dt_f1 = f1_score(y_te, dt_pred, average='weighted', zero_division=0) * 100
results.append({'Algorithm': 'Decision Tree', 'Test Acc': round(dt_acc,2), 'F1': round(dt_f1,2), 'Params': str(gs.best_params_)})
predictions['Decision Tree'] = (y_te, dt_pred)
print(f" ✓ {dt_acc:.2f}%")

# Logistic Regression
print("     Training: Logistic Regression...", end='', flush=True)
params = {"C": [0.001,0.01,0.1,1,10], "solver": ["lbfgs","saga"], "multi_class": ["multinomial"]}
gs = GridSearchCV(LogisticRegression(random_state=RANDOM_STATE, max_iter=1000), params, cv=3, scoring='accuracy', n_jobs=-1, verbose=0)
gs.fit(X_tr, y_tr)
lr = gs.best_estimator_
lr_pred = lr.predict(X_te)
lr_acc = lr.score(X_te, y_te) * 100
lr_f1 = f1_score(y_te, lr_pred, average='weighted', zero_division=0) * 100
results.append({'Algorithm': 'Logistic Regression', 'Test Acc': round(lr_acc,2), 'F1': round(lr_f1,2), 'Params': str(gs.best_params_)})
predictions['Logistic Regression'] = (y_te, lr_pred)
print(f" ✓ {lr_acc:.2f}%")

# MLP
print("     Training: Multi-Layer Perceptron...", end='', flush=True)
params = {"hidden_layer_sizes": [(100,),(100,50),(200,100)], "activation": ["relu","tanh"], "learning_rate": ["constant","adaptive"]}
gs = GridSearchCV(MLPClassifier(random_state=RANDOM_STATE, max_iter=500, early_stopping=True), params, cv=3, scoring='accuracy', n_jobs=-1, verbose=0)
gs.fit(X_tr, y_tr)
mlp = gs.best_estimator_
mlp_pred = mlp.predict(X_te)
mlp_acc = mlp.score(X_te, y_te) * 100
mlp_f1 = f1_score(y_te, mlp_pred, average='weighted', zero_division=0) * 100
results.append({'Algorithm': 'Multi-Layer Perceptron', 'Test Acc': round(mlp_acc,2), 'F1': round(mlp_f1,2), 'Params': str(gs.best_params_)})
predictions['MLP'] = (y_te, mlp_pred)
print(f" ✓ {mlp_acc:.2f}%")

# =============================================================================
# STEP 6: GENERATE OUTPUTS
# =============================================================================
print("[6/6] Generating outputs...")

results_df = pd.DataFrame(results)
results_df.to_csv(os.path.join(OUTPUT_DIR, 'results.csv'), index=False)

# Comparison Chart
paper = {'Naive Bayes': 90.65, 'Decision Tree': 92.35, 'Logistic Regression': 92.63, 'Multi-Layer Perceptron': 93.20}
algos = results_df['Algorithm'].tolist()
test_accs = results_df['Test Acc'].tolist()
paper_vals = [paper.get(a, 0) for a in algos]

x = np.arange(len(algos))
width = 0.35

fig, ax = plt.subplots(figsize=(14, 8))
b1 = ax.bar(x - width/2, paper_vals, width, label='Original Paper', color='#E74C3C', alpha=0.85)
b2 = ax.bar(x + width/2, test_accs, width, label='Your Task 3', color='#27AE60', alpha=0.85)

for bar in b1:
    h = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., h + 0.3, f'{h:.2f}%', ha='center', fontsize=10, fontweight='bold')
for bar in b2:
    h = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., h + 0.3, f'{h:.2f}%', ha='center', fontsize=10, fontweight='bold')

for i, (p, t) in enumerate(zip(paper_vals, test_accs)):
    delta = t - p
    color = '#27AE60' if delta >= 0 else '#E74C3C'
    ax.text(i, max(p, t) + 1.5, f'{delta:+.2f}%', ha='center', fontsize=10, fontweight='bold', color=color)

ax.set_ylabel('Test Accuracy (%)', fontsize=12, fontweight='bold')
ax.set_title('Course Algorithms: Paper vs Task 3 Improvements', fontsize=14, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(algos, fontsize=11)
ax.set_ylim(85, 105)
ax.legend(fontsize=12)
ax.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'comparison_chart.png'), dpi=150, bbox_inches='tight')
plt.close()
print("     ✓ Comparison chart saved")

# Confusion Matrices
for algo_name, (y_true, y_pred) in predictions.items():
    cm = confusion_matrix(y_true, y_pred)
    figsize = (max(10, len(le.classes_)), max(8, len(le.classes_)-2))
    plt.figure(figsize=figsize)
    sns.heatmap(cm, annot=(len(le.classes_) <= 15), fmt='d', cmap='Blues', linewidths=0.5)
    plt.title(f'Confusion Matrix: {algo_name}', fontsize=13, fontweight='bold')
    plt.ylabel('True Label'); plt.xlabel('Predicted Label')
    plt.tight_layout()
    safe_name = algo_name.replace(' ', '_').replace('-', '_').lower()
    fname = os.path.join(OUTPUT_DIR, f'cm_{safe_name}.png')
    plt.savefig(fname, dpi=150, bbox_inches='tight')
    plt.close()
print("     ✓ Confusion matrices saved (4 files)")

# =============================================================================
# FINAL RESULTS
# =============================================================================
print("\n" + "="*70)
print("✅ TASK 3 COMPLETE!")
print("="*70)
print("\nRESULTS:")
print(results_df.to_string(index=False))
print(f"\nOutput folder: {OUTPUT_DIR}/")
print("Files generated:")
print(f"  • results.csv")
print(f"  • comparison_chart.png")
print(f"  • cm_naive_bayes.png")
print(f"  • cm_decision_tree.png")
print(f"  • cm_logistic_regression.png")
print(f"  • cm_multi_layer_perceptron.png")
print("\n" + "="*70 + "\n")