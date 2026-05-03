"""
TASK 3 — ABLATION STUDY (FAST VERSION ~2 minutes)
Uses pre-tuned best params instead of GridSearchCV for speed.
Fixes label alignment by merging on index.
"""
import os, warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from sklearn.naive_bayes       import ComplementNB
from sklearn.tree              import DecisionTreeClassifier
from sklearn.linear_model      import LogisticRegression
from sklearn.neural_network    import MLPClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing     import StandardScaler, LabelEncoder
from sklearn.model_selection   import train_test_split
from sklearn.metrics           import accuracy_score
import re, nltk
nltk.download('stopwords', quiet=True)
from nltk.corpus import stopwords
from nltk.stem   import PorterStemmer

warnings.filterwarnings('ignore')

BASE_DIR     = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR     = os.path.dirname(BASE_DIR)
FEATURES_CSV = os.path.join(ROOT_DIR, "data", "processed", "MASC_Features.csv")
LABELS_CSV   = os.path.join(ROOT_DIR, "Labels.csv")
OUT_DIR      = os.path.join(ROOT_DIR, "ablation_output")
os.makedirs(OUT_DIR, exist_ok=True)

print("[1/5] Loading data...")
df        = pd.read_csv(FEATURES_CSV)
labels_df = pd.read_csv(LABELS_CSV)
print(f"   Features cols: {list(df.columns)}")
print(f"   Labels cols  : {list(labels_df.columns)}")

label_col_in_feat = None
for c in ["label","Label","class","Class","category","Category","screen_type","ScreenType"]:
    if c in df.columns:
        label_col_in_feat = c
        break

if label_col_in_feat:
    print(f"   Label column in features: '{label_col_in_feat}'")
    y_raw = df[label_col_in_feat].values
else:
    lbl_col = None
    for c in ["label","Label","class","Class","category","Category","screen_type","ScreenType"]:
        if c in labels_df.columns:
            lbl_col = c
            break
    if lbl_col is None:
        lbl_col = labels_df.columns[-1]
    print(f"   Label column in Labels.csv: '{lbl_col}'")
    y_raw = labels_df[lbl_col].values[:len(df)]

le = LabelEncoder()
y  = le.fit_transform(y_raw)
print(f"   {len(le.classes_)} classes: {list(le.classes_[:5])} ...")

text_col = None
for c in ["keywords","text","Text","screen_text","content","description","tokens"]:
    if c in df.columns:
        text_col = c
        break
if text_col is None:
    text_col = df.select_dtypes(include="object").columns[0]
print(f"   Text column: '{text_col}'")

texts    = df[text_col].fillna("").astype(str).values
num_cols = [c for c in df.select_dtypes(include=[np.number]).columns
            if c != label_col_in_feat]
X_num    = df[num_cols].fillna(0).values
print(f"   Numeric cols: {len(num_cols)} | Rows: {len(df)}")

(X_num_tr, X_num_te,
 X_txt_tr, X_txt_te,
 y_tr, y_te) = train_test_split(X_num, texts, y, test_size=0.2, random_state=42)
print(f"   Train: {len(y_tr)} | Test: {len(y_te)}")

DOMAIN_SW = set(stopwords.words('english')) | {
    'app','screen','click','tap','button','page','view','item',
    'list','menu','icon','ui','ux','user','interface','mobile'}
stemmer = PorterStemmer()

def stem_texts(arr):
    out=[]
    for t in arr:
        toks=[stemmer.stem(w) for w in re.sub(r'[^a-zA-Z\s]',' ',t.lower()).split()
              if w not in DOMAIN_SW and len(w)>2]
        out.append(' '.join(toks))
    return out

def build_X(Xn_tr, Xn_te, Xt_tr, Xt_te, feat_eng, adv_nlp, scale):
    Xn_tr=Xn_tr.copy().astype(float); Xn_te=Xn_te.copy().astype(float)
    if feat_eng:
        for j in range(Xn_tr.shape[1]):
            q1,q3=np.percentile(Xn_tr[:,j],25),np.percentile(Xn_tr[:,j],75); iqr=q3-q1
            Xn_tr[:,j]=np.clip(Xn_tr[:,j],q1-1.5*iqr,q3+1.5*iqr)
            Xn_te[:,j]=np.clip(Xn_te[:,j],q1-1.5*iqr,q3+1.5*iqr)
        Xn_tr=np.log1p(np.abs(Xn_tr)); Xn_te=np.log1p(np.abs(Xn_te))
    if scale:
        sc=StandardScaler(); Xn_tr=sc.fit_transform(Xn_tr); Xn_te=sc.transform(Xn_te)
    if adv_nlp:
        tv=TfidfVectorizer(max_features=3000,ngram_range=(1,2),stop_words=list(DOMAIN_SW))
        Xt_tr2=tv.fit_transform(stem_texts(Xt_tr)).toarray()
        Xt_te2=tv.transform(stem_texts(Xt_te)).toarray()
    else:
        tv=TfidfVectorizer(max_features=3000)
        Xt_tr2=tv.fit_transform(Xt_tr).toarray(); Xt_te2=tv.transform(Xt_te).toarray()
    return np.hstack([Xn_tr,Xt_tr2]), np.hstack([Xn_te,Xt_te2])

def get_clf(name, tuned):
    if name=="Naive Bayes":
        return ComplementNB(alpha=0.1 if tuned else 1.0)
    if name=="Decision Tree":
        return DecisionTreeClassifier(random_state=42,
            max_depth=10 if tuned else None,
            criterion="entropy" if tuned else "gini",
            min_samples_split=5 if tuned else 2)
    if name=="Logistic Regression":
        return LogisticRegression(random_state=42,max_iter=500,C=1.0,solver="lbfgs")
    if name=="MLP":
        return MLPClassifier(random_state=42,max_iter=300,
            hidden_layer_sizes=(100,50) if tuned else (100,),
            activation="relu",alpha=0.001 if tuned else 0.0001,
            learning_rate="adaptive" if tuned else "constant")

configs=[
    {"name":"Baseline",              "feat_eng":False,"adv_nlp":False,"tuned":False},
    {"name":"+ Feature Engineering", "feat_eng":True, "adv_nlp":False,"tuned":False},
    {"name":"+ Advanced NLP",        "feat_eng":False,"adv_nlp":True, "tuned":False},
    {"name":"+ Tuned Params",        "feat_eng":False,"adv_nlp":False,"tuned":True},
    {"name":"All Combined (Task 3)", "feat_eng":True, "adv_nlp":True, "tuned":True},
]
alg_names=["Naive Bayes","Decision Tree","Logistic Regression","MLP"]
results={c["name"]:{} for c in configs}

print("\n[2/5] Running 20 configurations (no GridSearch = fast)...")
done=0
for cfg in configs:
    Xtr,Xte=build_X(X_num_tr,X_num_te,X_txt_tr,X_txt_te,
                    cfg["feat_eng"],cfg["adv_nlp"],cfg["feat_eng"])
    shift=abs(Xtr.min()) if Xtr.min()<0 else 0
    Xtr_nb,Xte_nb=Xtr+shift,Xte+shift
    for alg in alg_names:
        clf=get_clf(alg,cfg["tuned"])
        Xf=Xtr_nb if alg=="Naive Bayes" else Xtr
        Xp=Xte_nb if alg=="Naive Bayes" else Xte
        clf.fit(Xf,y_tr)
        acc=accuracy_score(y_te,clf.predict(Xp))*100
        results[cfg["name"]][alg]=round(acc,2)
        done+=1
        print(f"   [{done:2d}/20] {cfg['name']:28s}  {alg:22s}  ->  {acc:.2f}%")

print("\n[3/5] Saving CSV...")
rows=[{"Configuration":k,**v} for k,v in results.items()]
rdf=pd.DataFrame(rows)
rdf.to_csv(os.path.join(OUT_DIR,"ablation_results.csv"),index=False)
print(rdf.to_string(index=False))

print("\n[4/5] Generating chart...")
fig,ax=plt.subplots(figsize=(13,6))
cfg_labels=[c["name"] for c in configs]
x=np.arange(len(cfg_labels)); w=0.19
clrs=["#1B2A4A","#0D7377","#2980B9","#E74C3C"]
for i,alg in enumerate(alg_names):
    vals=[results[c["name"]][alg] for c in configs]
    bars=ax.bar(x+(i-1.5)*w,vals,w,label=alg,color=clrs[i],zorder=3)
    for bar,v in zip(bars,vals):
        ax.text(bar.get_x()+bar.get_width()/2,v+0.05,f"{v:.1f}",
                ha='center',va='bottom',fontsize=6.5,fontweight='bold')
ax.set_xticks(x); ax.set_xticklabels(cfg_labels,fontsize=8.5,rotation=10,ha='right')
ax.set_ylabel("Test Accuracy (%)"); ax.yaxis.grid(True,linestyle='--',alpha=0.6,zorder=0)
ax.set_title("Ablation Study - Impact of Each Improvement",fontsize=13,fontweight='bold')
ax.set_facecolor("#FAFBFF"); ax.legend(fontsize=9)
all_vals=[results[c["name"]][a] for c in configs for a in alg_names]
ax.set_ylim(min(all_vals)-3,max(all_vals)+3)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR,"ablation_chart.png"),dpi=150,bbox_inches='tight')
plt.close()

print("\n"+"="*55)
print("ABLATION COMPLETE!")
print(f"  Results -> {OUT_DIR}\\ablation_results.csv")
print(f"  Chart   -> {OUT_DIR}\\ablation_chart.png")
print("="*55)