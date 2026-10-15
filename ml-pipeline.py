"""
ONE PIPELINE FOR EVERYTHING
Edit only the CONFIG block (CSV, TARGET, TASK, MODEL, GRID). Run the file.
Preprocessing, PCA (optional), CV, grid search and evaluation all live in one sklearn Pipeline,
so nothing leaks between train and validation folds.
"""
import time, warnings
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
from sklearn.compose import ColumnTransformer, make_column_selector as selector
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.model_selection import (train_test_split, StratifiedKFold, KFold,
                                     GridSearchCV, cross_validate)
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             confusion_matrix, ConfusionMatrixDisplay, classification_report,
                             roc_curve, roc_auc_score, mean_squared_error,
                             mean_absolute_error, r2_score)
warnings.filterwarnings("ignore")

# =====================================================================
# CONFIG  (the only part you change)
# =====================================================================
CSV       = "data.csv"
TARGET    = "target"
TASK      = "clf"            # "clf" or "reg"
DROP_COLS = []               # e.g. ["ID", "Name"]
USE_PCA   = False            # True -> PCA(95% variance) after preprocessing
TEST_SIZE = 0.2
CV_FOLDS  = 5
RS        = 42

# ---- pick ONE model (uncomment the one you need) ----------------------
from sklearn.svm import SVC
MODEL = SVC(probability=True, random_state=RS)
GRID  = {"C": [0.1, 1, 10], "kernel": ["linear", "rbf"], "gamma": ["scale", 0.01]}

# from sklearn.linear_model import LogisticRegression
# MODEL = LogisticRegression(max_iter=2000); GRID = {"C": [0.1, 1, 10]}

# from sklearn.tree import DecisionTreeClassifier
# MODEL = DecisionTreeClassifier(random_state=RS)
# GRID  = {"criterion": ["gini", "entropy"], "max_depth": [3, 5, 10, None], "min_samples_leaf": [1, 2, 4]}

# from sklearn.ensemble import RandomForestClassifier
# MODEL = RandomForestClassifier(random_state=RS)
# GRID  = {"n_estimators": [50, 100, 200], "max_depth": [3, 5, None], "max_features": ["sqrt", "log2"]}

# from sklearn.ensemble import GradientBoostingClassifier
# MODEL = GradientBoostingClassifier(random_state=RS)
# GRID  = {"n_estimators": [50, 100, 200], "learning_rate": [0.01, 0.1, 0.5], "max_depth": [1, 3]}

# from sklearn.ensemble import AdaBoostClassifier
# MODEL = AdaBoostClassifier(random_state=RS)
# GRID  = {"n_estimators": [50, 100, 200], "learning_rate": [0.01, 0.1, 1]}

# from sklearn.ensemble import BaggingClassifier
# MODEL = BaggingClassifier(random_state=RS)
# GRID  = {"n_estimators": [10, 50, 100], "max_samples": [0.5, 1.0], "max_features": [0.5, 1.0]}

# from sklearn.neural_network import MLPClassifier
# MODEL = MLPClassifier(max_iter=300, early_stopping=True, random_state=RS)
# GRID  = {"hidden_layer_sizes": [(64,), (128, 64)], "activation": ["relu", "tanh"], "learning_rate_init": [0.001, 0.01]}

# ---- regression models ----
# from sklearn.linear_model import LinearRegression;  MODEL = LinearRegression(); GRID = {}
# from sklearn.linear_model import Ridge;             MODEL = Ridge();             GRID = {"alpha": [0.01, 0.1, 1, 10, 100]}
# from sklearn.linear_model import Lasso;             MODEL = Lasso(max_iter=10000); GRID = {"alpha": [0.001, 0.01, 0.1, 1]}
# from sklearn.linear_model import ElasticNet;        MODEL = ElasticNet(max_iter=10000); GRID = {"alpha": [0.01, 0.1, 1], "l1_ratio": [0.2, 0.5, 0.8]}
# from sklearn.ensemble import RandomForestRegressor; MODEL = RandomForestRegressor(random_state=RS); GRID = {"n_estimators": [50, 100], "max_depth": [5, None]}

# =====================================================================
# 1. LOAD
# =====================================================================
df = pd.read_csv(CSV).drop(columns=DROP_COLS, errors="ignore").dropna(subset=[TARGET])
print(df.head(), "\nShape:", df.shape, "\nMissing:\n", df.isnull().sum())

X = df.drop(columns=TARGET)
y = df[TARGET]
if TASK == "clf" and not pd.api.types.is_numeric_dtype(y):
    le = LabelEncoder(); y = pd.Series(le.fit_transform(y))       # encode target labels

# =====================================================================
# 2. PIPELINE  (numeric + categorical transformers -> [PCA] -> model)
# =====================================================================
prep = ColumnTransformer([
    ("num", Pipeline([("imp", SimpleImputer(strategy="median")),
                      ("sc",  StandardScaler())]),            selector(dtype_include=np.number)),
    ("cat", Pipeline([("imp", SimpleImputer(strategy="most_frequent")),
                      ("oh",  OneHotEncoder(handle_unknown="ignore", sparse_output=False))]),
                                                              selector(dtype_exclude=np.number)),
])
steps = [("prep", prep)]

if USE_PCA: steps.append(("pca", PCA(0.95)))

steps.append(("model", MODEL))

pipe = Pipeline(steps)

# =====================================================================
# 3. SPLIT + CV SETUP
# =====================================================================
X_tr, X_te, y_tr, y_te = train_test_split(
    X, y, test_size=TEST_SIZE, random_state=RS, stratify=y if TASK == "clf" else None)

cv = (StratifiedKFold(CV_FOLDS, shuffle=True, random_state=RS) if TASK == "clf"
      
      else KFold(CV_FOLDS, shuffle=True, random_state=RS))

scoring = ({"accuracy": "accuracy", "precision": "precision_weighted",
            "recall": "recall_weighted", "f1": "f1_weighted"} if TASK == "clf"
            
           else {"r2": "r2", "mae": "neg_mean_absolute_error", "mse": "neg_mean_squared_error"})

refit_metric = "accuracy" if TASK == "clf" else "r2"

# =====================================================================
# 4. TRAIN  (grid search with CV if GRID given, else plain fit)
# =====================================================================
t = time.time()
if GRID:
    gs = GridSearchCV(pipe, {f"model__{k}": v for k, v in GRID.items()},
                      cv=cv, scoring=scoring, refit=refit_metric, n_jobs=-1).fit(X_tr, y_tr)
    best = gs.best_estimator_
    print("\nBest params:", gs.best_params_, "| Best CV", refit_metric, round(gs.best_score_, 4))
    pd.DataFrame(gs.cv_results_).sort_values(f"rank_test_{refit_metric}") \
        .filter(regex="param_|mean_test_").head(10).pipe(print)       # top-10 table
else:
    best = pipe.fit(X_tr, y_tr)
print(f"Train time: {time.time()-t:.3f}s")

# =====================================================================
# 5. K-FOLD CV REPORT (on the best pipeline, train set)
# =====================================================================
cvr = cross_validate(best, X_tr, y_tr, cv=cv, scoring=scoring)
print("\n%d-fold CV (mean +- std):" % CV_FOLDS)
for m in scoring:
    s = cvr[f"test_{m}"]; s = -s if m in ("mae", "mse") else s      # un-negate error metrics
    print(f"  {m:10s} folds={s.round(3)}  mean={s.mean():.4f} std={s.std():.4f}")

# =====================================================================
# 6. TEST EVALUATION + PLOTS
# =====================================================================
t = time.time(); p = best.predict(X_te); print(f"\nPredict time: {time.time()-t:.4f}s")
if TASK == "clf":
    avg = "binary" if y_te.nunique() == 2 else "weighted"
    print("Train acc:", round(best.score(X_tr, y_tr), 4), "| Test acc:", round(accuracy_score(y_te, p), 4))
    print("Precision:", round(precision_score(y_te, p, average=avg), 4),
          "Recall:", round(recall_score(y_te, p, average=avg), 4),
          "F1:", round(f1_score(y_te, p, average=avg), 4))
    print(classification_report(y_te, p))
    ConfusionMatrixDisplay(confusion_matrix(y_te, p)).plot(cmap="Blues"); plt.title("Confusion Matrix"); plt.show()
    if y_te.nunique() == 2:                                         # ROC only for binary
        s = best.predict_proba(X_te)[:, 1] if hasattr(best, "predict_proba") else best.decision_function(X_te)
        fpr, tpr, _ = roc_curve(y_te, s)
        plt.plot(fpr, tpr, label=f"AUC={roc_auc_score(y_te, s):.3f}"); plt.plot([0, 1], [0, 1], "k--")
        plt.xlabel("FPR"); plt.ylabel("TPR"); plt.title("ROC"); plt.legend(); plt.show()
else:
    mse = mean_squared_error(y_te, p)
    print(f"Train R2: {best.score(X_tr, y_tr):.4f} | Test: MAE={mean_absolute_error(y_te,p):.4f} "
          f"MSE={mse:.4f} RMSE={np.sqrt(mse):.4f} R2={r2_score(y_te,p):.4f}")
    plt.scatter(y_te, p); plt.plot(y_te, y_te, "r"); plt.xlabel("Actual"); plt.ylabel("Predicted"); plt.show()
    plt.hist(y_te - p, bins=30); plt.title("Residuals"); plt.show()