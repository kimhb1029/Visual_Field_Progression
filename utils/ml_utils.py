### Import Libraries
import numpy as np              
import pandas as pd             
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    confusion_matrix, accuracy_score, precision_score, f1_score
)
from xgboost import XGBClassifier

### input info
seed = 0

### Make code 
def modi_dataframe(dataframe):
    df = dataframe.copy()
    df['TDVmean'] = df.filter(regex='TDV_').mean(axis=1)
    df['TDVstd'] = df.filter(regex='TDV_').std(axis=1)
    df['PDVmean'] = df.filter(regex='PDV_').mean(axis=1)
    df['PDVstd'] = df.filter(regex='PDV_').std(axis=1)
    return df

def make_test_train(dataframe, seed=seed):
    df = dataframe.copy()
    X = df.copy()
    y = df[['Consensus_label', 'Wiggs_label']].copy()
    X = X.drop(['Consensus_label','Wiggs_label'], axis=1)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=seed
    )

    train_df = pd.concat([X_train,y_train],axis=1)
    test_df = pd.concat([X_test,y_test],axis=1)
    return train_df, test_df


def eval_binary_no_proba(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0,1]).ravel()
    specificity = tn/(tn+fp) if (tn+fp)>0 else np.nan
    sensitivity = tp/(tp+fn) if (tp+fn)>0 else np.nan
    return {
        "Accuracy": round(accuracy_score(y_true, y_pred), 3),
        "Recall": round(sensitivity, 3),
        "Specificity":  round(specificity, 3),
        "Precision":round(precision_score(y_true, y_pred, zero_division=0), 3),
        "F1":       round(f1_score(y_true, y_pred, zero_division=0), 3)

    }

def fit_eval(models, X_train, X_test, y_train_bin, y_test_bin, label_name):
    rows = []
    for name, pipe in models:
        pipe.fit(X_train, y_train_bin)
        y_pred = pipe.predict(X_test)
        met = eval_binary_no_proba(y_test_bin, y_pred)
        rows.append({"Target": label_name, "Model": name, **met})

    return pd.DataFrame(rows)

def get_models(seed):
    return [
        ("SVM", Pipeline([
            ("scaler", StandardScaler()),
            ("clf", SVC(random_state=seed))
        ])),
        ("Random Forest", Pipeline([
            ("scaler", StandardScaler()),  
            ("clf", RandomForestClassifier(random_state=seed))
        ])),
        ("Logistic Regression", Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(random_state=seed))
        ])),
        ("XGBoost", Pipeline([
            ("scaler", StandardScaler()),
            ("clf", XGBClassifier(random_state=seed))
        ])),
    ]


