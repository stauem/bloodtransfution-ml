import streamlit as st
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
import shap
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import RocCurveDisplay, roc_curve, auc, accuracy_score
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import confusion_matrix
from imblearn.over_sampling import BorderlineSMOTE
from imblearn.under_sampling import TomekLinks
from joblib import dump

trans1 = 'E:/代码集和互通数据/特征工程后数据集/boruta+Lasso+logsitic R.csv'
t1 = pd.read_csv(trans1)
X1 = t1[t1.columns[:-1]]
y = t1['bloodtransfusion']


def is_continuous(series):
    return pd.api.types.is_numeric_dtype(series) and series.nunique() > 10  # 假设分类变量通常不会有很多唯一值


# 标准化连续变量的函数
def standardize_continuous_features(df):
    # 识别连续变量
    continuous_vars = [var for var in df.columns if is_continuous(df[var])]

    # 对连续变量进行标准化
    scaler = StandardScaler()
    df_scaled = df.copy()
    for var in continuous_vars:
        df_scaled[var] = scaler.fit_transform(df_scaled[[var]])

    return df_scaled


scaler1 = StandardScaler()
scaler1.fit(X1)
dump(scaler1, 'scaler.joblib')
df = pd.DataFrame(t1)
X = standardize_continuous_features(X1)
xname = X.columns
print(xname)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=11509)
smote = BorderlineSMOTE(random_state=11509)
X_res, y_res = smote.fit_resample(X_train, y_train)

tl = TomekLinks(sampling_strategy='all')
X_res_tl, y_res_tl = tl.fit_resample(X_res, y_res)
gbc = GradientBoostingClassifier(learning_rate=0.01, max_depth=5, min_samples_leaf=2, min_samples_split=4,
                                 n_estimators=200)
gbc.fit(X_res_tl, y_res_tl)

# 使用SHAP解释器
explainer = shap.TreeExplainer(gbc)
shap_values = explainer.shap_values(X)
y_1 = explainer.expected_value

dump(gbc, 'model.joblib')

dump(explainer, 'shap.joblib')
