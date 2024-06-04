import pickle
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
import joblib

# 用户输入数据
def user_input_features():
    age = st.number_input("年龄（岁）:", min_value=0, max_value=120)
    weight = st.number_input("体重（kg）:", min_value=0, max_value=200)
    HB = st.number_input("血红蛋白（g/L）:", min_value=80, max_value=200)
    ALB = st.number_input("白蛋白（g/L）:", min_value=0, max_value=60)
    Operationtime = st.number_input("预计手术时间（分钟）:", min_value=1, max_value=1500)
    bloodloss = st.selectbox("预计失血量:", ("小于200ml", "200-500ml", "500-1000ml", "大于1000ml"))
    ASA = st.selectbox("ASA分级:", ("I", "II", "III", "IV", "V"))
    options_dict = {"小于200ml": 0, "200-500ml": 1, "500-1000ml": 2, "大于1000ml": 3}
    options_dict1 = {"I": 0, "II": 0, "III": 1, "IV": 1, "V": 1}
    value = options_dict[bloodloss]
    value1 = options_dict1[ASA]
    data = {'age': age, 'HB': HB, 'ALB': ALB, 'Operation-time': Operationtime,
            'weight': weight, 'ASA': value1, 'blood-loss': value,}
    features = pd.DataFrame(data, index=[0])
    return features


df1 = user_input_features()

# 数据标准化
scaler = joblib.load('app/scaler.joblib')
gbc = joblib.load('app/model.joblib')
df3 = scaler.transform(df1)
df2 = pd.DataFrame(df3,index=[0])
df2.columns = df1.columns
df2.iloc[0, 5] = df1.iloc[0, 5]
df2.iloc[0, 6] = df1.iloc[0, 6]
xname = df2.columns
# 数据解释
explainer = joblib.load('app/shap.joblib')
shap_values = explainer.shap_values(df2)
shap_values1 = explainer(df2)
shap_values2 = explainer(df1)
shap_values2.data, shap_values1.data = shap_values1.data, shap_values2.data


if st.button("确认"):
    a = shap_values[0].sum()-0.034
    if a >= 1:
        test = ('你评估的患者需要进行输血')
    elif a <= 0:
        test = ('你评估的患者不需要进行输血')
    else:
        b = "{:.2%}".format(a)
        test = ('你评估的患者需要进行输血的几率为：'+b)
    st.markdown(f"""
    <div style='text-align: center; font-size: 24px; line-height: 1.6;'>
        {test}
    </div>
    """, unsafe_allow_html=True)
    # shap.initjs()
    # fig = shap.force_plot(explainer.expected_value, shap_values[0], df2[xname].iloc[0])
    fig = shap.plots.waterfall(shap_values1[0])
    st.pyplot(fig)





