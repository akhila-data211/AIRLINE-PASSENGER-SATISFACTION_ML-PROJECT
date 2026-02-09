import streamlit as st
import pandas as pd
import numpy as np

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# --------------------------------
# PAGE SETTINGS
# --------------------------------
st.set_page_config(page_title="ML Prediction App", layout="centered")
st.title("📊 Machine Learning Prediction App")

# --------------------------------
# LOAD DATA
# --------------------------------
@st.cache_data
def load_data():
    return pd.read_csv("test.csv")

df = load_data()

st.subheader("Dataset Preview")
st.dataframe(df.head())

# --------------------------------
# HANDLE MISSING VALUES
# --------------------------------
df.fillna(method="ffill", inplace=True)

# --------------------------------
# ENCODE CATEGORICAL DATA
# --------------------------------
encoder = LabelEncoder()
for col in df.columns:
    if df[col].dtype == "object":
        df[col] = encoder.fit_transform(df[col])

# --------------------------------
# TARGET SELECTION
# --------------------------------
st.subheader("Target Selection")
target = st.selectbox("Select target column", df.columns)

X = df.drop(target, axis=1)
y = df[target]

# --------------------------------
# TRAIN-TEST SPLIT
# --------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# --------------------------------
# MODEL TRAINING
# --------------------------------
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# --------------------------------
# MODEL EVALUATION
# --------------------------------
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

st.success(f"Model Accuracy: {accuracy * 100:.2f}%")

# --------------------------------
# USER INPUT
# --------------------------------
st.subheader("Make a Prediction")

user_data = []
for col in X.columns:
    value = st.number_input(f"{col}", float(X[col].min()), float(X[col].max()))
    user_data.append(value)

if st.button("Predict"):
    result = model.predict([user_data])
    st.success(f"Prediction Result: {result[0]}")
