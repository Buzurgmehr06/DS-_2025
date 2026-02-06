import streamlit as st
import pandas as pd
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import roc_auc_score, accuracy_score
import matplotlib.pyplot as plt


# -------------------------------
# Загрузка и подготовка данных
# -------------------------------
@st.cache_data
def load_data():
    data = load_breast_cancer()
    X = pd.DataFrame(data.data, columns=data.feature_names)
    y = pd.Series(data.target)
    return X, y


# -------------------------------
# Обучение модели
# -------------------------------
@st.cache_resource
def train_model(X, y, n_neighbors, weights):
    model = KNeighborsClassifier(
        n_neighbors=n_neighbors,
        weights=weights
    )
    model.fit(X, y)
    return model


# -------------------------------
# Интерфейс
# -------------------------------
st.title("Прогноз рака молочной железы (KNN)")
st.write("Streamlit-приложение на основе модели из ДЗ-2")

X, y = load_data()

# Боковая панель — параметры модели
st.sidebar.header("Параметры модели")

n_neighbors = st.sidebar.slider(
    "Количество соседей (n_neighbors)",
    min_value=1,
    max_value=20,
    value=5
)

weights = st.sidebar.selectbox(
    "Тип весов",
    ["uniform", "distance"]
)

# Обучение модели
model = train_model(X, y, n_neighbors, weights)

# Разделение для метрик
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

model.fit(X_train, y_train)
y_pred = model.predict(X_test)
y_proba = model.predict_proba(X_test)[:, 1]

accuracy = accuracy_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_proba)

st.subheader("Метрики модели")
st.write(f"Accuracy: **{accuracy:.3f}**")
st.write(f"ROC-AUC: **{roc_auc:.3f}**")


# -------------------------------
# Ввод пользовательских данных
# -------------------------------
st.subheader("Ввод параметров пациента")

user_input = {}
for col in X.columns[:5]:  # только первые 5 признаков для простоты
    user_input[col] = st.number_input(
        f"{col}",
        float(X[col].min()),
        float(X[col].max()),
        float(X[col].mean())
    )

if st.button("Сделать прогноз"):
    input_df = pd.DataFrame([user_input])

    # Добавим недостающие признаки средними значениями
    for col in X.columns:
        if col not in input_df.columns:
            input_df[col] = X[col].mean()

    input_df = input_df[X.columns]

    prediction = model.predict(input_df)[0]
    probability = model.predict_proba(input_df)[0][1]

    if prediction == 1:
        st.success(f"Вероятность доброкачественной опухоли: {probability:.2f}")
    else:
        st.error(f"Вероятность злокачественной опухоли: {1 - probability:.2f}")


# -------------------------------
# Визуализация
# -------------------------------
st.subheader("Распределение классов")

fig, ax = plt.subplots()
y.value_counts().plot(kind="bar", ax=ax)
ax.set_xlabel("Класс")
ax.set_ylabel("Количество")
st.pyplot(fig)
