import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
import joblib
from pandas.plotting import scatter_matrix
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import StackingClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, rand_score
import os
import pickle
import numpy as np

# Load your dataset
df = pd.read_csv("data/fire.csv")
X = df.drop('Fire Alarm', axis=1)
y = df['Fire Alarm']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

def load_models():
    with open('models/stacking_classifier.pkl', 'rb') as pickle_in: 
        model1 = pickle.load(pickle_in)
    with open('models/knn.pkl', 'rb') as pickle_in: 
        model2 = pickle.load(pickle_in)
    with open('models/kmeans.pkl', 'rb') as pickle_in: 
        model3 = pickle.load(pickle_in)
    with open('models/rf.pkl', 'rb') as pickle_in: 
        model4 = pickle.load(pickle_in)
    with open('models/bagging_classifier.pkl', 'rb') as pickle_in: 
        model5 = pickle.load(pickle_in)

    model6 = tf.keras.models.load_model("models/model_ml6.h5")

    return model1, model2, model3, model4, model5, model6

# Web App
st.title("Веб приложение ML")

# Sidebar for navigation
st.sidebar.title("Навигация")
page = st.sidebar.selectbox("Выберите страницу", ["Разработчик", "Датасет", "Визуализация", "Инференс модели"])

# Page 1: Developer Information
if page == "Разработчик":
    st.title("Информация о разработчике")
    
    col1, col2 = st.columns(2)
    with col1:
        st.header("Контактная информация")
        st.write("ФИО: Лещёв Дмитрий Евгеньеивч")
        st.write("Номер учебной группы: ФИТ-221")
    
    with col2:
        st.header("Фотография")
        st.image("imgs/me.jpg", width=300)  
    
    st.header("Тема РГР")
    st.write("Разработка Web-приложения для инференса моделей ML и анализа данных")

# Page 2: Dataset Information
elif page == "Датасет":
    st.header("Информация о датасете:")
    
    st.subheader("Тематика")
    st.markdown("Датасет посвящен измерениям различных параметров для определения состояния сигнала `fire_alarm`. "
                "В нем содержатся данные о температуре, влажности, концентрации веществ в воздухе, "
                "сырых данных о водороде и этаноле, давлении, концентрации частиц в воздухе и другие измерения.")

    st.subheader("Описание датасета:")
    st.markdown("""
        **Цель и задачи:**
        Главной задачей данного датасета является определение состояния `fire_alarm` на основе различных измерений и параметров.

        **Процесс обработки данных:**
        1. **Обработка пропущенных значений:** Первым шагом была проведена проверка наличия пропущенных значений в датасете. Выявленные пропуски были удалены или заполнены, чтобы обеспечить качественную подготовку данных для дальнейшего анализа.

        2. **Нормализация данных:** Для обеспечения стабильности и сравнимости различных измерений была применена нормализация данных с использованием метода Min-Max Scaling. Этот шаг позволяет привести значения признаков к одному диапазону и избежать проблемы сильного влияния признаков с большими значениями на модели.

        3. **Обработка категориальных признаков:** В данном датасете отсутствуют категориальные признаки, что упрощает процесс обработки данных. В случае наличия категориальных переменных, их требовалось бы обработать с использованием соответствующих методов, таких как кодирование или преобразование.

        4. **Удаление коррелирующих признаков:** Для улучшения эффективности модели и избежания мультиколлинеарности были удалены фичи, имеющие высокую корреляцию между собой.
    """)


    description_features_list = ['Температура в градусах Цельсия', 'Влажность в процентах', 
                                'Концентрация летучих органических соединений в частицах на миллиард (ppb)', 
                                'Концентрация диоксида углерода в частицах на миллион (ppm)', 
                                'Сырые данные о содержании водорода', 
                                'Сырые данные о содержании этанола',
                                'Давление в гектопаскалях (гПа)',
                                'Столбец, указывающий на состояние сигнала пожарной тревоги']
                                
    st.subheader("Основные признаки в датасете:")
    # Вывод названий колонок с использованием блоков Markdown
    for column, description in zip(df.columns, description_features_list):
        st.markdown(f"`{column}` - {df[column].dtype}: {description}")

    st.dataframe(df.head())  # Вывод нескольких первых строк датасета
    st.subheader("Дополнительная информация:")
    st.text(f"Количество строк в датасете: {df.shape[0]}")
    st.text(f"Количество категориальных признаков: 1")
    st.text(f"Количество численных признаков: 7") 

# Page 3: Data Visualizations
elif page == "Визуализация":
    st.header("Визуализация данных")

    # Выбираем численные признаки для визуализации
    numeric_features = df.select_dtypes(include=["float64", "int64"]).columns

    # Тепловая карта корреляции
    st.subheader("Тепловая карта корреляции")
    fig_heatmap, ax_heatmap = plt.subplots(figsize=(14, 11))
    correlation_matrix = df[numeric_features].corr()
    sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", linewidths=.5, ax=ax_heatmap)
    plt.savefig("heatmap.png")
    st.image("heatmap.png")

    # Гистограммы
    st.subheader("Гистограммы")
    for feature in ['Temperature[C]', 'Humidity[%]']:
        fig_hist = plt.figure()
        sns.histplot(data=df, x=feature, kde=True)
        plt.title(f"Гистограмма для {feature}")
        plt.savefig(f"hist_{feature}.png")
        st.image(f"hist_{feature}.png")
    
    # Боксплоты
    st.subheader("Боксплоты")
    fig_boxplot, ax_boxplot = plt.subplots(figsize=(12, 8))
    sns.boxplot(data=df[numeric_features], ax=ax_boxplot)
    plt.title("Боксплоты для числовых признаков")
    plt.savefig("boxplot.png")
    st.image("boxplot.png")

    # Scatter Plot Matrix
    st.subheader("Матрица диаграмм рассеяния")
    scatter_matrix_fig = scatter_matrix(df[numeric_features], figsize=(12, 12), alpha=0.8, diagonal='hist')
    plt.savefig("scatter_matrix.png")
    st.image("scatter_matrix.png")

# Page 4: Model Inference
elif page == "Инференс модели":
    st.title("Предсказания моделей машинного обучения")

    # Виджет для загрузки файла
    uploaded_file = st.file_uploader("Загрузите ваш CSV файл", type="csv")

    # Интерактивный ввод данных, если файл не загружен
    if uploaded_file is None:
        st.subheader("Введите данные для предсказания:")

        # Интерактивные поля для ввода данных
        input_data = {}
        feature_names = ['Temperature[C]', 'Humidity[%]', 'TVOC[ppb]', 'eCO2[ppm]', 'Raw H2', 'Raw Ethanol', 'Pressure[hPa]']
        for feature in feature_names:
            input_data[feature] = st.number_input(f"{feature}", min_value=0.0, value=10.0)

        if st.button('Сделать предсказание'):
            model1, model2, model3, model4, model5, model6 = load_models()
            input_df = pd.DataFrame([input_data])
            st.write("Входные данные:", input_df)

            # Сделать предсказания на тестовых данных
            predictions_ml1 = model1.predict(input_df)
            predictions_ml2 = model2.predict(input_df)
            predictions_ml4 = model4.predict(input_df)
            predictions_ml5 = model5.predict(input_df)
            probabilities_ml6 = model6.predict(input_df)
            predictions_ml6 = np.argmax(probabilities_ml6, axis=1)
            pred6 = "[Yes]" if predictions_ml6==1 else "[No]"

            st.success(f"Предсказанние Stacking Classifier: {predictions_ml1}")
            st.success(f"Предсказанние KNN: {predictions_ml2}")
            st.success(f"Предсказанние RandomForestClassifier: {predictions_ml4}")
            st.success(f"Предсказанние Bagging Classifier: {predictions_ml5}")
            st.success(f"Предсказанние Tensorflow: {pred6}")
    else:
        try:
            with open('models/stacking_classifier.pkl', 'rb') as pickle_in: 
                model1 = pickle.load(pickle_in)
            with open('models/knn.pkl', 'rb') as pickle_in: 
                model2 = pickle.load(pickle_in)
            with open('models/kmeans.pkl', 'rb') as pickle_in: 
                model3 = pickle.load(pickle_in)
            with open('models/rf.pkl', 'rb') as pickle_in: 
                model4 = pickle.load(pickle_in)
            with open('models/bagging_classifier.pkl', 'rb') as pickle_in: 
                model5 = pickle.load(pickle_in)

            model6 = tf.keras.models.load_model("models/model_ml6.h5")

            predictions_ml1 = model1.predict(X_test)
            predictions_ml2 = model2.predict(X_test)
            predictions_ml3 = model3.predict(X_test)
            predictions_ml4 = model4.predict(X_test)
            predictions_ml5 = model5.predict(X_test)
            probabilities_ml6 = model6.predict(X_test)
            predictions_ml6 = np.argmax(probabilities_ml6, axis=1)

            # Оценить результаты
            accuracy_ml1 = accuracy_score(y_test, predictions_ml1)
            rand_score_ml3 = round(rand_score(y_test, predictions_ml3))
            accuracy_ml4 = accuracy_score(y_test, predictions_ml4)
            accuracy_ml5 = accuracy_score(y_test, predictions_ml5)
            accuracy_ml2 = accuracy_score(y_test, predictions_ml2)

            y = df['Fire Alarm'].map({"No": 0, "Yes": 1})
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            accuracy_ml6 = accuracy_score(y_test, predictions_ml6)

            st.success(f"Точность Stacking Classifier: {accuracy_ml1}")
            st.success(f"Точность KNN: {accuracy_ml2}")
            st.success(f"Rand Score Kmeans: {rand_score_ml3}")
            st.success(f"Точность RandomForestClassifier: {accuracy_ml4}")
            st.success(f"Точность Bagging Classifier: {accuracy_ml5}")
            st.success(f"Точность Tensorflow: {accuracy_ml6}")
        except Exception as e:
            st.error(f"Произошла ошибка при чтении файла: {e}")

# Additional interactivity or input components can be added here

