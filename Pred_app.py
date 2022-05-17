import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.datasets import load_diabetes, load_boston
import numpy as np
#---------------------------------#
# Макет страницы
## Страница расширяется на всю ширину
st.set_page_config(page_title='Pred_APP',
    layout='wide')

#---------------------------------#
# Построение модели
def build_model(df):
    X = df.iloc[:,:-1] # Используем все столбцы, кроме последнего, как X
    Y = df.iloc[:,-1] # Выбираем последний столбец - Y

    # Разделение данных
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=(100-split_size)/100)
    
    st.markdown('**1.2. Разделение данных**')
    st.write('Обучающий набор')
    st.info(X_train.shape)
    st.write('Тестовый набор')
    st.info(X_test.shape)

    st.markdown('**1.3. Используемые переменные**:')
    st.write('Переменные X')
    st.info(list(X.columns))
    st.write('Переменная Y')
    st.info(Y.name)

    rf = RandomForestRegressor(n_estimators=parameter_n_estimators,
        random_state=parameter_random_state,
        max_features=parameter_max_features,
        criterion=parameter_criterion,
        min_samples_split=parameter_min_samples_split,
        min_samples_leaf=parameter_min_samples_leaf,
        bootstrap=parameter_bootstrap,
        oob_score=parameter_oob_score,
        n_jobs=parameter_n_jobs)
    rf.fit(X_train, Y_train)

    st.subheader('2. Производительность модели')

    st.markdown('**2.1. Обучающий набор**')
    Y_pred_train = rf.predict(X_train)
    st.write('Коэффициент детерминации ($R^2$):')
    st.info(r2_score(Y_train, Y_pred_train))

    st.write('Ошибка (MSE или MAE):')
    st.info(mean_squared_error(Y_train, Y_pred_train))

    st.markdown('**2.2. Тестовый набор**')
    Y_pred_test = rf.predict(X_test)
    st.write('Коэффициент детерминации ($R^2$):')
    st.info(r2_score(Y_test, Y_pred_test))

    st.write('Ошибка (MSE или MAE):')
    st.info(mean_squared_error(Y_test, Y_pred_test))
    
    st.write('График прогноза:')
    a = pd.DataFrame(np.column_stack([Y_pred_test, Y_test]), columns=['Y_pred', 'Y_test']) 
    st.line_chart(a) # Вывод графика с предсказанными и тестовыми Y
    st.write(a) # Таблица Y_pred Y_test
    
   
    st.subheader('3. Параметры модели')
    st.write(rf.get_params())

#---------------------------------#
st.write("""
# Приложение для прогнозирования

В данном приложении используется функция *RandomForestRegressor()* для построения регрессионной модели с использованием алгоритма **Random Forest**.

Попробуйте настроить гиперпараметры!

""")

#---------------------------------#
# Боковая панель - Собирает элементы пользовательского ввода в фрейм данных
with st.sidebar.header('1. Загрузите ваши данные в формате CSV'):
    uploaded_file = st.sidebar.file_uploader("Загрузите входной CSV-файл", type=["csv"])
    st.sidebar.markdown("""
[Пример входного CSV-файла](https://disk.yandex.ru/d/v5GeJJ3hm1YUVQ)
""")

# Боковая панель - Настройки параметров
with st.sidebar.header('2. Установите параметры'):
    split_size = st.sidebar.slider('Разделение данных (% для обучающего набора)', 10, 90, 70, 5)

with st.sidebar.subheader('2.1. Параметры обучения'):
    parameter_n_estimators = st.sidebar.slider('Число деревьев (n_estimators)', 0, 1000, 100, 50)
    parameter_max_features = st.sidebar.select_slider('Число признаков для выбора расщепления (max_features)', options=['auto', 'sqrt', 'log2'])
    parameter_min_samples_split = st.sidebar.slider('Минимальное число объектов, при котором выполняется расщепление (min_samples_split)', 1, 10, 2, 1)
    parameter_min_samples_leaf = st.sidebar.slider('Минимальное число объектов в листьях (min_samples_leaf)', 1, 10, 5, 1)

with st.sidebar.subheader('2.2. Общие параметры'):
    parameter_random_state = st.sidebar.slider('Номер семени (random_state)', 0, 1000, 42, 1)
    parameter_criterion = st.sidebar.select_slider('Показатель производительности (criterion)', options=['mse', 'mae'])
    parameter_bootstrap = st.sidebar.select_slider('Образцы начальной загрузки при построении деревьев (bootstrap)', options=[True, False])
    parameter_oob_score = st.sidebar.select_slider('Следует ли использовать часть обучающего набора для оценки эффективности модели (oob_score)', options=[False, True])
    parameter_n_jobs = st.sidebar.select_slider('Количество используемых процессоров *-1 - строить на максимально возможном числе процессоров (n_jobs)', options=[1, -1])

#---------------------------------#
# Главная панель

# Отображение набора данных
st.subheader('1. Dataset')

# Функция если - Если файл выбран, то загружаем его и строим модель, если нет - строим модель по встроенному набору данных
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.markdown('**1.1. Краткий обзор набора данных**')
    st.write(df)
    st.line_chart(df)
    build_model(df)
else:
    st.info('Ожидание загрузки CSV-файла.')
    if st.button('Нажмите, чтобы использовать пример набора данных'):
        # Diabetes dataset
        #diabetes = load_diabetes()
        #X = pd.DataFrame(diabetes.data, columns=diabetes.feature_names)
        #Y = pd.Series(diabetes.target, name='response')
        #df = pd.concat( [X,Y], axis=1 )

        #st.markdown('The Diabetes dataset is used as the example.')
        #st.write(df.head(5))

        # Boston housing dataset
        boston = load_boston()
        X = pd.DataFrame(boston.data, columns=boston.feature_names)
        Y = pd.Series(boston.target, name='response')
        df = pd.concat( [X,Y], axis=1 )

        st.markdown('В качестве примера используется встроенный набор данных Boston housing - данные о стоимости жилья в Бостоне.')
        st.write(df.head(5))
        st.line_chart(df)
        build_model(df)