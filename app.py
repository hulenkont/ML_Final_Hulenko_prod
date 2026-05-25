import streamlit as st
import pandas as pd
import numpy as np
import requests
from geopy.geocoders import Nominatim
from datetime import date, timedelta
import plotly.express as px
import plotly.graph_objects as go
import os

# ML бібліотеки
from sklearn.model_selection import train_test_split, GridSearchCV, TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, mean_absolute_error, r2_score
from sklearn.feature_selection import SelectFromModel

# Налаштування сторінки Streamlit
st.set_page_config(page_title="Прогноз опадів ML", page_icon="☔", layout="wide")

# ДОПОМІЖНІ ФУНКЦІЇ

@st.cache_data
def get_coordinates(city_name):
    # Визначає географічні координати населеного пункту за його текстовою назвою.
    try:
        geolocator = Nominatim(user_agent="weather_ml_app_ua")
        location = geolocator.geocode(city_name)
        if location:
            return location.latitude, location.longitude
        return None, None
    except Exception as e:
        st.error(f"Помилка геокодування: {e}")
        return None, None

@st.cache_data
def fetch_weather_data(lat, lon, start_date, end_date, is_forecast=False):
    # Завантажує метеорологічні дані з API Open-Meteo та автоматично інтерполює можливі пропуски.
    if is_forecast:
        url = "https://api.open-meteo.com/v1/forecast"
    else:
        url = "https://archive-api.open-meteo.com/v1/archive"
        
    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": start_date,
        "end_date": end_date,
        "daily": [
            "temperature_2m_max", "temperature_2m_min", "temperature_2m_mean",
            "apparent_temperature_max", "apparent_temperature_min",
            "wind_speed_10m_max", "wind_gusts_10m_max", "wind_direction_10m_dominant",
            "shortwave_radiation_sum", "sunshine_duration", "daylight_duration",
            "et0_fao_evapotranspiration", "precipitation_sum"
        ],
        "timezone": "auto"
    }
    
    response = requests.get(url, params=params)
    if response.status_code == 200:
        data = response.json()
        if 'daily' in data:
            df = pd.DataFrame(data['daily'])
            df['time'] = pd.to_datetime(df['time'])
            
            # Перевірка та обробка пропусків
            missing_count = df.isnull().sum().sum()
            if missing_count > 0:
                cols_to_interpolate = df.columns.drop('time')
                df[cols_to_interpolate] = df[cols_to_interpolate].apply(pd.to_numeric, errors='coerce')
                df[cols_to_interpolate] = df[cols_to_interpolate].interpolate(method='linear').ffill().bfill()
                
            return df, missing_count
    return None, 0

def engineer_features(df):
    # Здійснює Feature Engineering, виділяючи з дати порядковий номер дня року та місяць.
    df_engineered = df.copy()
    df_engineered['month'] = df_engineered['time'].dt.month
    df_engineered['day_of_year'] = df_engineered['time'].dt.dayofyear
    return df_engineered


# ДЛЯ ІНТЕРФЕЙСУ

st.title("☔ Сервіс для прогнозування опадів")
st.markdown("Проєкт виконав: Гуленко Назар, група ІДС-501.")

# Збереження стану сесії
# Ініціалізує змінні стану сесії Streamlit для збереження даних та навчених моделей між перезавантаженнями.
if 'current_step' not in st.session_state:
    st.session_state.current_step = "🔴 Завантаження даних (Load data)"
if 'historical_df' not in st.session_state:
    st.session_state.historical_df = None
if 'best_model' not in st.session_state:
    st.session_state.best_model = None
if 'best_model_name' not in st.session_state:
    st.session_state.best_model_name = ""
if 'best_regressor' not in st.session_state:
    st.session_state.best_regressor = None
if 'best_regressor_name' not in st.session_state:
    st.session_state.best_regressor_name = ""
if 'feature_cols' not in st.session_state:
    st.session_state.feature_cols = None
if 'lat' not in st.session_state:
    st.session_state.lat = None
if 'lon' not in st.session_state:
    st.session_state.lon = None
if 'training_results' not in st.session_state:
    st.session_state.training_results = None
if 'reg_training_results' not in st.session_state:
    st.session_state.reg_training_results = None

# Створюємо список кроків
steps = [
    "🔴 Завантаження даних (Load data)", 
    "🔴 Навчання ML моделі (ML model training)", 
    "🔴 Прогноз (Forecast)"
]

# Навігація через бічну панель (Sidebar)
step = st.sidebar.radio("Навігація по проєкту:", steps, key="navigation_radio", 
                        index=steps.index(st.session_state.current_step))

st.session_state.current_step = step


# ==========================================
# 1. ОТРИМАННЯ ТА АНАЛІЗ ДАНИХ (EDA)
# ==========================================

if st.session_state.current_step == "🔴 Завантаження даних (Load data)":
    st.header("Отримання історичних метеоданих про погоду")
    
    # Конструює елементи інтерфейсу для вибору локації та часового проміжку завантаження історії.
    col1, col2 = st.columns(2)
    with col1:
        location_type = st.radio("Як задати локацію?", ["Назва міста", "Координати"])
        if location_type == "Назва міста":
            city = st.text_input("Введіть назву населеного пункту (напр. Kyiv, Lviv)", "Kyiv")
        else:
            lat_input = st.number_input("Широта (Latitude)", value=50.45)
            lon_input = st.number_input("Довгота (Longitude)", value=30.52)
            
    with col2:
        default_start = date.today() - timedelta(days=365*2)
        default_end = date.today() - timedelta(days=7) 
        start_d = st.date_input("Дата початку", default_start)
        end_d = st.date_input("Дата кінця", default_end)
        
    # Запускає процес звернення до API, зберігає отриманий файл та перенаправляє на етап навчання.
    if st.button("Завантажити дані", type="primary"):
        with st.spinner("Отримання даних..."):
            if location_type == "Назва міста":
                lat, lon = get_coordinates(city)
                if lat is None:
                    st.error("Не вдалося знайти координати міста. Спробуйте ввести координати вручну.")
                    st.stop()
            else:
                lat, lon = lat_input, lon_input
            
            st.session_state.lat = lat
            st.session_state.lon = lon
            
            df, missing_count = fetch_weather_data(lat, lon, start_d, end_d, is_forecast=False)
            
            if df is not None:
                st.session_state.historical_df = df
                file_name = 'weather_daily.csv'
                df.to_csv(file_name, index=False)
                
                st.success(f"✅ Успішно завантажено {len(df)} записів та збережено у CSV!")
                st.session_state.current_step = "🔴 Навчання ML моделі (ML model training)"
                st.rerun()
            else:
                st.error("Помилка при завантаженні даних")

    # Візуалізує первинний статистичний аналіз (EDA) завантажених історичних метеоданих.
    if st.session_state.historical_df is not None:
        st.write("### 📊 Попередній аналіз завантаженого датасету (EDA)")
        hist_col1, hist_col2 = st.columns(2)
        
        with hist_col1:
            has_precip = (st.session_state.historical_df['precipitation_sum'] > 0).map({True: 'З опадами (1)', False: 'Без опадів (0)'})
            fig_pie = px.pie(names=has_precip, title="Баланс класів в історичних даних", 
                             color_discrete_sequence=['#4caf50', '#2196f3'], hole=0.4)
            st.plotly_chart(fig_pie, use_container_width=True)
            
        with hist_col2:
            hist_df = st.session_state.historical_df.copy()
            hist_df['Місяць'] = hist_df['time'].dt.strftime('%B')
            fig_box = px.box(hist_df[hist_df['precipitation_sum'] > 0], x='Місяць', y='precipitation_sum',
                             title="Розподіл кількості опадів (мм) за місяцями (тільки дощові дні)",
                             color_discrete_sequence=['#1f77b4'])
            st.plotly_chart(fig_box, use_container_width=True)

        st.write("### 📋 Останні 5 рядків завантажених даних")
        st.dataframe(st.session_state.historical_df.tail(), use_container_width=True)
        csv = st.session_state.historical_df.to_csv(index=False).encode('utf-8')
        st.download_button("⬇️ Завантажити CSV", data=csv, file_name='weather_daily.csv', mime='text/csv')


# ==========================================
# 2. ДВОКАНАЛЬНЕ НАВЧАННЯ (ML TRAINING)
# ==========================================

elif st.session_state.current_step == "🔴 Навчання ML моделі (ML model training)":
    st.header("Навчання, підбір гіперпараметрів та оцінка моделей")
    
    st.write("### 🏗️ Архітектура паралельного моделювання (Multi-task Pipeline)")
    st.info("""
    Застосунок реалізує два паралельних треки прогнозування на основі хронологічного розбиття Time-Series Splitting (без перемішування даних):
    1. **Канал класифікації (Класифікатори):** Прогнозує ймовірність та факт настання опадів (так/ні).
      - **Test Set - 20% вибірки**: Хронологічна частина даних, яка відкладається без перемішування (`shuffle=False`).
      - **Train Set - перші 80% вибірки**: Передається у GridSearchCV.
      - **Крос-валідація**: Модель вчиться на минулому і перевіряється на наступному періоді, ніколи не заглядаючи в майбутнє.
    2. **Канал регресії (Регресори):** Прогнозує точний фізичний обсяг опадів у міліметрах (мм) для днів, коли вони очікуються.
    """)
    
    if st.session_state.historical_df is None:
        st.warning("Спочатку завантажте дані на першому кроці.")
    else:
        df = st.session_state.historical_df.copy()
        
        # ---------------------------------------------------------
        # [ФОРМУВАННЯ ЦІЛЬОВИХ ЗМІННИХ ДЛЯ ОБОХ ТРЕКІВ]
        # ---------------------------------------------------------
        df['target_class'] = (df['precipitation_sum'] > 0).astype(int)  # Ціль 1: Бінарна мітка (0 або 1)
        df['target_reg'] = df['precipitation_sum']                     # Ціль 2: Чиста кількість у мм
        
        cols_to_drop = ['time', 'precipitation_sum', 'target_class', 'target_reg']
        df = engineer_features(df)
        
        X = df.drop(columns=cols_to_drop)
        y_class = df['target_class']
        y_reg = df['target_reg']
        st.session_state.feature_cols = X.columns.tolist()

        if st.button("Запустити паралельне навчання моделей", type="primary"):
            with st.spinner("Виконується оптимізація класифікаторів та регресорів..."):
                
                X_train, X_test, y_train_cls, y_test_cls = train_test_split(X, y_class, test_size=0.2, shuffle=False)
                _, _, y_train_reg, y_test_reg = train_test_split(X, y_reg, test_size=0.2, shuffle=False)
                
                tscv = TimeSeriesSplit(n_splits=5)
                
                # =========================================================
                # 🟩 ТРЕК №1: КОНВЕЄР КЛАСИФІКАЦІЇ (ФАКТ ОПАДІВ ТАК/НІ)
                # =========================================================
                cls_models = {
                    "Random Forest": {
                        "pipeline": Pipeline([
                            ('imputer', SimpleImputer(strategy='median')),
                            ('scaler', StandardScaler()),
                            ('feature_selection', SelectFromModel(RandomForestClassifier(n_estimators=50, random_state=123))),
                            ('classifier', RandomForestClassifier(class_weight='balanced', random_state=123))
                        ]),
                        "params": {'classifier__n_estimators': [50, 100], 'classifier__max_depth': [None, 10]}
                    },
                    "Logistic Regression": {
                        "pipeline": Pipeline([
                            ('imputer', SimpleImputer(strategy='median')),
                            ('scaler', StandardScaler()),
                            ('feature_selection', SelectFromModel(LogisticRegression(penalty='l1', solver='liblinear', random_state=123))),
                            ('classifier', LogisticRegression(class_weight='balanced', max_iter=1000, random_state=123))
                        ]),
                        "params": {'classifier__C': [0.1, 1.0]}
                    }
                }
                
                cls_results = []
                best_f1 = 0
                best_cls_pipeline = None
                best_cls_name = ""
                best_cls_params = {}
                
                # Виконує крос-валідацію часових рядів та пошук гіперпараметрів для моделей класифікації.
                for name, config in cls_models.items():
                    grid = GridSearchCV(estimator=config["pipeline"], param_grid=config["params"], cv=tscv, scoring='f1', n_jobs=-1)
                    grid.fit(X_train, y_train_cls)
                    model = grid.best_estimator_
                    
                    preds = model.predict(X_test)
                    probas = model.predict_proba(X_test)[:, 1]
                    
                    f1 = f1_score(y_test_cls, preds, zero_division=0)
                    cls_results.append({
                        "Модель": name, "CV F1": grid.best_score_, "Test F1": f1,
                        "Accuracy": accuracy_score(y_test_cls, preds), "Precision": precision_score(y_test_cls, preds, zero_division=0),
                        "Recall": recall_score(y_test_cls, preds, zero_division=0), "ROC-AUC": roc_auc_score(y_test_cls, probas)
                    })
                    if f1 > best_f1:
                        best_f1 = f1
                        best_cls_name = name
                        best_cls_pipeline = model
                    best_cls_params[name] = {k.replace('classifier__', ''): v for k, v in grid.best_params_.items()}
                
                # =========================================================
                # 🟦 ТРЕК №2: КОНВЕЄР РЕГРЕСІЇ (ОБСЯГ ОПАДІВ У МІЛІМЕТРАХ)
                # =========================================================
                reg_models = {
                    "Random Forest Regressor": {
                        "pipeline": Pipeline([
                            ('imputer', SimpleImputer(strategy='median')),
                            ('scaler', StandardScaler()),
                            ('feature_selection', SelectFromModel(RandomForestRegressor(n_estimators=50, random_state=123))),
                            ('regressor', RandomForestRegressor(random_state=123))
                        ]),
                        "params": {'regressor__n_estimators': [50, 100]}
                    },
                    "Ridge Regression": {
                        "pipeline": Pipeline([
                            ('imputer', SimpleImputer(strategy='median')),
                            ('scaler', StandardScaler()),
                            ('feature_selection', SelectFromModel(Ridge())),
                            ('regressor', Ridge())
                        ]),
                        "params": {'regressor__alpha': [0.1, 1.0, 10.0]}
                    }
                }
                
                reg_results = []
                best_mae = float('inf')
                best_reg_pipeline = None
                best_reg_name = ""
                
                # Виконує крос-валідацію та пошук оптимальних параметрів для моделей регресії кількості опадів.
                for name, config in reg_models.items():
                    grid_reg = GridSearchCV(estimator=config["pipeline"], param_grid=config["params"], cv=tscv, scoring='neg_mean_absolute_error', n_jobs=-1)
                    grid_reg.fit(X_train, y_train_reg)
                    model_r = grid_reg.best_estimator_
                    
                    preds_r = model_r.predict(X_test)
                    mae = mean_absolute_error(y_test_reg, preds_r)
                    
                    reg_results.append({
                        "Модель регресії": name, "CV MAE": -grid_reg.best_score_, "Test MAE": mae, "Test R²": r2_score(y_test_reg, preds_r)
                    })
                    if mae < best_mae:
                        best_mae = mae
                        best_reg_name = name
                        best_reg_pipeline = model_r
                
                # ---------------------------------------------------------
                # [ЗБЕРЕЖЕННЯ ОПТИМАЛЬНИХ МОДЕЛЕЙ З ОБОХ ТРЕКІВ У СЕСІЮ]
                # ---------------------------------------------------------
                st.session_state.best_model = best_cls_pipeline
                st.session_state.best_model_name = best_cls_name
                st.session_state.best_regressor = best_reg_pipeline
                st.session_state.best_regressor_name = best_reg_name
                
                st.session_state.training_results = cls_results
                st.session_state.reg_training_results = reg_results
                st.session_state.best_params_display = best_cls_params
                st.session_state.y_test = y_test_cls
                st.session_state.best_y_pred = best_cls_pipeline.predict(X_test)
                
                st.success(f"🎉 Навчання завершено! Класифікатор: {best_cls_name}, Регресор: {best_reg_name}")
                st.session_state.current_step = "🔴 Прогноз (Forecast)"
                st.rerun()

        # Формує зведені таблиці порівняння якості моделей та знайдені гіперпараметри.
        if st.session_state.best_model is not None:
            col_tab1, col_tab2 = st.columns(2)
            
            with col_tab1:
                st.write("### 📊 Результати класифікації (Факт опадів)")
                st.dataframe(pd.DataFrame(st.session_state.training_results).style.highlight_max(subset=['Test F1', 'Accuracy', 'ROC-AUC'], color='lightgreen'), use_container_width=True)
            
            with col_tab2:
                st.write("### 📉 Результати регресії (Обсяг у мм)")
                st.dataframe(pd.DataFrame(st.session_state.reg_training_results).style.highlight_min(subset=['Test MAE'], color='lightgreen'), use_container_width=True)
            
            st.write("### ⚙️ Знайдені найкращі гіперпараметри")
            for m_name, p in st.session_state.best_params_display.items():
                st.markdown(f"- **{m_name}:** {p}")

            # Матриця помилок класифікатора
            # Будує теплову карту матриці помилок для детального аналізу хибнопозитивних та хибнонегативних передбачень.
            st.divider()
            st.write(f"### 🧮 Матриця помилок для {st.session_state.best_model_name}")
            cm = confusion_matrix(st.session_state.y_test, st.session_state.best_y_pred)
            fig_cm = px.imshow(cm, text_auto=True, color_continuous_scale='Blues',
                               labels=dict(x="Прогноз моделі", y="Фактично", color="Кількість"),
                               x=['Без опадів (0)', 'З опадами (1)'], y=['Без опадів (0)', 'З опадами (1)'])
            st.plotly_chart(fig_cm, use_container_width=True)

            # Важливість ознак
            # Візуалізує рівень математичної важливості (Feature Importance) кожної відібраної ознаки для фінальної моделі.
            st.divider()
            st.write("### 🔍 Відбір ознак (Feature Selection)")
            pipeline = st.session_state.best_model
            selector = pipeline.named_steps['feature_selection']
            classifier = pipeline.named_steps['classifier']
            
            feature_names = np.array(st.session_state.feature_cols)
            selected_features = feature_names[selector.get_support()]
            
            st.write("#### Важливість відібраних ознак для фінальної моделі:")
            importances = classifier.feature_importances_ if "Random Forest" in st.session_state.best_model_name else np.abs(classifier.coef_[0])
            
            feat_imp_df = pd.DataFrame({'Ознака': selected_features, 'Вплив на прогноз': importances}).sort_values(by='Вплив на прогноз', ascending=True)
            fig = px.bar(feat_imp_df, x='Вплив на прогноз', y='Ознака', orientation='h', color='Вплив на прогноз', color_continuous_scale='Viridis')
            st.plotly_chart(fig, use_container_width=True)


# ==========================================
# 3. ДВОКАНАЛЬНИЙ ПРОГНОЗ ТА КАРТКИ МЕТРИК
# ==========================================

elif st.session_state.current_step == "🔴 Прогноз (Forecast)":
    st.header("Прогноз опадів на наступні дні")
    
    if st.session_state.best_model is None or st.session_state.best_regressor is None:
        st.warning("Спочатку навчіть моделі на кроці 2.")
    elif st.session_state.lat is None:
        st.warning("Спочатку завантажте локацію на кроці 1.")
    else:
        forecast_days = st.slider("Кількість днів для прогнозу", min_value=1, max_value=14, value=7)

        # Здійснює запит на прогнозні дні та пропускає метеопараметри через обидві паралельні ML-моделі.
        if st.button("Отримати прогноз", type="primary"):
            with st.spinner("Отримання прогнозу..."):
                start_f = date.today()
                end_f = start_f + timedelta(days=forecast_days - 1)
                
                df_forecast, missing_f = fetch_weather_data(st.session_state.lat, st.session_state.lon, start_f, end_f, is_forecast=True)

                if df_forecast is not None:
                    df_features = engineer_features(df_forecast)
                    X_forecast = df_features[st.session_state.feature_cols]

                    # 🟩 Виклик Треку №1: Класифікація ймовірності дощу
                    model_cls = st.session_state.best_model
                    predictions = model_cls.predict(X_forecast)
                    probabilities = model_cls.predict_proba(X_forecast)[:, 1]
                    
                    # 🟦 Виклик Треку №2: Регресія очікуваних міліметрів від нашого ШІ
                    model_reg = st.session_state.best_regressor
                    pred_mm = model_reg.predict(X_forecast)

                    # Об'єднує результати розрахунків обох моделей та сирі дані API в один фінальний датафрейм.
                    results_display = pd.DataFrame({
                        "Дата": df_forecast['time'].dt.date,
                        "Макс. Темп. (°C)": df_forecast['temperature_2m_max'],
                        "Вітер (км/год)": df_forecast['wind_speed_10m_max'],
                        "Очікуються опади?": ["🌧️ ТАК" if p == 1 else "☀️ НІ" for p in predictions],
                        "Ймовірність опадів": [f"{prob*100:.1f}%" for prob in probabilities],
                        "Кількість опадів (Наш ШІ)": [f"{max(0.0, mm):.1f} мм" if p == 1 else "0.0 мм" for p, mm in zip(predictions, pred_mm)],
                        "Кількість опадів (Open-Meteo)": [f"{mm:.1f} мм" for mm in df_forecast['precipitation_sum']],
                        "prob_raw": probabilities * 100
                    })

                    # Генерує великі інформаційні картки-віджети стану погоди на поточний день.
                    st.write("### 🌡️ Метеопоказники на сьогодні")
                    m1, m2, m3, m4 = st.columns(4)
                    m1.metric("Макс. Температура", f"{results_display.iloc[0]['Макс. Темп. (°C)']:.1f} °C")
                    m2.metric("Максимальний вітер", f"{results_display.iloc[0]['Вітер (км/год)']:.1f} км/год")
                    
                    today_rain = results_display.iloc[0]['Очікуються опади?']
                    m3.metric("Рішення ШІ щодо дощу", today_rain, delta="Risk опадів!" if "ТАК" in today_rain else "Сухо")
                    m4.metric("Обсяг: Наш ШІ vs Open-Meteo", 
                              results_display.iloc[0]['Кількість опадів (Наш ШІ)'], 
                              delta=f"В API: {results_display.iloc[0]['Кількість опадів (Open-Meteo)']}", 
                              delta_color="off")
                    st.divider()

                    # Виводить інтерактивну таблицю з кольоровою стилізацією клітинок залежно від вердикту ШІ.
                    st.write("### Result прогнозування")
                    def color_rain(val):
                        if 'ТАК' in str(val):
                            return 'background-color: #d32f2f; color: white; font-weight: bold;' 
                        elif 'НІ' in str(val):
                            return 'background-color: #388e3c; color: white; font-weight: bold;'
                        return ''

                    show_cols = [
                        "Дата", "Макс. Темп. (°C)", "Вітер (км/год)", 
                        "Очікуються опади?", "Ймовірність опадів", 
                        "Кількість опадів (Наш ШІ)", "Кількість опадів (Open-Meteo)"
                    ]
                    st.dataframe(results_display[show_cols].style.map(color_rain, subset=['Очікуються опади?']), use_container_width=True)

                    # Будує фінальний комбінований графік динаміки температур та ймовірності дощу.
                    st.write("### Візуалізація прогнозу")
                    fig2 = go.Figure()
                    fig2.add_trace(go.Bar(
                        x=results_display['Дата'], y=results_display['prob_raw'], name="Ймовірність опадів (%)",
                        marker_color=['#1f77b4' if p == 1 else '#b0bec5' for p in predictions], opacity=0.85
                    ))
                    fig2.add_trace(go.Scatter(
                        x=results_display['Дата'], y=results_display['Макс. Темп. (°C)'],
                        mode='lines+markers', name='Температура (°C)', yaxis='y2', 
                        line=dict(color='#d32f2f', width=2.5), marker=dict(size=8, symbol='square', line=dict(color='white', width=1))
                    ))
                    fig2.update_layout(
                        title="Динаміка температури та ймовірності опадів за прогнозом ШІ",
                        template="simple_white",
                        yaxis=dict(title="Ймовірність (%)", range=[0, 105], showgrid=True, gridcolor="#a5a4a4"),
                        yaxis2=dict(title="Температура (°C)", overlaying='y', side='right', showgrid=False),
                        legend=dict(x=0.01, y=0.99, bgcolor='rgba(255,255,255,0.9)', bordercolor='black', borderwidth=1),
                        hovermode="x unified"
                    )
                    st.plotly_chart(fig2, use_container_width=True)
                else:
                    st.error("Помилка отримання даних прогнозу")