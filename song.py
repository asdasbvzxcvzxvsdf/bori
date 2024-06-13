import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.ensemble import RandomForestRegressor

st.set_page_config(page_title="Crop Production Prediction", layout="wide")

st.title("맥류 생산 예측 모델")

# Sidebar for navigation
st.sidebar.title("Navigation")
options = st.sidebar.radio("Go to", ["Home", "Model Evaluation", "Predict Future Production"])

# Load Data
def load_data():
    temperature_data = {
        'Year': [2001, 2002, 2003, 2004, 2005, 2006, 2007, 2008, 2009, 2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019, 2020, 2021],
        'Average_Temp': [11.1, 11.2, 11.5, 11.8, 10.9, 11.4, 11.8, 11.5, 11.9, 12.4, 10.7, 10.6, 11.2, 11.5, 12.1, 12.5, 11.6, 12.0, 12.2, 12.0, 12.5],
        'Min_Temp': [5.3, 5.6, 6.1, 5.8, 5.2, 5.8, 6.2, 5.6, 5.8, 6.1, 7.3, 6.5, 5.6, 5.8, 6.5, 7.0, 6.6, 6.4, 6.4, 6.9, 7.2],
        'Max_Temp': [18.1, 17.8, 17.8, 18.9, 17.5, 18.4, 18.8, 18.6, 18.9, 18.2, 17.3, 16.9, 17.6, 18.1, 18.6, 19.1, 18.6, 18.8, 19.1, 18.0, 18.7]
    }

    rainfall_data = {
        'Year': [2001, 2002, 2003, 2004, 2005, 2006, 2007, 2008, 2009, 2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019, 2020, 2021],
        'Rainfall': [1298.1, 1580.3, 1974.2, 1274.1, 1418.4, 1260.7, 1438.1, 915.3, 1134.4, 1795.1, 1700.0, 1503.9, 1400.0, 1340.5, 843.1, 1326.5, 958.3, 1468.2, 1069.7, 1869.8, 1259.2]
    }

    fertilizer_data = {
        'Year': [2001, 2002, 2003, 2004, 2005, 2006, 2007, 2008, 2009, 2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019, 2020, 2021],
        'Total_Fertilizer_Usage': [717, 690, 678, 747, 722, 477, 631, 570, 500, 423, 447, 472, 459, 453, 439, 451, 442, 434, 441, 431, 461],
        'Fertilizer_Usage_per_ha': [343, 342, 350, 385, 376, 257, 340, 311, 267, 233, 249, 267, 262, 258, 261, 268, 270, 262, 268, 266, 286]
    }

    sunshine_data = {
        'Year': [2001, 2002, 2003, 2004, 2005, 2006, 2007, 2008, 2009, 2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019, 2020, 2021],
        'Sunshine_Duration': [2398.6, 2369.1, 2251.5, 2573.4, 2523.2, 2392.7, 1833.4, 2008.8, 2073.5, 1813.6, 1963.6, 1951.1, 2067.1, 1974.8, 1964.9, 1965.9, 2182.1, 2205.6, 2095.8, 1986.3, 2080.1],
        'Sunshine_Rate': [53.92, 53.22, 50.58, 57.68, 56.68, 53.75, 41.17, 45.03, 46.58, 40.74, 44.11, 43.73, 46.44, 44.36, 44.14, 44.07, 49.02, 49.54, 47.08, 44.52, 46.73]
    }

    humidity_data = {
        'Year': [2001, 2002, 2003, 2004, 2005, 2006, 2007, 2008, 2009, 2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019, 2020, 2021],
        'Average_Humidity': [69.6, 71.7, 70.7, 69.0, 73.0, 73.4, 72.8, 71.4, 69.7, 70.3, 68.8, 68.7, 70.6, 71.7, 71.8, 72.4, 70.5, 71.3, 72.1, 77.6, 76.7],
        'Min_Humidity': [13.0, 15.0, 14.0, 10.0, 12.0, 6.0, 8.0, 7.0, 7.0, 11.0, 5.0, 7.0, 5.0, 10.0, 2.0, 8.0, 7.0, 7.0, 9.0, 14.0, 11.0]
    }

    crop_production_data = {
        'Year': [2001, 2002, 2003, 2004, 2005, 2006, 2007, 2008, 2009, 2010, 2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019, 2020, 2021],
        'Naked_Barley_Production': [8170, 7650, 5096, 7038, 5339, 4717, 5895, 6234, 2395, 3143, 4081, 9684, 11968, 13721, 13737, 7137, 11866, 15693, 18655, 9612, 9467],
        'Rice_Barley_Production': [52725, 24743, 19747, 31227, 40525, 24928, 31384, 38220, 33244, 32195, 39795, 36328, 23631, 36772, 23923, 13881, 11540, 21299, 26403, 21768, 23845],
        'Wheat_Production': [188, 1114, 2148, 3362, 1417, 518, 1674, 2791, 5725, 1410, 3830, 10127, 4970, 5908, 6501, 10358, 8637, 6399, 3805, 5759, 8552]
    }

    # 데이터프레임 생성
    temp_df = pd.DataFrame(temperature_data)
    rainfall_df = pd.DataFrame(rainfall_data)
    fertilizer_df = pd.DataFrame(fertilizer_data)
    sunshine_df = pd.DataFrame(sunshine_data)
    humidity_df = pd.DataFrame(humidity_data)
    crop_df = pd.DataFrame(crop_production_data)

    # 데이터 결합
    merged_df = pd.merge(pd.merge(pd.merge(pd.merge(pd.merge(temp_df, rainfall_df, on='Year'), fertilizer_df, on='Year'), sunshine_df, on='Year'), humidity_df, on='Year'), crop_df, on='Year')

    # 결측치 처리 (필요한 경우)
    merged_df.fillna(merged_df.mean(), inplace=True)

    return merged_df

# 데이터 로드
merged_df = load_data()

# 특성과 타겟 데이터 분리
X = merged_df[['Average_Temp', 'Min_Temp', 'Max_Temp', 'Rainfall', 'Total_Fertilizer_Usage', 'Fertilizer_Usage_per_ha', 'Sunshine_Duration', 'Sunshine_Rate', 'Average_Humidity', 'Min_Humidity']]
y_naked_barley = merged_df['Naked_Barley_Production']
y_rice_barley = merged_df['Rice_Barley_Production']
y_wheat = merged_df['Wheat_Production']

# 데이터 스케일링
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 선형 회귀 모델 생성
model_naked_barley = LinearRegression().fit(X_scaled, y_naked_barley)
model_rice_barley = LinearRegression().fit(X_scaled, y_rice_barley)
model_wheat = LinearRegression().fit(X_scaled, y_wheat)

# 랜덤 포레스트 모델 생성
rf_model_naked_barley = RandomForestRegressor(n_estimators=100, random_state=42).fit(X, y_naked_barley)
rf_model_rice_barley = RandomForestRegressor(n_estimators=100, random_state=42).fit(X, y_rice_barley)
rf_model_wheat = RandomForestRegressor(n_estimators=100, random_state=42).fit(X, y_wheat)

# 모델 평가 함수
def evaluate_model(model, X, y, model_name):
    predictions = model.predict(X)
    mse = mean_squared_error(y, predictions)
    mae = mean_absolute_error(y, predictions)
    r2 = r2_score(y, predictions)
    return mse, mae, r2

# 선형 회귀 모델 평가
mse_nb_lr, mae_nb_lr, r2_nb_lr = evaluate_model(model_naked_barley, X_scaled, y_naked_barley, "Naked Barley Production (Linear Regression)")
mse_rb_lr, mae_rb_lr, r2_rb_lr = evaluate_model(model_rice_barley, X_scaled, y_rice_barley, "Rice Barley Production (Linear Regression)")
mse_w_lr, mae_w_lr, r2_w_lr = evaluate_model(model_wheat, X_scaled, y_wheat, "Wheat Production (Linear Regression)")

# 랜덤 포레스트 모델 평가
mse_nb_rf, mae_nb_rf, r2_nb_rf = evaluate_model(rf_model_naked_barley, X, y_naked_barley, "Naked Barley Production (Random Forest)")
mse_rb_rf, mae_rb_rf, r2_rb_rf = evaluate_model(rf_model_rice_barley, X, y_rice_barley, "Rice Barley Production (Random Forest)")
mse_w_rf, mae_w_rf, r2_w_rf = evaluate_model(rf_model_wheat, X, y_wheat, "Wheat Production (Random Forest)")

# 2022년도 예측 (예시 데이터 사용)
new_weather_data = np.array([[12.0, 6.4, 18.5, 966.7, 410, 255, 2166.9, 48.67, 72.3, 10.0]])
new_weather_scaled = scaler.transform(new_weather_data)

# 선형 회귀 모델의 예측값
naked_barley_pred_lr = model_naked_barley.predict(new_weather_scaled)
rice_barley_pred_lr = model_rice_barley.predict(new_weather_scaled)
wheat_pred_lr = model_wheat.predict(new_weather_scaled)

# 랜덤 포레스트 모델의 예측값
naked_barley_pred_rf = rf_model_naked_barley.predict(new_weather_data)
rice_barley_pred_rf = rf_model_rice_barley.predict(new_weather_data)
wheat_pred_rf = rf_model_wheat.predict(new_weather_data)

if options == "Home":
    st.header("어서오세요 맥류생산예측 모델입니다")
    st.write("""
    이 웹페이지로 맥류생산예측과 모델성능평가를 보실수있습니다
    """)

elif options == "Model Evaluation":
    st.header("Model Evaluation Results")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Linear Regression Model")
        st.write("**Naked Barley Production**")
        st.write(f"MSE: {mse_nb_lr:.2f}")
        st.write(f"MAE: {mae_nb_lr:.2f}")
        st.write(f"R²: {r2_nb_lr:.2f}")
        st.write("---")
        st.write("**Rice Barley Production**")
        st.write(f"MSE: {mse_rb_lr:.2f}")
        st.write(f"MAE: {mae_rb_lr:.2f}")
        st.write(f"R²: {r2_rb_lr:.2f}")
        st.write("---")
        st.write("**Wheat Production**")
        st.write(f"MSE: {mse_w_lr:.2f}")
        st.write(f"MAE: {mae_w_lr:.2f}")
        st.write(f"R²: {r2_w_lr:.2f}")

    with col2:
        st.subheader("Random Forest Model")
        st.write("**Naked Barley Production**")
        st.write(f"MSE: {mse_nb_rf:.2f}")
        st.write(f"MAE: {mae_nb_rf:.2f}")
        st.write(f"R²: {r2_nb_rf:.2f}")
        st.write("---")
        st.write("**Rice Barley Production**")
        st.write(f"MSE: {mse_rb_rf:.2f}")
        st.write(f"MAE: {mae_rb_rf:.2f}")
        st.write(f"R²: {r2_rb_rf:.2f}")
        st.write("---")
        st.write("**Wheat Production**")
        st.write(f"MSE: {mse_w_rf:.2f}")
        st.write(f"MAE: {mae_w_rf:.2f}")
        st.write(f"R²: {r2_w_rf:.2f}")

elif options == "Predict Future Production":
    st.header("Predict Future Production for 2022, 2023, and 2024")

    # 미래 데이터를 코드에 하드코딩
    future_data = np.array([
        [12.0, 6.4, 18.5, 966.7, 410, 255, 2166.9, 48.67, 72.3, 10.0],  # 2022 데이터 예시
        [13.1, 7.8, 19.3, 1984.4, 450, 270, 2069.1, 46.48, 74.8, 8.0],  # 2023 데이터 예시
        [14.8, 8.3, 21.7, 413.3, 460, 275, 996.7, 49.77, 71.0, 15.0]   # 2024 데이터 예시
    ])
    future_data_scaled = scaler.transform(future_data)

    # 선형 회귀 모델 예측
    future_pred_lr = {
        "Naked Barley": model_naked_barley.predict(future_data_scaled),
        "Rice Barley": model_rice_barley.predict(future_data_scaled),
        "Wheat": model_wheat.predict(future_data_scaled)
    }

    # 랜덤 포레스트 모델 예측
    future_pred_rf = {
        "Naked Barley": rf_model_naked_barley.predict(future_data),
        "Rice Barley": rf_model_rice_barley.predict(future_data),
        "Wheat": rf_model_wheat.predict(future_data)
    }

    if st.button("22년, 23년, 24년도 맥류 생산량 예측해보기"):
        st.write("### Predicted Production for 2022, 2023, and 2024 (Linear Regression)")
        st.write(f"**2022 Naked Barley Production**: {future_pred_lr['Naked Barley'][0]:.2f} M/T")
        st.write(f"**2022 Rice Barley Production**: {future_pred_lr['Rice Barley'][0]:.2f} M/T")
        st.write(f"**2022 Wheat Production**: {future_pred_lr['Wheat'][0]:.2f} M/T")
        st.write(f"**2023 Naked Barley Production**: {future_pred_lr['Naked Barley'][1]:.2f} M/T")
        st.write(f"**2023 Rice Barley Production**: {future_pred_lr['Rice Barley'][1]:.2f} M/T")
        st.write(f"**2023 Wheat Production**: {future_pred_lr['Wheat'][1]:.2f} M/T")
        st.write(f"**2024 Naked Barley Production**: {future_pred_lr['Naked Barley'][2]:.2f} M/T")
        st.write(f"**2024 Rice Barley Production**: {future_pred_lr['Rice Barley'][2]:.2f} M/T")
        st.write(f"**2024 Wheat Production**: {future_pred_lr['Wheat'][2]:.2f} M/T")

        st.write("---")
        st.write("### Predicted Production for 2022, 2023, and 2024 (Random Forest)")
        st.write(f"**2022 Naked Barley Production**: {future_pred_rf['Naked Barley'][0]:.2f} M/T")
        st.write(f"**2022 Rice Barley Production**: {future_pred_rf['Rice Barley'][0]:.2f} M/T")
        st.write(f"**2022 Wheat Production**: {future_pred_rf['Wheat'][0]:.2f} M/T")
        st.write(f"**2023 Naked Barley Production**: {future_pred_rf['Naked Barley'][1]:.2f} M/T")
        st.write(f"**2023 Rice Barley Production**: {future_pred_rf['Rice Barley'][1]:.2f} M/T")
        st.write(f"**2023 Wheat Production**: {future_pred_rf['Wheat'][1]:.2f} M/T")
        st.write(f"**2024 Naked Barley Production**: {future_pred_rf['Naked Barley'][2]:.2f} M/T")
        st.write(f"**2024 Rice Barley Production**: {future_pred_rf['Rice Barley'][2]:.2f} M/T")
        st.write(f"**2024 Wheat Production**: {future_pred_rf['Wheat'][2]:.2f} M/T")

        # 그래프 그리기
        fig, ax = plt.subplots(2, 1, figsize=(12, 16))

        # Linear Regression Predictions
        ax[0].plot(merged_df['Year'], merged_df['Naked_Barley_Production'], marker='o', label='Naked Barley Production')
        ax[0].plot(merged_df['Year'], merged_df['Rice_Barley_Production'], marker='o', label='Rice Barley Production')
        ax[0].plot(merged_df['Year'], merged_df['Wheat_Production'], marker='o', label='Wheat Production')
        ax[0].plot([2022, 2023, 2024], future_pred_lr['Naked Barley'], 'ro-', label='Predicted Naked Barley Production (2022, 2023, 2024)')
        ax[0].plot([2022, 2023, 2024], future_pred_lr['Rice Barley'], 'go-', label='Predicted Rice Barley Production (2022, 2023, 2024)')
        ax[0].plot([2022, 2023, 2024], future_pred_lr['Wheat'], 'bo-', label='Predicted Wheat Production (2022, 2023, 2024)')
        ax[0].set_title('Production of Various Crops from 2001 to 2024 (Linear Regression)')
        ax[0].set_xlabel('Year')
        ax[0].set_ylabel('Production (M/T)')
        ax[0].legend()
        ax[0].grid(True)

        # Random Forest Predictions
        ax[1].plot(merged_df['Year'], merged_df['Naked_Barley_Production'], marker='o', label='Naked Barley Production')
        ax[1].plot(merged_df['Year'], merged_df['Rice_Barley_Production'], marker='o', label='Rice Barley Production')
        ax[1].plot(merged_df['Year'], merged_df['Wheat_Production'], marker='o', label='Wheat Production')
        ax[1].plot([2022, 2023, 2024], future_pred_rf['Naked Barley'], 'ro-', label='Predicted Naked Barley Production (2022, 2023, 2024)')
        ax[1].plot([2022, 2023, 2024], future_pred_rf['Rice Barley'], 'go-', label='Predicted Rice Barley Production (2022, 2023, 2024)')
        ax[1].plot([2022, 2023, 2024], future_pred_rf['Wheat'], 'bo-', label='Predicted Wheat Production (2022, 2023, 2024)')
        ax[1].set_title('Production of Various Crops from 2001 to 2024 (Random Forest)')
        ax[1].set_xlabel('Year')
        ax[1].set_ylabel('Production (M/T)')
        ax[1].legend()
        ax[1].grid(True)

        # Streamlit을 통해 그래프 출력
        st.pyplot(fig)
