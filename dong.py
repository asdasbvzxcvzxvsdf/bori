import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

# 기상 데이터
gisang_data = pd.DataFrame({
    '지점': [140, 140, 140, 140, 146, 146, 146, 146, 172, 172, 172, 172, 243, 243, 243, 243, 244, 244, 244, 244, 245, 245, 245, 245, 247, 247, 247, 248, 248, 248],
    '지점명': ['군산', '군산', '군산', '군산', '전주', '전주', '전주', '전주', '고창', '고창', '고창', '고창', '부안', '부안', '부안', '부안', '임실', '임실', '임실', '임실', '정읍', '정읍', '정읍', '정읍', '남원', '남원', '남원', '장수', '장수', '장수'],
    '일시': [2018, 2019, 2020, 2021, 2018, 2019, 2020, 2021, 2018, 2019, 2020, 2021, 2018, 2019, 2020, 2021, 2018, 2019, 2020, 2021, 2018, 2019, 2020, 2021, 2018, 2020, 2021, 2018, 2019, 2020],
    '합계 강수량(mm)': [1637.1, 1008.1, 1664.8, 1151.7, 1332.5, 968.9, 1780.8, 1496.6, 1316.9, 1099.5, 1554.7, 1255.8, 1359.5, 1087.5, 1694.8, 1354.4, 1468.2, 1069.7, 1869.8, 1259.2, 1305.53, 1554.7, 1718.9, 1354.6, 1332.8, 2007.2, 1136.8, 1665.49, 1262.8, 2126],
    '일 최고 기온(°C)': [57.66, 55.66, 52.13, 53.69, 55.1, 52.98, 50.19, 51.08, 53.8, 52.09, 47.88, 48.52, 52.75, 53.61, 47.32, 48.14, 49.54, 47.08, 44.82, 46.19, 72.5, 47.46, 49.45, 50.23, 52.02, 48.47, 50.22, 53.9, 47.14, 44.58]
})

# 보리 생산 데이터
bori_data = pd.DataFrame({
    '맥류작물별(1)': ['Outer barley', 'Outer barley', 'Outer barley', 'Outer barley', 'Rice Barley', 'Rice Barley', 'Rice Barley', 'Rice Barley', 'wheat', 'wheat', 'wheat', 'wheat', 'Beer Barley', 'Beer Barley', 'Beer Barley', 'Beer Barley'],
    '항목': ['면적 (ha)', '생산량 (M/T)', '단위생산량 (kg/10a)', '면적 (ha)', '생산량 (M/T)', '단위생산량 (kg/10a)', '면적 (ha)', '생산량 (M/T)', '단위생산량 (kg/10a)', '면적 (ha)', '생산량 (M/T)', '단위생산량 (kg/10a)', '면적 (ha)', '생산량 (M/T)', '단위생산량 (kg/10a)', '면적 (ha)'],
    '2018': [7366, 15693, 213, 8666, 21299, 246, 1513, 6399, 423, 409, 954, 233, 409, 954, 233, 409],
    '2019': [5814, 18655, 321, 7839, 24603, 337, 863, 3805, 441, 286, 1037, 363, 286, 1037, 363, 286],
    '2020': [3426, 9612, 281, 7145, 21768, 305, 1355, 5759, 425, 28, 87, 313, 28, 87, 313, 28],
    '2021': [3038, 9467, 312, 7324, 23845, 326, 1855, 8552, 461, 47, 163, 348, 47, 163, 348, 47]
})
# 보리 생산 데이터를 '맥류작물별(1)'과 '항목'을 기준으로 긴 형태로 변환
bori_data_long = pd.melt(bori_data, id_vars=['맥류작물별(1)', '항목'], var_name='연도', value_name='값')

# 필요한 지역 데이터만 필터링
gisang_data_filtered = gisang_data[gisang_data['지점명'].isin(['군산', '전주', '고창', '부안', '임실', '정읍', '남원', '장수'])]

# '연도' 데이터 타입을 통일시키기 위해, int로 변환 (copy 메서드 사용)
bori_data_long['연도'] = bori_data_long['연도'].astype(int)

# 작물 목록
crops = ['Outer barley', 'Rice Barley', 'wheat', 'Beer Barley']

# Streamlit 앱 시작
st.title("보리 생산과 기상 조건의 관계 분석")

# 작물 선택을 위한 셀렉트박스 추가
selected_crop = st.selectbox("분석할 작물 선택", crops)

# 특정 작물의 생산량 데이터만 추출
bori_production = bori_data_long[(bori_data_long['항목'] == '생산량 (M/T)') & (bori_data_long['맥류작물별(1)'] == selected_crop)]

# 데이터 병합
merged_data = pd.merge(bori_production, gisang_data_filtered, left_on=['연도'], right_on=['일시'])

# 필요한 열만 선택
merged_data = merged_data[['연도', '값', '합계 강수량(mm)', '일 최고 기온(°C)']]

# 열 이름 변경
merged_data.columns = ['연도', '생산량', '강수량', '기온']

# 결측치 제거
merged_data = merged_data.dropna()

# 특성과 타겟 데이터 설정
X = merged_data[['강수량', '기온']]
y = merged_data['생산량']

# 학습 데이터와 테스트 데이터로 나누기
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 선형 회귀 모델 생성 및 학습
model = LinearRegression()
model.fit(X_train, y_train)

# 테스트 데이터로 예측
y_pred = model.predict(X_test)

# 평가 지표 계산
mse = mean_squared_error(y_test, y_pred)
rmse = mse ** 0.5

# 결과 출력
st.write(f"{selected_crop} - Root Mean Squared Error: {rmse}")

# 그래프로 시각화
fig, ax = plt.subplots()
ax.scatter(y_test, y_pred, color='blue', label='Forecast')
ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2, color='red', label='Actuals')
ax.set_xlabel('Actual production')
ax.set_ylabel('Forecast Production')
ax.set_title(f'{selected_crop}')
ax.legend()

# 그래프를 Streamlit 앱에 추가
st.pyplot(fig)

# 년도를 표시하는 부분 추가
for j in range(len(y_test)):
    plt.text(y_test.values[j], y_pred[j], f'{merged_data.iloc[X_test.index[j]]["연도"]}', fontsize=9, color='black')

# Streamlit 앱 실행
st.write("그래프 상에 년도 표시")
st.pyplot(fig)
