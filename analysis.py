import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

import numpy as np
import os

RESULT_DIR = "result"
os.makedirs(RESULT_DIR, exist_ok=True)

# ===============================
# 1. 데이터 로드
# ===============================
use_cols = [
    "Start_Time",
    "Severity",
    "Temperature(F)",
    "Humidity(%)",
    "Visibility(mi)",
    "Wind_Speed(mph)",
    "Precipitation(in)",
    "Weather_Condition"
]

df = pd.read_csv(
    "data/US_Accidents_March23.csv",
    usecols=use_cols,
    nrows=300000
)

print(df.info())

# ===============================
# 2. 시간 처리 및 결측치 제거
# ===============================
df["Start_Time"] = pd.to_datetime(df["Start_Time"])
df["Date"] = df["Start_Time"].dt.date
df = df.dropna()

# ===============================
# 3. 날씨 카테고리 생성 
# ===============================
def weather_category(cond):
    cond = str(cond).lower()
    if "rain" in cond:
        return "Rain"
    elif "snow" in cond:
        return "Snow"
    elif "fog" in cond or "mist" in cond or "haze" in cond:
        return "Fog"
    elif "clear" in cond:
        return "Clear"
    else:
        return "Other"

df["Weather_Category"] = df["Weather_Condition"].apply(weather_category)

# ===============================
# 4. 하루 단위 데이터 생성
# ===============================
daily = df.groupby(["Date", "Weather_Category"]).agg({
    "Severity": "count",            
    "Temperature(F)": "mean",
    "Humidity(%)": "mean",
    "Visibility(mi)": "mean",
    "Wind_Speed(mph)": "mean",
    "Precipitation(in)": "mean"
}).reset_index()

daily.rename(columns={"Severity": "Accident_Count"}, inplace=True)

print(daily.head())
print(daily.info())

summary_table = daily.groupby("Weather_Category").agg({
    "Accident_Count": "mean",
    "Temperature(F)": "mean",
    "Humidity(%)": "mean",
    "Visibility(mi)": "mean",
    "Precipitation(in)": "mean"
})


summary_table = summary_table.round(2)
summary_table.to_csv(f"{RESULT_DIR}/summary_by_weather.csv")

print("\n[날씨 유형별 사고 및 기상 조건 요약]")
print(summary_table)

# ===============================
# 5. 그래프 ① 히스토그램
# ===============================
plt.figure(figsize=(8,5))

# Clear → 가장 진한 색 + 높은 alpha
daily[daily["Weather_Category"] == "Clear"]["Accident_Count"].plot(
    kind="hist",
    bins=30,
    alpha=0.9,                 
    color="black",             
    edgecolor="black",
    label="Clear"
)

# Rain → 빨강
daily[daily["Weather_Category"] == "Rain"]["Accident_Count"].plot(
    kind="hist",
    bins=30,
    alpha=0.6,
    color="tab:red",
    edgecolor="black",
    label="Rain"
)

# Snow → 초록
daily[daily["Weather_Category"] == "Snow"]["Accident_Count"].plot(
    kind="hist",
    bins=30,
    alpha=0.6,
    color="tab:green",
    edgecolor="black",
    label="Snow"
)

plt.title("Distribution of Accident Counts by Weather")
plt.xlabel("Number of Accidents per Day")
plt.ylabel("Frequency")
plt.legend()
plt.tight_layout()
plt.savefig(f"{RESULT_DIR}/hist_accident_by_weather.png", dpi=300)
plt.show()
plt.close()


# ===============================
# 6. 그래프 ③ 모든 날씨 유형 boxplot 
# ===============================
box_df = daily.pivot(
    columns="Weather_Category",
    values="Accident_Count"
)

print(box_df.head())

plt.figure(figsize=(8,5))

ax = box_df.plot(
    kind="box",
    grid=True,
    patch_artist=True   # ← 색 채우기 핵심
)

# 박스 색상 지정 (컬럼 순서대로)
colors = ["#A7C7E7", "#C1E1C1", "#FFD966", "#F4A6A6", "#D5B6E8"]

for patch, color in zip(ax.artists, colors):
    patch.set_facecolor(color)

plt.title("Accident Counts by Weather Condition")
plt.xlabel("Weather Category")
plt.ylabel("Accident Count")
plt.tight_layout()
plt.savefig(f"{RESULT_DIR}/boxplot_accident_by_weather.png", dpi=300)
plt.show()
plt.close()

# ===============================
# 7.사고 건수
# ===============================

fig = plt.figure(figsize=(15, 5))

ax1 = fig.add_subplot(1, 3, 1)
ax2 = fig.add_subplot(1, 3, 2)
ax3 = fig.add_subplot(1, 3, 3)

# Clear
sns.regplot(
    x="Visibility(mi)",
    y="Accident_Count",
    data=daily[daily["Weather_Category"] == "Clear"],
    ax=ax1,
    scatter_kws={"alpha": 0.4},
    line_kws={"color": "red"}
)
ax1.set_title("Clear Weather")
ax1.set_xlabel("Visibility (miles)")
ax1.set_ylabel("Accident Count")

# Rain
sns.regplot(
    x="Visibility(mi)",
    y="Accident_Count",
    data=daily[daily["Weather_Category"] == "Rain"],
    ax=ax2,
    scatter_kws={"alpha": 0.4},
    line_kws={"color": "red"}
)
ax2.set_title("Rainy Weather")
ax2.set_xlabel("Visibility (miles)")
ax2.set_ylabel("Accident Count")

# Snow
sns.regplot(
    x="Visibility(mi)",
    y="Accident_Count",
    data=daily[daily["Weather_Category"] == "Snow"],
    ax=ax3,
    scatter_kws={"alpha": 0.4},
    line_kws={"color": "red"}
)
ax3.set_title("Snowy Weather")
ax3.set_xlabel("Visibility (miles)")
ax3.set_ylabel("Accident Count")

plt.suptitle(
    "Relationship Between Visibility and Accident Count by Weather Condition",
    fontsize=14
)

plt.tight_layout()
plt.savefig(f"{RESULT_DIR}/regplot_visibility_by_weather_subplot.png", dpi=300)
plt.show()
plt.close()

# 8.모델 예측
daily = df.groupby("Date").agg({
    "Severity": "count",
    "Temperature(F)": "mean",
    "Humidity(%)": "mean",
    "Visibility(mi)": "mean",
    "Wind_Speed(mph)": "mean",
    "Precipitation(in)": "mean",
    "Weather_Category": lambda x: x.mode()[0]
}).reset_index()

daily.rename(columns={"Severity": "Accident_Count"}, inplace=True)

# ===============================
# 5.사고가 '많이 난 날' = 위험한 날
# ===============================
threshold = daily["Accident_Count"].median()

daily["High_Risk"] = (daily["Accident_Count"] >= threshold).astype(int)

print("High_Risk 분포")
print(daily["High_Risk"].value_counts())

# ===============================
# 6. 입력 변수 구성
# ===============================
X_numeric = daily[[
    "Temperature(F)",
    "Humidity(%)",
    "Visibility(mi)",
    "Wind_Speed(mph)",
    "Precipitation(in)"
]]

X_weather = pd.get_dummies(
    daily["Weather_Category"],
    drop_first=True   # 기준: Clear
)

X = pd.concat([X_numeric, X_weather], axis=1)
y = daily["High_Risk"]

# ===============================
# 7. 학습 / 테스트 분리
# ===============================
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# ===============================
# 8. LinearRegression 기반 위험도(확률) 예측
# ===============================
lin_reg = LinearRegression()
lin_reg.fit(X_train, y_train)

# 연속값 → 위험도 점수
y_pred_prob = lin_reg.predict(X_test)
y_pred_prob = np.clip(y_pred_prob, 0, 1)

print("\n예측 위험도(확률) 샘플:")
print(y_pred_prob[:10])

# ===============================
# 9. 임계값 기준 분류 성능 확인
# ===============================
y_pred_class = (y_pred_prob >= 0.5).astype(int)

print("\n[LinearRegression 기반 사고 위험도 모델]")
print("Accuracy:", accuracy_score(y_test, y_pred_class))

print("\nConfusion Matrix")
print(confusion_matrix(y_test, y_pred_class))

print("\nClassification Report")
print(classification_report(y_test, y_pred_class))

