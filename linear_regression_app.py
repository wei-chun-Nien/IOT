
import streamlit as st
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics
import matplotlib.pyplot as plt

# --- 1. 商業理解 (Business Understanding) ---
# 目標：建立一個互動式網頁應用程式，讓使用者了解線性回歸模型如何根據不同的參數（斜率、雜訊、資料量）進行擬合。
# 使用者可以透過調整這些參數，直觀地看到資料分佈和模型擬合結果的變化。

st.set_page_config(layout="wide")

st.title("CRISP-DM 流程：互動式線性回歸模型")
st.write("這個應用程式遵循 CRISP-DM 流程，展示一個簡單的線性回歸模型。您可以調整左側的參數來觀察模型的變化。")

# --- 2. 資料理解 (Data Understanding) ---
# 在這個階段，我們將生成並探索我們的資料。
# 資料是基於一個簡單的線性方程式 y = ax + b 加上一些隨機雜訊生成的。

st.sidebar.header("調整參數")
st.sidebar.write("透過調整以下參數來生成新的資料集並重新訓練模型。")

# 使用者可調整的參數
a_param = st.sidebar.slider("斜率 (a)", min_value=0.1, max_value=10.0, value=2.0, step=0.1)
noise_param = st.sidebar.slider("雜訊 (Noise)", min_value=0.0, max_value=50.0, value=10.0, step=1.0)
n_points_param = st.sidebar.slider("資料點數量 (Number of data points)", min_value=50, max_value=1000, value=200, step=50)
b_param = 5 # 我們將截距 b 設為一個常數

# --- 3. 資料準備 (Data Preparation) ---
# 這個階段我們將生成資料並將其轉換為適合模型的格式。

# 生成資料
@st.cache_data
def generate_data(a, b, n_points, noise):
    """
    根據給定的參數生成線性資料。
    - a: 斜率
    - b: 截距
    - n_points: 資料點數量
    - noise: 雜訊標準差
    """
    X = np.random.rand(n_points, 1) * 10  # X 值範圍在 0 到 10 之間
    # 根據 y = ax + b 生成 y，並加入常態分佈的雜訊
    y = a * X.squeeze() + b + np.random.normal(loc=0, scale=noise, size=n_points)
    df = pd.DataFrame({'X': X.squeeze(), 'y': y})
    return df

data_df = generate_data(a_param, b_param, n_points_param, noise_param)

# 將資料分割為特徵 (X) 和目標 (y)
X = data_df[['X']]
y = data_df['y']

# 將資料分割為訓練集和測試集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# 在網頁上展示資料
st.header("1 & 2. 資料理解與準備")
col1, col2 = st.columns([1, 2])

with col1:
    st.write("#### 生成的資料")
    st.write("我們根據您設定的參數生成了以下資料。")
    st.dataframe(data_df.head())
    st.write(f"總共生成了 **{len(data_df)}** 筆資料。")
    st.write(f"訓練集大小: **{len(X_train)}**")
    st.write(f"測試集大小: **{len(X_test)}**")

with col2:
    st.write("#### 資料分佈圖")
    fig, ax = plt.subplots()
    ax.scatter(data_df['X'], data_df['y'], alpha=0.7, label="生成的資料點")
    ax.set_xlabel("X")
    ax.set_ylabel("y")
    ax.set_title("資料視覺化")
    ax.legend()
    ax.grid(True)
    st.pyplot(fig)


# --- 4. 模型建立 (Modeling) ---
# 在這個階段，我們選擇並訓練一個線性回歸模型。

st.header("3. 模型建立")
st.write("我們使用 Scikit-learn 的 `LinearRegression` 模型來擬合訓練資料。")

# 建立並訓練模型
model = LinearRegression()
model.fit(X_train, y_train)

st.write("模型訓練完成！")
st.write(f"模型找到的斜率 (coefficient): **{model.coef_[0]:.2f}**")
st.write(f"模型找到的截距 (intercept): **{model.intercept_:.2f}**")


# --- 5. 模型評估 (Evaluation) ---
# 在這個階段，我們使用測試集來評估模型的表現。

st.header("4. 模型評估")
st.write("我們使用訓練好的模型對「測試集」進行預測，並計算評估指標。")

# 進行預測
y_pred = model.predict(X_test)

# 計算評估指標
mae = metrics.mean_absolute_error(y_test, y_pred)
mse = metrics.mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = metrics.r2_score(y_test, y_pred)

col1_eval, col2_eval = st.columns(2)
with col1_eval:
    st.write("#### 評估指標")
    st.write(f"平均絕對誤差 (MAE): **{mae:.2f}**")
    st.write(f"均方誤差 (MSE): **{mse:.2f}**")
    st.write(f"均方根誤差 (RMSE): **{rmse:.2f}**")
    st.write(f"R-squared (R²): **{r2:.2f}**")

with col2_eval:
    st.write("#### 實際值 vs. 預測值")
    eval_df = pd.DataFrame({'實際值 (Actual)': y_test, '預測值 (Predicted)': y_pred})
    st.dataframe(eval_df.head())


# --- 6. 部署 (Deployment) ---
# 這個 Streamlit 應用程式本身就是一個部署的例子。
# 我們將模型的結果視覺化，讓使用者可以直觀地看到模型的擬合情況。

st.header("5. 部署與視覺化")
st.write("最後，我們將原始資料點和模型擬合出的迴歸線畫在一起，來視覺化模型的表現。")

fig_final, ax_final = plt.subplots(figsize=(10, 6))

# 繪製所有資料點
ax_final.scatter(data_df['X'], data_df['y'], alpha=0.6, label="所有資料點")

# 繪製迴歸線
# 為了繪製平滑的線，我們在 X 的最小值和最大值之間生成一些點
line_X = np.linspace(X.min(), X.max(), 100).reshape(-1, 1)
line_y = model.predict(line_X)
ax_final.plot(line_X, line_y, color='red', linewidth=3, label="線性迴歸線")

# 繪製原始的真實直線 (用於比較)
true_line_y = a_param * line_X.squeeze() + b_param
ax_final.plot(line_X, true_line_y, color='green', linestyle='--', linewidth=2, label=f"真實直線 (y={a_param:.1f}x+{b_param})")


ax_final.set_xlabel("X")
ax_final.set_ylabel("y")
ax_final.set_title("線性回歸模型擬合結果")
ax_final.legend()
ax_final.grid(True)

st.pyplot(fig_final)

st.info("""
**如何解讀這個應用程式：**
1.  **左側參數調整：**
    *   **斜率 (a):** 改變真實直線的斜率。
    *   **雜訊 (Noise):** 增加或減少資料點的隨機性。雜訊越大，資料點越分散，模型越難擬合。
    *   **資料點數量:** 改變資料集的規模。
2.  **資料視覺化：** 觀察您生成的資料分佈。
3.  **模型評估指標：**
    *   **R-squared (R²):** 介於 0 和 1 之間，越接近 1 表示模型解釋力越強。
    *   **MSE/RMSE:** 誤差指標，越小表示模型的預測越準確。
4.  **最終圖表：**
    *   **藍色點** 是您生成的資料。
    *   **綠色虛線** 是資料生成時所依據的「真實」數學公式直線。
    *   **紅色實線** 是我們的線性回歸模型根據藍色點「學習」到的擬合線。
    *   觀察紅色線如何隨著參數的變化去逼近綠色線。
""")
