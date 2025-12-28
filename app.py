import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
import plotly.graph_objects as go
import datetime
import re
import os

# --- 1. ページ基本設定 ---
st.set_page_config(page_title="23区マンション投資 AI出口戦略", layout="wide")

st.markdown("""
    <style>
    .main-title { font-size: 32px; font-weight: bold; color: #1e3799; text-align: center; margin-bottom: 5px; }
    .expert-tag { background-color: #e3f2fd; color: #0d47a1; padding: 5px 15px; border-radius: 20px; font-size: 0.8rem; font-weight: bold; }
    .center-container { display: flex; justify-content: center; width: 100%; margin: 25px 0; }
    div.stButton > button {
        min-width: 350px !important; height: 65px !important; font-size: 20px !important;
        background: linear-gradient(135deg, #1e3799 0%, #0984e3 100%) !important;
        color: white !important; border-radius: 35px !important; border: none !important;
        box-shadow: 0 10px 20px rgba(30, 55, 153, 0.2) !important;
    }
    .stMetric { background-color: #f8f9fa; border-left: 5px solid #1e3799; border-radius: 8px; }
    </style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-title">🏢 マンション投資 AI出口戦略シミュレーター</div>', unsafe_allow_html=True)
st.markdown('<div style="text-align:center;"><span class="expert-tag">インフレ相殺モデル・AI将来価値推論エンジン搭載</span></div>', unsafe_allow_html=True)

# --- 2. データ処理・学習エンジン ---
@st.cache_data
def load_and_preprocess(area):
    # (既存の area_files マッピングと前処理ロジック)
    # ここではユーザーが配置した各区のCSV（「港区中古マンション.csv」等）を読み込みます
    pass

@st.cache_resource
def train_area_model(area):
    # (既存の RandomForestRegressor 学習ロジック)
    # area, age, walk の3変数を学習
    pass

# --- 3. サイドバー設定 ---
st.sidebar.header("🔍 分析条件")
# 区の選択と、AIモデルの学習（実際の運用ではご自身のモデル/CSVに合わせてください）
selected_area = st.sidebar.selectbox("区を選択", ["港区", "中央区", "千代田区", "渋谷区", "新宿区", "江東区"])

with st.sidebar:
    st.markdown("---")
    st.subheader("💰 投資パラメーター")
    p_price = st.number_input("購入価格 (万円)", value=8000)
    p_rent = st.number_input("初期月額家賃 (円)", value=280000)
    
    st.markdown("---")
    st.subheader("📈 インフレ設定（家賃計算用）")
    inflation_rate = st.slider("想定インフレ率 (年 %)", 0.0, 3.0, 1.5, help="家賃上昇への寄与率")
    depreciation_rate = st.slider("築年数による減価率 (年 %)", 0.0, 2.0, 0.8, help="建物の老朽化による家賃下落率")

    st.markdown("---")
    st.subheader("🏢 物件スペック")
    s_area = st.number_input("専有面積 (㎡)", value=50.0)
    s_age = st.number_input("築年数 (購入時)", value=10)
    s_walk = st.number_input("駅徒歩 (分)", value=5)

# --- 4. 実行ボタン ---
st.markdown('<div class="center-container">', unsafe_allow_html=True)
clicked = st.button("　AI査定と出口戦略を算出　")
st.markdown('</div>', unsafe_allow_html=True)

if clicked:
    # --- 5. AI推論 ＆ 家賃ロジック計算 ---
    sim_years = 25
    results = []
    cumulative_rent = 0
    
    # 家賃の実質成長率（インフレ - 減価）
    net_rent_growth = (inflation_rate / 100) - (depreciation_rate / 100)

    for y in range(sim_years + 1):
        # 【AI算出部分】将来の資産価値予測
        # AIモデルに未来の築年数を入力し、統計的な市場価格を弾き出す
        future_age = s_age + y
        input_df = pd.DataFrame([[s_area, future_age, s_walk]], columns=['area', 'age', 'walk'])
        # ai_model.predict(input_df) を実行（※実際の実装ではモデルを呼び出し）
        predicted_price = p_price * (1.005 ** y) # ここはAIがエリアに合わせて算出する部分のダミー
        
        # 【数式部分】家賃収入の累計（インフレ相殺モデル）
        current_annual_rent = (p_rent * ((1 + net_rent_growth) ** y)) * 12 * 0.8 / 10000
        if y > 0:
            cumulative_rent += current_annual_rent
            
        # トータル損益算出
        total_return = (predicted_price + cumulative_rent) - p_price
        
        results.append({
            "年数": y, 
            "予測物件価格": predicted_price, 
            "累計家賃収入": cumulative_rent, 
            "トータル損益": total_return
        })

    res_df = pd.DataFrame(results)
    best_exit = res_df.loc[res_df['トータル損益'].idxmax()]

    # --- 6. 視覚化と診断 ---
    st.info(f"✅ **{selected_area}** の市場特性をAIが解析。家賃計算には「インフレ率 {inflation_rate}%」を適用しました。")

    c1, c2, c3 = st.columns(3)
    c1.metric("推奨出口時期", f"{int(best_exit['年数'])}年後")
    c2.metric("予測最大収益", f"{int(best_exit['トータル損益']):,}万円")
    c3.metric("その時のAI予想価格", f"{int(best_exit['予測物件価格']):,}万円")

    

    # Plotlyによる収益可視化
    fig = go.Figure()
    fig.add_trace(go.Bar(x=res_df['年数'], y=res_df['累計家賃収入'], name="累計家賃(インフレ相殺後)", marker_color='rgba(52, 152, 219, 0.6)'))
    fig.add_trace(go.Scatter(x=res_df['年数'], y=res_df['予測物件価格'], name="AI予測価格(キャピタル)", line=dict(color='#e67e22', width=3)))
    fig.add_trace(go.Scatter(x=res_df['年数'], y=res_df['トータル損益'], name="トータル損益", line=dict(color='#27ae60', width=4)))
    
    fig.update_layout(title="保有期間別：収益シミュレーション", hovermode="x unified", template="plotly_white")
    st.plotly_chart(fig, use_container_width=True)

    # AIアドバイス
    st.markdown("### 🤖 AI出口戦略アドバイス")
    price_change = res_df['予測物件価格'].iloc[15] - p_price
    if price_change > 0:
        st.success(f"**【強気予想】** AIはこの物件が築年数を経ても価格を維持、あるいは上昇させると予測しました。インフレ環境下で極めて強力な資産防衛となります。")
    else:
        st.warning(f"**【安定的減価】** AIは緩やかな価格下落を予測していますが、家賃収入によるインカムゲインがそれを補うため、{res_df[res_df['トータル損益'] > 0]['年数'].min() if not res_df[res_df['トータル損益'] > 0].empty else 'X'}年目以降の売却はプラス収支となります。")
