import streamlit as st
import pandas as pd
import pickle
import lightgbm as lgb

# 診断結果のセクション
    st.divider()
    st.subheader(f"📊 診断結果: {ku} {loc}")

    # メトリクス（価格や利回り）の表示のあとに...
    
    st.info(f"🏙️ **{ku}のエリア分析レポート**")
    # 改行を含むテキストをきれいに表示
    st.write(ku_details.get(ku, "データ準備中"))

# ページ設定
st.set_page_config(page_title="23区マンションAI査定", layout="centered")

st.title("🏙️ 東京23区マンション AI査定システム")
st.write("AIが統計データから「ベース価格」を算出し、ブランド価値を含めた「期待レンジ」を表示します。")

# モデルの読み込み
@st.cache_resource
def load_model():
    with open('satei_model.pkl', 'rb') as f:
        return pickle.load(f)

try:
    model = load_model()
except Exception as e:
    st.error(f"モデルの読み込みに失敗しました。'satei_model.pkl' があるか確認してください。")

# エリア別賃料単価係数
rent_factor = {
    '港区': 1.85, '千代田区': 1.85, '中央区': 1.75, '渋谷区': 1.75, '新宿区': 1.65,
    '文京区': 1.55, '目黒区': 1.55, '豊島区': 1.45, '台東区': 1.45, '品川区': 1.45,
    '世田谷区': 1.35, '中野区': 1.35, '杉並区': 1.30, '江東区': 1.30, '大田区': 1.25,
    '墨田区': 1.20, '荒川区': 1.15, '北区': 1.15, '練馬区': 1.10, '板橋区': 1.10,
    '江戸川区': 1.05, '足立区': 1.00, '葛飾区': 1.00
}

# --- 入力フォーム ---
with st.container():
    col1, col2 = st.columns(2)
    with col1:
        ku = st.selectbox("区を選択", list(rent_factor.keys()))
        loc = st.text_input("所在地（例：南青山、勝どき）", "芝浦")
    with col2:
        area = st.number_input("専有面積 (㎡)", min_value=10.0, max_value=300.0, value=60.0, step=1.0)
        walk = st.slider("駅より徒歩 (分)", 0, 30, 5)

    year_now = st.number_input("築年月 (西暦)", min_value=1970, max_value=2025, value=2015)

if st.button("AI査定を実行する"):
    # 推論用データの作成
    input_df = pd.DataFrame([{
        '区': ku,
        '所在': f"東京都{ku}{loc}",
        '専有面積': area,
        '駅より徒歩': walk,
        '築年月': year_now
    }])

    # --- 【修正ポイント】型を明示的にカテゴリー型に変換 ---
    input_df['区'] = input_df['区'].astype('category')
    input_df['所在'] = input_df['所在'].astype('category')

    # 1. 現在価格予測
    price_base = model.predict(input_df)[0]
    
    # 2. 5年後価格予測
    input_future = input_df.copy()
    input_future['築年月'] = year_now - 5
    price_future = model.predict(input_future)[0]
    
    # 3. 賃料・利回り計算
    f = rent_factor.get(ku, 1.0)
    age_effect = max(0.65, 1.0 - (max(0, 2025 - year_now) * 0.008))
    m2_rent = 3300 * f * age_effect
    monthly_rent = m2_rent * area
    annual_rent_man = (monthly_rent * 12) / 10000
    yield_rate = (annual_rent_man / price_base) * 100

    # 結果表示
    st.divider()
    st.subheader(f"📊 診断結果: {ku} {loc}")
    
    m1, m2 = st.columns(2)
    m1.metric("AI統計ベース価格", f"{price_base:,.0f} 万円")
    m2.metric("AI予想利回り", f"{yield_rate:.2f} %")
    
    st.info(f"✨ **ブランド期待価格レンジ**: {int(price_base*0.95):,} 〜 {int(price_base*1.2):,} 万円")
    st.write(f"💡 5年後の予想価格: **{price_future:,.0f} 万円**")

    # 診断結果のセクション
    st.divider()
    st.subheader(f"📊 診断結果: {ku} {loc}")

    # メトリクス（価格や利回り）の表示のあとに...
    
    st.info(f"🏙️ **{ku}のエリア分析レポート**")
    # 改行を含むテキストをきれいに表示
    st.write(ku_details.get(ku, "データ準備中"))
