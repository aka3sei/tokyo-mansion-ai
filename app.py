import streamlit as st
import pandas as pd
import pickle
import lightgbm as lgb

# ページ設定
st.set_page_config(page_title="23区マンションAI査定", layout="centered")

st.title("🏙️ 東京23区マンション AI査定システム")
st.write("AIが統計データから「ベース価格」を算出し、ブランド価値を含めた「期待レンジ」を表示します。")

# モデルの読み込み
@st.cache_resource
def load_model():
    # ローカルで作成した satei_model.pkl を読み込みます
    with open('satei_model.pkl', 'rb') as f:
        return pickle.load(f)

try:
    model = load_model()
except Exception as e:
    st.error(f"モデルの読み込みに失敗しました。'satei_model.pkl' が同じフォルダにあるか確認してください。")

# エリア別賃料単価係数
rent_factor = {
    '港区': 1.85, '千代田区': 1.85, '中央区': 1.75, '渋谷区': 1.75, '新宿区': 1.65,
    '文京区': 1.55, '目黒区': 1.55, '豊島区': 1.45, '台東区': 1.45, '品川区': 1.45,
    '世田谷区': 1.35, '杉並区': 1.25, '中野区': 1.35, '江東区': 1.35, '練馬区': 1.10,
    '墨田区': 1.20, '北区': 1.15, '荒川区': 1.15, '板橋区': 1.05, '大田区': 1.15,
    '足立区': 0.90, '葛飾区': 0.85, '江戸川区': 0.85
}

# 入力フォーム
with st.form("assessment_form"):
    col1, col2 = st.columns(2)
    with col1:
        ku = st.selectbox("1. 区を選択", list(rent_factor.keys()))
        loc = st.text_input("2. 町名を入力", "白金台３丁目")
    with col2:
        area = st.number_input("3. 専有面積 (㎡)", value=70.0, step=0.1)
        walk = st.number_input("4. 駅徒歩 (分)", value=5, step=1)
    
    year_now = st.number_input("5. 築年 (西暦)", value=2015, step=1)
    
    submit_button = st.form_submit_button("AI査定を実行")

if submit_button:
    # 推論用データ作成（ブランド物件フラグは0=一般としてベースを算出）
    input_df = pd.DataFrame({
        '区': [ku], '所在': [loc], '専有面積': [area], 
        '駅より徒歩': [walk], '築年月': [year_now], 'ブランド物件': [0]
    })
    
    # カテゴリ型に変換
    for col in ['区', '所在']:
        input_df[col] = input_df[col].astype('category')
    
    # 1. 統計ベース価格の予測
    price_base = model.predict(input_df)[0]
    
    # 2. 5年後価格予測 (築年を5年分引く)
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

    # --- 結果の表示 ---
    st.divider()
    st.subheader(f"📊 診断結果: {ku} {loc}")
    
    # メイン指標
    m1, m2 = st.columns(2)
    m1.metric("AI統計ベース価格", f"{price_base:,.0f} 万円")
    m2.metric("AI予想利回り", f"{yield_rate:.2f} %")
    
    # ブランド期待レンジ
    st.info(f"✨ **ブランド期待価格レンジ**: {price_base*1.1:,.0f} ～ {price_base*1.2:,.0f} 万円")
    st.caption("※大手デベロッパー物件（パークハウス・プラウド等）の場合の期待値です。")
    
    st.write("---")
    
    # 将来予測
    c1, c2 = st.columns(2)
    with c1:
        st.write("📉 **5年後の予想(ベース)**")
        st.write(f"### {price_future:,.0f} 万円")
        st.caption(f"（現在比: ▲{price_base - price_future:,.0f} 万円）")
    with c2:
        st.write("🏁 **5年後の実質収支**")
        total_profit = annual_rent_man * 5 - (price_base - price_future)
        st.write(f"### {total_profit:,.0f} 万円")
        st.caption("（5年間の賃料収入 － 値下がり損）")

    if total_profit > 0:
        st.success("🟢 判定: 賃料収入が下落を上回る資産防衛力の高い物件です。")
    else:
        st.warning("🟡 判定: 下落幅が賃料収入を上回る可能性があります。")