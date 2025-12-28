import streamlit as st
import pandas as pd
import pickle
import lightgbm as lgb

# --- 1. データ定義（内容は維持） ---
rent_factor = {
    '千代田区': 1.25, '中央区': 1.18, '港区': 1.35, '新宿区': 1.10, '文京区': 1.05,
    '台東区': 1.00, '墨田区': 0.95, '江東区': 1.02, '品川区': 1.08, '目黒区': 1.15,
    '大田区': 0.92, '世田谷区': 1.03, '渋谷区': 1.20, '中野区': 0.98, '杉並区': 0.96,
    '豊島区': 1.02, '北区': 0.90, '荒川区': 0.88, '板橋区': 0.87, '練馬区': 0.86,
    '足立区': 0.82, '葛飾区': 0.80, '江戸川区': 0.83
}
# ※ town_data, ku_market_data は前回同様のため省略（お手元のファイルを維持してください）
# ... (town_data の定義)
# ... (ku_market_data の定義)

# --- 2. ページ設定とスタイル ---
st.set_page_config(page_title="23区マンションAI査定", layout="centered")

st.markdown("""
    <style>
    .stApp { background-color: #f8f9fa; }
    
    /* ボタンを強制的に中央に配置する親コンテナ */
    .center-container {
        display: flex;
        justify-content: center;
        width: 100%;
        margin: 30px 0; /* 上下の余白を少し詰めました */
    }

    /* ボタン自体のデザイン：少しだけサイズダウン */
    div.stButton > button {
        display: inline-block;
        width: auto !important;
        min-width: 320px !important; /* 380px -> 320px に縮小 */
        height: 65px !important;     /* 80px -> 65px に縮小 */
        font-size: 20px !important;   /* 26px -> 20px に縮小 */
        font-weight: bold !important;
        background: linear-gradient(135deg, #ff4b4b 0%, #ff7575 100%) !important;
        color: white !important;
        border-radius: 32px !important;
        box-shadow: 0 6px 15px rgba(255, 75, 75, 0.2) !important;
        border: none !important;
        transition: all 0.3s ease;
        padding: 0 45px !important; /* 左右の余白も少し調整 */
    }
    
    div.stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 10px 20px rgba(255, 75, 75, 0.3) !important;
    }

    /* マーケットカード */
    .market-card {
        background-color: white; padding: 20px; border-radius: 15px;
        border-left: 5px solid #ff4b4b; box-shadow: 0 4px 15px rgba(0,0,0,0.05);
        height: 160px; margin-bottom: 20px;
    }
    .market-title { font-weight: bold; color: #ff4b4b; margin-bottom: 10px; font-size: 1.1rem; }
    .market-content { font-size: 0.95rem; color: #333; line-height: 1.6; }
    </style>
    """, unsafe_allow_html=True)

# --- 3. モデル読み込み ---
@st.cache_resource
def load_model():
    with open('satei_model.pkl', 'rb') as f:
        return pickle.load(f)

model = load_model()

# --- 4. 入力フォーム ---
st.title("🏙️ 東京23区マンション AI査定")
st.caption("AIが最新の市場データに基づき、あなたのマンションの価値を瞬時に算出します。")

with st.container():
    col1, col2 = st.columns(2)
    with col1:
        selected_ku = st.selectbox("区を選択", list(ku_market_data.keys()))
        town_options = town_data.get(selected_ku, ["その他"])
        selected_loc = st.selectbox("所在地（町名）を選択", town_options)
        
    with col2:
        area = st.number_input("専有面積 (㎡)", min_value=10, max_value=300, value=60, step=1, format="%d")
        walk = st.slider("駅より徒歩 (分)", 0, 30, 5)
    
    year_now = st.number_input("築年月 (西暦)", min_value=1970, max_value=2025, value=2015, step=1, format="%d")

# --- 5. 査定実行ボタン（少しだけコンパクトな中央配置） ---
st.write("") 
st.markdown('<div class="center-container">', unsafe_allow_html=True)
# テキストの空白を調整し、スッキリさせました
clicked = st.button("　AI査定を実行する　") 
st.markdown('</div>', unsafe_allow_html=True)

# --- 6. 査定ロジックと結果表示 ---
if clicked:
    full_address = f"東京都{selected_ku}{selected_loc}"
    input_df = pd.DataFrame([{
        '区': selected_ku, '所在': full_address, '専有面積': area, 
        '駅より徒歩': walk, '築年月': year_now
    }])
    input_df['区'] = input_df['区'].astype('category')
    input_df['所在'] = input_df['所在'].astype('category')
    
    try:
        price_base = model.predict(input_df)[0]
        
        # 駅近逆転現象の簡易補正
        if walk <= 5:
            bonus = (6 - walk) * 0.015
            price_base = price_base * (1 + bonus)
        
        st.divider()
        st.balloons() 
        st.subheader(f"📊 査定結果: {selected_ku} {selected_loc}")
        
        m1, m2 = st.columns(2)
        m1.metric("AI統計ベース価格", f"{round(price_base):,} 万円")
        
        f = rent_factor.get(selected_ku, 1.0)
        age_effect = max(0.65, 1.0 - (max(0, 2025 - year_now) * 0.008))
        m2_rent = 3300 * f * age_effect
        annual_rent_man = (m2_rent * area * 12) / 10000
        yield_rate = (annual_rent_man / price_base) * 100
        m2.metric("AI予想利回り", f"{yield_rate:.2f} %")
        
        st.success(f"✨ **ブランド期待価格レンジ**: {round(price_base):,} 〜 {round(price_base*1.25):,} 万円")

        # --- マーケット分析 ---
        st.divider()
        st.subheader(f"🏙️ {selected_ku}のマーケット詳細分析")
        
        data = ku_market_data.get(selected_ku)
        mc1, mc2 = st.columns(2)
        with mc1:
            st.markdown(f'<div class="market-card"><div class="market-title">📍 特徴</div><div class="market-content">{data["特徴"]}</div></div>', unsafe_allow_html=True)
            st.markdown(f'<div class="market-card"><div class="market-title">🏢 ブランド</div><div class="market-content">{data["ブランド"]}</div></div>', unsafe_allow_html=True)
        with mc2:
            st.markdown(f'<div class="market-card"><div class="market-title">🗺️ 人気エリア</div><div class="market-content">{data["人気"]}</div></div>', unsafe_allow_html=True)
            st.markdown(f'<div class="market-card"><div class="market-title">🏗️ 開発・将来性</div><div class="market-content">{data["開発"]}</div></div>', unsafe_allow_html=True)

    except Exception as e:
        st.error(f"エラーが発生しました: {e}")
