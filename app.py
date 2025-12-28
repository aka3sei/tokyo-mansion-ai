import streamlit as st
import pandas as pd
import pickle
import lightgbm as lgb

# --- 1. データ定義（town_data, ku_market_data, rent_factor は維持） ---
# (前述の膨大なリストをここに配置してください)

# --- 2. ページ設定とスタイル ---
st.set_page_config(page_title="23区マンションAI査定", layout="centered")

st.markdown("""
    <style>
    header[data-testid="stHeader"] { visibility: hidden; display: none; }
    footer { visibility: hidden; }
    .block-container { padding-top: 2rem !important; padding-bottom: 7rem !important; }
    .stApp { background-color: #f8f9fa; }
    .center-container { display: flex; justify-content: center; width: 100%; margin: 40px 0; }
    div.stButton { text-align: center; }
    div.stButton > button {
        min-width: 340px !important; height: 60px !important; font-size: 26px !important;
        font-weight: bold !important; background: linear-gradient(135deg, #ff4b4b 0%, #ff7575 100%) !important;
        color: white !important; border-radius: 40px !important;
        box-shadow: 0 8px 20px rgba(255, 75, 75, 0.3) !important; border: none !important;
    }
    .up-card {
        background: linear-gradient(135deg, #fff5f5 0%, #ffffff 100%);
        padding: 20px; border-radius: 15px; border: 2px solid #ff7575;
        text-align: center; box-shadow: 0 4px 15px rgba(255, 75, 75, 0.1);
    }
    .up-label { color: #ff4b4b; font-size: 1.1rem; font-weight: bold; margin-bottom: 10px; }
    .up-price { color: #ff4b4b; font-size: 1.8rem; font-weight: bold; }
    .stable-card {
        background-color: #ffffff; padding: 20px; border-radius: 15px;
        border: 1px solid #e0e0e0; text-align: center;
    }
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
st.caption("AIが膨大な取引データから、将来の「価値向上」の可能性を分析します。")

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

# --- 5. 査定実行ボタン ---
st.markdown('<div class="center-container">', unsafe_allow_html=True)
clicked = st.button("　　AI査定を実行する　　")
st.markdown('</div>', unsafe_allow_html=True)

# --- 6. 査定ロジックと結果表示 ---
if clicked:
    full_address = f"東京都{selected_ku}{selected_loc}"
    
    def predict_price(y_offset):
        input_df = pd.DataFrame([{
            '区': selected_ku, '所在': full_address, '専有面積': area, 
            '駅より徒歩': walk, '築年月': year_now - y_offset 
        }])
        input_df['区'] = input_df['区'].astype('category')
        input_df['所在'] = input_df['所在'].astype('category')
        return model.predict(input_df)[0]

    try:
        price_now = predict_price(0)
        price_5y = predict_price(5)
        price_10y = predict_price(10)

        st.divider()
        st.balloons()
        st.subheader(f"📊 査定結果: {selected_ku} {selected_loc}")
        
        # メイン現在価格
        st.metric("AI査定価格（現在）", f"{round(price_now):,} 万円")

        # --- 将来予測の条件分岐表示 ---
        st.write("📈 **AI将来価値インサイト**")
        
        # 5年後判定
        if price_5y > price_now:
            st.markdown(f"""<div class="up-card">
                <div class="up-label">🚀 5年後のさらなる価値向上予測</div>
                <div class="up-price">{round(price_5y):,} 万円</div>
                <div style="font-size:0.9rem; color:#ff4b4b;">AIはこのエリアの希少性が経年減価を上回ると予測しています</div>
            </div>""", unsafe_allow_html=True)
        else:
            st.info("✅ **5年後の見通し**: このエリアは高い流動性を維持しており、資産としての安定性が極めて高いと分析されました。")

        # 10年後判定
        if price_10y > price_now:
            st.markdown(f"""<div style="margin-top:15px;" class="up-card">
                <div class="up-label">🌟 10年後のプレミアム価格予測</div>
                <div class="up-price">{round(price_10y):,} 万円</div>
                <div style="font-size:0.9rem; color:#ff4b4b;">長期にわたり「ヴィンテージ」として価値を確立するポテンシャルがあります</div>
            </div>""", unsafe_allow_html=True)
        else:
            st.success("✅ **長期資産性**: 築年数が経過しても、{selected_ku}のブランド力が強固な支えとなり、着実な資産防衛が期待できます。")

        st.divider()
        st.subheader(f"🏙️ {selected_ku}のマーケット詳細分析")
        # (以下、マーケット分析の表示ロジック)
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
