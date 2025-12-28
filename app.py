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
    .main-title { font-size: 36px; font-weight: bold; color: #1e3799; text-align: center; margin-bottom: 10px; }
    .sub-title { font-size: 18px; color: #4a69bd; text-align: center; margin-bottom: 30px; }
    .expert-note { background-color: #fff9db; padding: 15px; border-radius: 10px; border-left: 5px solid #fcc419; margin-bottom: 20px; font-size: 0.9rem; }
    .stMetric { background-color: #f1f2f6; padding: 15px; border-radius: 10px; border-left: 5px solid #1e3799; }
    
    /* ボタンを中央に配置 */
    .center-container { display: flex; justify-content: center; width: 100%; margin: 30px 0; }
    div.stButton > button {
        min-width: 350px !important; height: 65px !important; font-size: 20px !important;
        background: linear-gradient(135deg, #1e3799 0%, #0984e3 100%) !important;
        color: white !important; border-radius: 32px !important; border: none !important;
        box-shadow: 0 6px 15px rgba(30, 55, 153, 0.2) !important;
    }
    </style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-title">🏢 マンション投資の出口戦略</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">東京23区・全取引データ解析 AIシミュレーター</div>', unsafe_allow_html=True)

# --- 2. 23区データファイルのマッピング ---
area_files = {
    "千代田区": "千代田区中古マンション.csv", "中央区": "中央区中古マンション.csv", "港区": "港区中古マンション.csv",
    "新宿区": "新宿区中古マンション.csv", "文京区": "文京区中古マンション.csv", "台東区": "台東区中古マンション.csv",
    "墨田区": "墨田区中古マンション.csv", "江東区": "江東区中古マンション.csv", "品川区": "品川区中古マンション.csv",
    "目黒区": "目黒区中古マンション.csv", "大田区": "大田区中古マンション.csv", "世田谷区": "世田谷区中古マンション.csv",
    "渋谷区": "渋谷区中古マンション.csv", "中野区": "中野区中古マンション.csv", "杉並区": "杉並区中古マンション.csv",
    "豊島区": "豊島区中古マンション.csv", "北区": "北区中古マンション.csv", "荒川区": "荒川区中古マンション.csv",
    "板橋区": "板橋区中古マンション.csv", "練馬区": "練馬区中古マンション.csv", "足立区": "足立区中古マンション.csv",
    "葛飾区": "葛飾区中古マンション.csv", "江戸川区": "江戸川区中古マンション.csv"
}

# --- 3. 高速データ処理・学習エンジン ---
@st.cache_data
def load_and_preprocess(area):
    file_path = area_files.get(area)
    if not file_path or not os.path.exists(file_path):
        return None
    
    df = pd.read_csv(file_path)
    
    def to_num(x):
        if pd.isna(x): return np.nan
        nums = re.findall(r'\d+', str(x).replace(',', ''))
        return float(nums[0]) if nums else np.nan

    df['price'] = df['販売価格'].apply(to_num)
    df['area'] = df['専有面積'].apply(to_num)
    
    this_year = datetime.datetime.now().year
    df['age'] = df['築年月'].apply(lambda x: this_year - int(re.findall(r'\d{4}', str(x))[0]) 
                                  if re.findall(r'\d{4}', str(x)) else 20)
    
    df['walk'] = df['沿線・駅'].apply(lambda x: int(re.findall(r'歩(\d+)分', str(x))[0]) 
                                   if re.findall(r'歩(\d+)分', str(x)) else 10)
    
    return df[['price', 'area', 'age', 'walk']].dropna()

@st.cache_resource
def train_area_model(area):
    df_clean = load_and_preprocess(area)
    if df_clean is None or df_clean.empty:
        return None, None
    
    X = df_clean[['area', 'age', 'walk']]
    y = df_clean['price']
    
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X, y)
    return model, df_clean

# --- 4. ユーザーインターフェース (サイドバー) ---
st.sidebar.header("🔍 条件設定")
selected_area = st.sidebar.selectbox("分析する区を選択", list(area_files.keys()))

ai_model, train_data = train_area_model(selected_area)

if ai_model is not None:
    with st.sidebar:
        st.markdown("---")
        st.subheader("💰 投資パラメーター")
        p_price = st.number_input("購入価格 (万円)", value=int(train_data['price'].median()))
        p_rent = st.number_input("想定月額家賃 (円)", value=150000)
        
        st.markdown("---")
        st.subheader("📈 インフレ・変動設定")
        inflation_rate = st.slider("将来の想定インフレ率 (年 %)", 0.0, 3.0, 1.5)
        depreciation_rate = st.slider("建物の経年減価率 (年 %)", 0.0, 2.0, 0.8)

        st.markdown("---")
        st.subheader("🏢 物件スペック")
        s_area = st.number_input("専有面積 (㎡)", value=50.0)
        s_age = st.number_input("築年数 (購入時)", value=10)
        s_walk = st.number_input("駅徒歩 (分)", value=5)

    # --- 5. 実行ボタン ---
    st.markdown('<div class="center-container">', unsafe_allow_html=True)
    clicked = st.button("　AI出口戦略シミュレーションを実行　")
    st.markdown('</div>', unsafe_allow_html=True)

    if clicked:
        # --- 6. 出口シミュレーション計算 ---
        sim_years = 25
        results = []
        cumulative_rent = 0
        
        # 実質家賃成長率（インフレ率 - 経年減価率）
        net_rent_growth = (inflation_rate / 100) - (depreciation_rate / 100)
        
        for y in range(sim_years + 1):
            # A. AIによる将来の資産価値予測
            future_age = s_age + y
            # 特徴量名付きDFで予測し、AIに値上がり・値下がりを判断させる
            input_df = pd.DataFrame([[s_area, future_age, s_walk]], columns=['area', 'age', 'walk'])
            predicted_price = ai_model.predict(input_df)[0]
            
            # B. 実質家賃収入（インフレ相殺モデル：経費率20%想定）
            current_annual_rent = (p_rent * ((1 + net_rent_growth) ** y)) * 12 * 0.8 / 10000
            
            if y > 0:
                cumulative_rent += current_annual_rent
                
            # トータル損益計算
            total_return = (predicted_price + cumulative_rent) - p_price
            results.append({
                "年数": y, 
                "予測物件価格": predicted_price, 
                "累計家賃収入": cumulative_rent, 
                "トータル損益": total_return
            })

        res_df = pd.DataFrame(results)
        best_exit = res_df.loc[res_df['トータル損益'].idxmax()]

        # --- 7. 結果の可視化 ---
        st.info(f"✅ **{selected_area}** の市場データ（{len(train_data)}件）から、将来価値をAIが直接推論しました。")

        c1, c2, c3 = st.columns(3)
        c1.metric("推奨売却時期", f"{int(best_exit['年数'])}年後")
        c2.metric("最大回収利益", f"{int(best_exit['トータル損益']):,}万円")
        c3.metric("売却時のAI予測価格", f"{int(best_exit['予測物件価格']):,}万円")

        

        # グラフ作成
        fig = go.Figure()
        fig.add_trace(go.Bar(x=res_df['年数'], y=res_df['累計家賃収入'], name="累積家賃（インフレ相殺）", marker_color='rgba(52, 152, 219, 0.6)'))
        fig.add_trace(go.Scatter(x=res_df['年数'], y=res_df['予測物件価格'], name="物件価値（AI推論）", line=dict(color='#e67e22', width=2, dash='dot')))
        fig.add_trace(go.Scatter(x=res_df['年数'], y=res_df['トータル損益'], name="トータル損益", line=dict(color='#27ae60', width=4)))
        
        fig.update_layout(
            title=f"【{selected_area}】保有期間別収益予測（インフレ率{inflation_rate}%想定）",
            xaxis_title="保有年数（年）", yaxis_title="金額（万円）",
            hovermode="x unified", template="plotly_white",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        st.plotly_chart(fig, use_container_width=True)

        # 専門家AIインサイト
        st.markdown("### 🤖 AI出口診断レポート")
        price_diff = res_df['予測物件価格'].iloc[10] - p_price
        trend_status = "上昇傾向" if price_diff > 0 else "緩やかな下落傾向"
        
        st.write(f"""
        - **価格トレンド:** AIは{selected_area}の統計から、今後10年間でこの物件が **{trend_status}** になると予測しました。
        - **インフレ耐性:** 想定インフレ率{inflation_rate}%に対し、実質賃料成長率は{net_rent_growth*100:.1f}%です。累積家賃が資産価値の変動をカバーする構造になっています。
        - **投資効率:** 利益が最大化される **{int(best_exit['年数'])}年後** が最も効率的な出口ですが、損益分岐点を超える **{res_df[res_df['トータル損益'] > 0]['年数'].min() if not res_df[res_df['トータル損益'] > 0].empty else '－'}年目** 以降であれば、現金化の選択肢が入ります。
        """)

else:
    st.error(f"エラー: {selected_area} のデータファイルが見つかりません。CSVファイルを確認してください。")
