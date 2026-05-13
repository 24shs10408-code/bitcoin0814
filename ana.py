import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime, timedelta
import os
import numpy as np
from sklearn.linear_model import LinearRegression

# 페이지 설정
st.set_page_config(
    page_title="Bitcoin Prediction Dashboard",
    page_icon="₿",
    layout="wide"
)

# 커스텀 스타일 적용
st.markdown("""
    <style>
    .main {
        background-color: #f8f9fa;
    }
    div[data-testid="stMetric"] {
        background-color: #ffffff;
        border: 1px solid #e9ecef;
        padding: 15px;
        border-radius: 10px;
    }
    .prediction-card {
        background-color: #ffffff;
        padding: 20px;
        border-radius: 15px;
        border-left: 5px solid #ffb000;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        margin-bottom: 20px;
    }
    </style>
    """, unsafe_allow_html=True)

@st.cache_data
def load_data(file_path):
    """
    coin.csv 파일을 로드하고 전처리합니다.
    """
    if not os.path.exists(file_path):
        return None

    df = pd.read_csv(file_path, sep=';')
    df['timeOpen'] = pd.to_datetime(df['timeOpen'])
    df = df.sort_values('timeOpen')
    
    # 등락 계산
    df['change_percent'] = df['close'].pct_change() * 100
    return df

def predict_next_day(df):
    """
    선형 회귀 모델을 사용하여 내일 가격을 예측합니다.
    """
    # 훈련 데이터 준비 (최근 60일 데이터 권장, 여기서는 전체 활용 가능)
    # 시간을 수치화 (Unix Timestamp)
    X = np.array(df['timeOpen'].map(datetime.timestamp)).reshape(-1, 1)
    y = df['close'].values

    # 모델 생성 및 학습
    model = LinearRegression()
    model.fit(X, y)

    # 내일 날짜 계산
    next_day = df['timeOpen'].max() + timedelta(days=1)
    next_timestamp = np.array([[datetime.timestamp(next_day)]])
    
    # 예측
    prediction = model.predict(next_timestamp)[0]
    return next_day, prediction

# 메인 타이틀
st.title("₿ 비트코인 가격 분석 및 예측 대시보드")

FILE_NAME = 'coin.csv'

try:
    df = load_data(FILE_NAME)

    if df is not None:
        # 사이드바: 필터 설정
        st.sidebar.header("⚙️ 분석 설정")
        min_date = df['timeOpen'].min().date()
        max_date = df['timeOpen'].max().date()
        
        selected_dates = st.sidebar.date_input(
            "분석 기간 선택",
            value=(min_date, max_date),
            min_value=min_date,
            max_value=max_date
        )

        # 데이터 필터링
        if isinstance(selected_dates, tuple) and len(selected_dates) == 2:
            start_date, end_date = selected_dates
            filtered_df = df[(df['timeOpen'].dt.date >= start_date) & 
                             (df['timeOpen'].dt.date <= end_date)].copy()
        else:
            filtered_df = df.copy()

        # --- 예측 섹션 (상단 배치) ---
        st.subheader("🚀 AI 내일 가격 예측 (Linear Regression)")
        next_date, pred_price = predict_next_day(filtered_df)
        current_price = filtered_df.iloc[-1]['close']
        diff = pred_price - current_price
        diff_pct = (diff / current_price) * 100
        
        direction = "상승" if diff > 0 else "하락"
        color = "#ff4b4b" if diff < 0 else "#00c853"

        st.markdown(f"""
            <div class="prediction-card">
                <h4 style='margin-top:0;'>{next_date.strftime('%Y년 %m월 %d일')} 예측 결과</h4>
                <p style='font-size: 1.2rem;'>예상 가격: <b>₩{pred_price:,.0f}</b></p>
                <p style='font-size: 1.1rem; color: {color};'>
                    현재가 대비 약 <b>{abs(diff_pct):.2f}% {direction}</b>할 것으로 전망됩니다.
                </p>
                <small>* 선형 회귀 모델 기반의 단순 예측이므로 투자 참고용으로만 활용하세요.</small>
            </div>
        """, unsafe_allow_html=True)

        # 주요 지표 (Metrics)
        latest = filtered_df.iloc[-1]
        m1, m2, m3, m4 = st.columns(4)
        with m1:
            prev_close = filtered_df.iloc[-2]['close'] if len(filtered_df) > 1 else latest['close']
            day_pct = ((latest['close'] - prev_close) / prev_close) * 100
            st.metric("현재 종가", f"₩{latest['close']:,.0f}", f"{day_pct:.2f}%")
        with m2:
            st.metric("기간 최고가", f"₩{filtered_df['high'].max():,.0f}")
        with m3:    
            st.metric("기간 최저가", f"₩{filtered_df['low'].min():,.0f}")
        with m4:
            mcap = latest['marketCap'] / 1e12
            st.metric("시가총액", f"₩{mcap:.2f}T")

        # 차트 영역
        tab1, tab2, tab3 = st.tabs(["🕯️ 가격 차트", "📊 거래량", "📑 데이터 시트"])

        with tab1:
            st.subheader("가격 추이 및 예측 선")
            fig = go.Figure()
            # 실제 가격
            fig.add_trace(go.Candlestick(
                x=filtered_df['timeOpen'],
                open=filtered_df['open'], high=filtered_df['high'],
                low=filtered_df['low'], close=filtered_df['close'],
                name="실제 가격"
            ))
            
            # 예측 포인트 추가
            fig.add_trace(go.Scatter(
                x=[filtered_df['timeOpen'].iloc[-1], next_date],
                y=[current_price, pred_price],
                mode='lines+markers',
                line=dict(color='orange', dash='dash'),
                name='내일 예측 연결선'
            ))

            fig.update_layout(template="plotly_white", xaxis_rangeslider_visible=False, height=500)
            st.plotly_chart(fig, use_container_width=True)

        with tab2:
            st.subheader("일별 거래량")
            vol_fig = go.Figure(data=[go.Bar(x=filtered_df['timeOpen'], y=filtered_df['volume'], marker_color='#636EFA')])
            vol_fig.update_layout(template="plotly_white", height=400)
            st.plotly_chart(vol_fig, use_container_width=True)

        with tab3:
            st.dataframe(filtered_df.sort_values('timeOpen', ascending=False), use_container_width=True)

    else:
        st.error(f"'{FILE_NAME}' 파일을 찾을 수 없습니다.")

except Exception as e:
    st.error(f"오류 발생: {e}")
