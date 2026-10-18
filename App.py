import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

try:
    from xoomar import Xoomar, XoomarRateLimited
    XOOMAR_OK = True
except Exception:
    XOOMAR_OK = False

st.set_page_config(page_title="美股分析工具 Pro", page_icon="📈", layout="wide")

# ============================================================
# 全局 CSS：強制文字顏色，解決過淺問題
# ============================================================
st.markdown("""
<style>
    /* 強制所有文字為深色 */
    html, body, [class*="css"], .stMarkdown, p, span, div {
        color: #1E1E1E !important;
    }
    /* 自訂判讀文字的樣式 */
    .read-text {
        color: #1E1E1E !important;
        font-size: 14px;
        font-weight: 500;
        background-color: #F0F2F6;
        padding: 8px 12px;
        border-radius: 6px;
        margin-top: 5px;
        margin-bottom: 15px;
        border-left: 4px solid #3B82F6;
    }
    /* Metric 標籤顏色 */
    div[data-testid="stMetricLabel"] > div {
        color: #333333 !important;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def get_xoomar():
    return Xoomar() if XOOMAR_OK else None

xoomar_client = get_xoomar()


# ============================================================
# 技術面數據與圖表準備
# ============================================================

@st.cache_data(ttl=3600)
def fetch_technicals(ticker: str):
    tk = yf.Ticker(ticker)
    hist = tk.history(period="1y", interval="1d", auto_adjust=False)
    if hist.empty or "Close" not in hist.columns:
        return {"error": "無法獲取歷史數據，請檢查美股代碼"}
    
    close = hist["Close"].dropna()
    if len(close) < 60:
        return {"error": "歷史數據不足，至少需要約 60 個交易日"}

    # 計算指標
    delta = close.diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    rs = gain / loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    current_rsi = float(rsi.iloc[-1])

    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    macd_line = ema12 - ema26
    signal_line = macd_line.ewm(span=9, adjust=False).mean()
    macd_hist = macd_line - signal_line

    ma20 = close.rolling(20).mean()
    ma50 = close.rolling(50).mean()
    current_price = float(close.iloc[-1])

    std20 = close.rolling(20).std()
    bb_upper = ma20 + 2 * std20
    bb_lower = ma20 - 2 * std20
    bb_width = round(float((bb_upper.iloc[-1] - bb_lower.iloc[-1]) / ma20.iloc[-1] * 100), 2)
    
    if current_price > float(bb_upper.iloc[-1]):
        bb_position, bb_read = "突破上軌", "超買，短線回調風險較高"
    elif current_price < float(bb_lower.iloc[-1]):
        bb_position, bb_read = "跌破下軌", "超賣，短線反彈機會較高"
    elif current_price > float(ma20.iloc[-1]):
        bb_position, bb_read = "中軌以上", "偏強，但未達超買"
    else:
        bb_position, bb_read = "中軌以下", "偏弱，但未達超賣"

    vol_ratio = None
    if "Volume" in hist.columns and len(hist) >= 20:
        v5 = float(hist["Volume"].iloc[-5:].mean())
        v20 = float(hist["Volume"].iloc[-20:].mean())
        vol_ratio = round(v5 / v20, 2) if v20 > 0 else None

    high = hist["High"].dropna()
    low = hist["Low"].dropna()
    tr = pd.concat([high - low, (high - close.shift()).abs(), (low - close.shift()).abs()], axis=1).max(axis=1)
    atr = tr.rolling(14).mean()
    atr_pct = round(float(atr.iloc[-1] / current_price * 100), 2) if current_price > 0 else None

    high20, low20 = float(close.iloc[-20:].max()), float(close.iloc[-20:].min())
    dist_high20, dist_low20 = round((current_price / high20 - 1) * 100, 2), round((current_price / low20 - 1) * 100, 2)
    if dist_high20 > -2: range_read = "接近 20 日高位，強勢，留意獲利回吐"
    elif dist_low20 < 2: range_read = "接近 20 日低位，弱勢，可能反彈"
    else: range_read = "處於 20 日區間中部"

    return {
        "price": round(current_price, 2), "rsi": round(current_rsi, 2),
        "rsi_signal": "overbought" if current_rsi > 70 else "oversold" if current_rsi < 30 else "neutral",
        "rsi_read": "超買（>70），短線回落風險較高" if current_rsi > 70 else "超賣（<30），短線反彈機會較高" if current_rsi < 30 else "中性（30–70），無明顯超買超賣",
        "macd_bullish_cross": bool(macd_hist.iloc[-1] > 0 and macd_hist.iloc[-2] <= 0),
        "macd_bearish_cross": bool(macd_hist.iloc[-1] < 0 and macd_hist.iloc[-2] >= 0),
        "price_above_ma20": current_price > float(ma20.iloc[-1]),
        "ma20_above_ma50": float(ma20.iloc[-1]) > float(ma50.iloc[-1]),
        "bb_position": bb_position, "bb_width": bb_width, "bb_read": bb_read,
        "vol_ratio": vol_ratio, "atr_pct": atr_pct,
        "dist_high20": dist_high20, "dist_low20": dist_low20, "range_read": range_read,
    }

@st.cache_data(ttl=3600)
def fetch_chart_data(ticker: str, timeframe: str):
    tk = yf.Ticker(ticker)
    if timeframe == "日線":
        hist = tk.history(period="1y", interval="1d", auto_adjust=False)
    elif timeframe == "週線":
        hist = tk.history(period="3y", interval="1wk", auto_adjust=False)
    else:
        hist = tk.history(period="5y", interval="1mo", auto_adjust=False)
        
    if hist.empty: return pd.DataFrame()
    
    close = hist["Close"]
    df = pd.DataFrame({
        "Open": hist["Open"], "High": hist["High"], "Low": hist["Low"],
        "Close": close, "Volume": hist["Volume"],
        "MA20": close.rolling(20).mean(), "MA50": close.rolling(50).mean(),
        "BB_Upper": close.rolling(20).mean() + 2 * close.rolling(20).std(),
        "BB_Lower": close.rolling(20).mean() - 2 * close.rolling(20).std()
    }).dropna()
    return df


# ============================================================
# 期權情緒與明細
# ============================================================
def _option_result(put_vol, call_vol, put_oi, call_oi, put_iv=None, call_iv=None):
    vr = put_vol / call_vol if call_vol > 0 else float("inf")
    oir = put_oi / call_oi if call_oi > 0 else float("inf")
    if vr < 0.8 and oir < 0.8: sentiment, sentiment_zh, sentiment_read = "bullish", "強烈看多", "Put 成交量及持倉量均明顯低於 Call"
    elif vr < 1.0 and oir < 1.0: sentiment, sentiment_zh, sentiment_read = "mildly_bullish", "輕微看多", "Put 略低於 Call"
    elif vr > 1.2 and oir > 1.2: sentiment, sentiment_zh, sentiment_read = "bearish", "強烈看空", "Put 成交量及持倉量均明顯高於 Call"
    elif vr > 1.0 and oir > 1.0: sentiment, sentiment_zh, sentiment_read = "mildly_bearish", "輕微看空", "Put 略高於 Call"
    else: sentiment, sentiment_zh, sentiment_read = "neutral", "中性", "成交量與持倉量信號不一致"

    divergence = "成交量與持倉量方向背離" if ((vr < 0.8 and oir > 1.2) or (vr > 1.2 and oir < 0.8)) else None
    iv_skew, iv_read = None, None
    if put_iv is not None and call_iv is not None and call_iv > 0:
        iv_skew = round((put_iv - call_iv) / call_iv * 100, 2)
        if iv_skew > 10: iv_read = "Put IV 明顯高於 Call，市場避險情緒強"
        elif iv_skew < -10: iv_read = "Call IV 明顯高於 Put，市場看多情緒強"
        else: iv_read = "Put 與 Call IV 接近，情緒平衡"

    return {
        "put_volume": int(put_vol), "call_volume": int(call_vol), "put_oi": int(put_oi), "call_oi": int(call_oi),
        "volume_put_call_ratio": round(vr, 2) if np.isfinite(vr) else None,
        "oi_put_call_ratio": round(oir, 2) if np.isfinite(oir) else None,
        "sentiment": sentiment, "sentiment_zh": sentiment_zh, "sentiment_read": sentiment_read,
        "divergence": divergence, "put_iv": round(put_iv, 2) if put_iv is not None else None,
        "call_iv": round(call_iv, 2) if call_iv is not None else None, "iv_skew_pct": iv_skew, "iv_read": iv_read,
    }

@st.cache_data(ttl=300)
def fetch_option_sentiment(ticker: str, max_expiries: int = 4):
    tk = yf.Ticker(ticker)
    try: all_expiries = list(tk.options)
    except Exception as e: return {"error": f"無法讀取期權到期日：{e}"}
    if not all_expiries: return {"error": "此股票無期權數據"}

    expiries = all_expiries[:max_expiries]
    total_put_vol = total_call_vol = total_put_oi = total_call_oi = 0
    put_ivs, call_ivs, expiry_details = [], [], []

    for exp in expiries:
        try:
            chain = tk.option_chain(exp)
            p_vol = pd.to_numeric(chain.puts.get("volume", pd.Series(dtype=float)), errors="coerce").fillna(0).sum()
            c_vol = pd.to_numeric(chain.calls.get("volume", pd.Series(dtype=float)), errors="coerce").fillna(0).sum()
            p_oi = pd.to_numeric(chain.puts.get("openInterest", pd.Series(dtype=float)), errors="coerce").fillna(0).sum()
            c_oi = pd.to_numeric(chain.calls.get("openInterest", pd.Series(dtype=float)), errors="coerce").fillna(0).sum()
            
            total_put_vol += p_vol; total_call_vol += c_vol
            total_put_oi += p_oi; total_call_oi += c_oi
            exp_ratio = round(c_vol / (c_vol + p_vol) * 100, 1) if (c_vol + p_vol) > 0 else 50.0
            
            expiry_details.append({"到期日": exp, "Call 成交": int(c_vol), "Put 成交": int(p_vol), "Call 持倉": int(c_oi), "Put 持倉": int(p_oi), "Call 佔比": exp_ratio})
            
            p_iv = pd.to_numeric(chain.puts.get("impliedVolatility", pd.Series(dtype=float)), errors="coerce").dropna()
            c_iv = pd.to_numeric(chain.calls.get("impliedVolatility", pd.Series(dtype=float)), errors="coerce").dropna()
            if len(p_iv) > 0: put_ivs.append(float(p_iv.mean()) * 100)
            if len(c_iv) > 0: call_ivs.append(float(c_iv.mean()) * 100)
        except Exception: continue

    if total_call_vol == 0 and total_call_oi == 0: return {"error": "期權鏈數據為空"}
    res = _option_result(total_put_vol, total_call_vol, total_put_oi, total_call_oi, np.mean(put_ivs) if put_ivs else None, np.mean(call_ivs) if call_ivs else None)
    res["expiry_details"] = expiry_details
    return res


# ============================================================
# 沽空數據
# ============================================================
def _f(x):
    try: return float(x)
    except: return None

def _build_short_result(spf, d2c, ss, ssp, source):
    rising = ss is not None and ssp is not None and ss > ssp * 1.05
    falling = ss is not None and ssp is not None and ss < ssp * 0.95
    change_pct = round((ss / ssp - 1) * 100, 2) if (ss is not None and ssp is not None and ssp > 0) else None
    change_read = f"較上期上升 {change_pct}%" if rising else f"較上期下降 {abs(change_pct)}%" if falling else "與上期基本持平"

    sig = []
    if rising and (spf is None or spf >= 5): sig.append("short_interest_rising_bearish")
    if rising: sig.append("short_interest_rising")
    if falling and spf is not None and spf >= 10: sig.append("short_covering_bullish")
    if spf is not None and spf >= 15 and falling: sig.append("squeeze_potential")
    if spf is not None and spf < 3 and not rising: sig.append("low_short_pressure")
    if d2c is not None and d2c >= 5 and rising: sig.append("days_to_cover_high_rising")
    if not sig: sig.append("neutral")

    return {
        "source": source, "short_pct_float": round(spf, 2) if spf is not None else None,
        "short_ratio_days_to_cover": round(d2c, 2) if d2c is not None else None,
        "shares_short": int(ss) if ss is not None else None, "shares_short_prior": int(ssp) if ssp is not None else None,
        "short_change_pct": change_pct, "change_read": change_read, "signals": sig,
    }

SIGNAL_ZH = {
    "neutral": "中性，無明顯沽空信號", "short_interest_rising_bearish": "空頭增加且比例高",
    "short_interest_rising": "空頭持倉上升", "short_covering_bullish": "空頭回補",
    "squeeze_potential": "潛在逼空", "low_short_pressure": "沽空壓力有限",
    "days_to_cover_high_rising": "回補天數高且上升", "daily_short_volume_rising": "每日沽空量升",
    "daily_short_volume_falling": "每日沽空量跌",
}

@st.cache_data(ttl=600)
def fetch_short_sentiment(ticker: str):
    if XOOMAR_OK and xoomar_client is not None:
        try:
            si_list = xoomar_client.short_interest(ticker)
            if si_list:
                si, si_prior = si_list[0], si_list[1] if len(si_list) > 1 else {}
                spf = _f(si.get("short_percent_float") or si.get("short_pct_float") or si.get("percent_float"))
                if spf is not None and spf <= 1: spf *= 100
                ss = _f(si.get("shares_short") or si.get("short_interest") or si.get("sharesShort"))
                ssp = _f(si.get("shares_short_prior") or si.get("sharesShortPriorMonth") or si_prior.get("short_interest"))
                d2c = _f(si.get("days_to_cover") or si.get("daysToCover"))
                if any(v is not None for v in [spf, ss, d2c]):
                    res = _build_short_result(spf, d2c, ss, ssp, "xoomar")
                    try:
                        sv = xoomar_client.short_volume(ticker, days=30)
                        if isinstance(sv, list) and len(sv) >= 5:
                            df_sv = pd.DataFrame(sv)
                            sv_col = next((c for c in df_sv.columns if "short" in c.lower() and "volume" in c.lower()), None)
                            if sv_col:
                                vals = pd.to_numeric(df_sv[sv_col], errors="coerce").dropna()
                                if len(vals) >= 5:
                                    recent, earlier = vals.iloc[:5].mean(), vals.iloc[5:10].mean() if len(vals) >= 10 else vals.mean()
                                    change = round((recent / earlier - 1) * 100, 2) if earlier > 0 else None
                                    if change is not None:
                                        if change > 10: trend, read = "rising", f"近5日較前5日上升 {change}%"
                                        elif change < -10: trend, read = "falling", f"近5日較前5日下降 {abs(change)}%"
                                        else: trend, read = "flat", f"基本持平（{change}%）"
                                        res["daily_short_volume_trend"], res["daily_short_volume_read"] = trend, read
                                        if trend == "rising": res["signals"].append("daily_short_volume_rising")
                                        elif trend == "falling": res["signals"].append("daily_short_volume_falling")
                    except Exception: pass
                    return res
        except Exception: pass

    try:
        info = yf.Ticker(ticker).info or {}
        spf = _f(info.get("shortPercentOfFloat"))
        if spf is not None and spf <= 1: spf *= 100
        ss = _f(info.get("sharesShort"))
        ssp = _f(info.get("sharesShortPriorMonth"))
        d2c = _f(info.get("shortRatio"))
        if any(v is not None for v in [spf, ss, d2c]):
            return _build_short_result(spf, d2c, ss, ssp, "yfinance_info")
    except Exception: pass
    return {"error": "無法獲取沽空數據"}


# ============================================================
# 綜合評分
# ============================================================
def score_and_decide(tech, opt, short):
    b = s = sh = 0
    details = []
    if tech["rsi_signal"] == "oversold": b += 2; details.append(("技術面", "RSI 超賣", 2, 0, 0))
    elif tech["rsi_signal"] == "overbought": s += 2; sh += 1; details.append(("技術面", "RSI 超買", 0, 2, 1))
    if tech["macd_bullish_cross"]: b += 2; details.append(("技術面", "MACD 金叉", 2, 0, 0))
    elif tech["macd_bearish_cross"]: s += 2; sh += 1; details.append(("技術面", "MACD 死叉", 0, 2, 1))
    if tech["price_above_ma20"] and tech["ma20_above_ma50"]: b += 2; details.append(("技術面", "多頭排列", 2, 0, 0))
    elif not tech["price_above_ma20"] and not tech["ma20_above_ma50"]: s += 1; sh += 1; details.append(("技術面", "空頭排列", 0, 1, 1))
    if tech["bb_position"] == "跌破下軌": b += 2; details.append(("技術面", "跌破布林下軌", 2, 0, 0))
    elif tech["bb_position"] == "突破上軌": s += 2; sh += 1; details.append(("技術面", "突破布林上軌", 0, 2, 1))

    if isinstance(opt, dict) and "error" not in opt:
        m = {"bullish": (3, 0, 0), "mildly_bullish": (2, 0, 0), "bearish": (0, 0, 3), "mildly_bearish": (0, 1, 2)}
        db, ds, dsh = m.get(opt["sentiment"], (0, 0, 0))
        b += db; s += ds; sh += dsh
        details.append(("期權", opt["sentiment_zh"], db, ds, dsh))
        if opt.get("divergence"): sh += 1; details.append(("期權", "量倉背離", 0, 0, 1))

    if isinstance(short, dict) and "error" not in short:
        sig = short.get("signals", [])
        if "short_interest_rising_bearish" in sig: sh += 3; s += 1; details.append(("沽空", "空頭增加且比例高", 0, 1, 3))
        if "short_interest_rising" in sig: sh += 1; details.append(("沽空", "空頭持倉上升", 0, 0, 1))
        if "short_covering_bullish" in sig: b += 2; details.append(("沽空", "空頭回補", 2, 0, 0))
        if "squeeze_potential" in sig: b += 3; details.append(("沽空", "潛在逼空", 3, 0, 0))
        if "low_short_pressure" in sig: b += 1; details.append(("沽空", "沽空壓力低", 1, 0, 0))
        if "daily_short_volume_rising" in sig: sh += 1; details.append(("沽空", "每日沽空量升", 0, 0, 1))
        if "daily_short_volume_falling" in sig: b += 1; details.append(("沽空", "每日沽空量跌", 1, 0, 0))

    total = b + s + sh
    bp = round(b / total * 100, 1) if total > 0 else 33.3
    sp = round(s / total * 100, 1) if total > 0 else 33.3
    shp = round(sh / total * 100, 1) if total > 0 else 33.3
    probs = {"買入": bp, "賣出": sp, "沽空": shp}
    return {"buy_probability": bp, "sell_probability": sp, "short_probability": shp,
            "top_direction": max(probs, key=probs.get), "raw_scores": {"buy": b, "sell": s, "short": sh}, "details": details}


# ============================================================
# 介面渲染
# ============================================================
st.title("📈 美股分析工具 Pro")
st.caption("技術面 · 期權情緒 · 沽空數據 · 綜合概率")

with st.form("analysis_form"):
    col1, col2 = st.columns([3, 1])
    with col1:
        ticker = st.text_input("美股代碼", value="MU", placeholder="AAPL / TSLA / NVDA / GCTS").upper().strip()
    with col2:
        st.write("期權設定：抓取最近 4 個到期日")
        max_expiries = 4
    submitted = st.form_submit_button("🚀 開始分析", type="primary", use_container_width=True)

if submitted and ticker:
    with st.spinner("分析中…"):
        tech = fetch_technicals(ticker)
    if "error" in tech:
        st.error(f"❌ {tech['error']}")
        st.stop()

    with st.spinner("抓取期權及沽空數據…"):
        opt = fetch_option_sentiment(ticker, max_expiries)
        short = fetch_short_sentiment(ticker)

    result = score_and_decide(tech, opt, short)

    # ============================================================
    # 專業 Plotly 陰陽燭圖表
    # ============================================================
    st.subheader(f"📊 {ticker} 專業走勢圖")
    
    col_chart1, col_chart2 = st.columns([1, 3])
    with col_chart1:
        timeframe = st.radio("時間週期", ["日線", "週線", "月線"], horizontal=True)
    
    df_chart = fetch_chart_data(ticker, timeframe)

    if not df_chart.empty:
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True, 
                            vertical_spacing=0.03, subplot_titles=(f'{ticker} K線', '成交量'), 
                            row_heights=[0.75, 0.25])

        # 1. 陰陽燭 (紅漲綠跌)
        fig.add_trace(go.Candlestick(
            x=df_chart.index, open=df_chart['Open'], high=df_chart['High'],
            low=df_chart['Low'], close=df_chart['Close'], 
            increasing_line_color='red', increasing_fillcolor='red',
            decreasing_line_color='green', decreasing_fillcolor='green',
            name='K線'
        ), row=1, col=1)

        # 2. 均線與布林帶
        fig.add_trace(go.Scatter(x=df_chart.index, y=df_chart['MA20'], line=dict(color='orange', width=1.5), name='MA20'), row=1, col=1)
        fig.add_trace(go.Scatter(x=df_chart.index, y=df_chart['MA50'], line=dict(color='blue', width=1.5), name='MA50'), row=1, col=1)
        fig.add_trace(go.Scatter(x=df_chart.index, y=df_chart['BB_Upper'], line=dict(color='rgba(150,150,150,0.6)', width=1), name='布林上軌'), row=1, col=1)
        fig.add_trace(go.Scatter(x=df_chart.index, y=df_chart['BB_Lower'], line=dict(color='rgba(150,150,150,0.6)', width=1), fill='tonexty', fillcolor='rgba(150,150,150,0.1)', name='布林下軌'), row=1, col=1)

        # 3. 成交量
        colors = ['red' if row['Close'] >= row['Open'] else 'green' for _, row in df_chart.iterrows()]
        fig.add_trace(go.Bar(x=df_chart.index, y=df_chart['Volume'], marker_color=colors, name='成交量'), row=2, col=1)

        # 美化圖表
        fig.update_layout(
            xaxis_rangeslider_visible=False, height=550,
            margin=dict(l=10, r=10, t=30, b=10),
            paper_bgcolor='white', plot_bgcolor='white',
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            hovermode="x unified"
        )
        fig.update_xaxes(showgrid=True, gridcolor='#E5E5E5')
        fig.update_yaxes(showgrid=True, gridcolor='#E5E5E5')
        
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("無法繪製圖表，數據為空。")

    # ============================================================
    # 三大維度數據 (使用 HTML 強制深色文字)
    # ============================================================
    c1, c2, c3 = st.columns(3)

    with c1:
        st.markdown("### 📊 技術面")
        st.metric("現價", tech["price"])
        st.metric("RSI(14)", tech["rsi"], help="14 日相對強弱指數：>70 超買，<30 超賣")
        st.markdown(f'<div class="read-text">RSI 判讀：{tech["rsi_read"]}</div>', unsafe_allow_html=True)
        st.write(f"**MACD 金叉**：{'✅ 是' if tech['macd_bullish_cross'] else '❌ 否'}")
        st.write(f"**MACD 死叉**：{'✅ 是' if tech['macd_bearish_cross'] else '❌ 否'}")
        st.write(f"**價格 > MA20**：{'✅ 是' if tech['price_above_ma20'] else '❌ 否'}")
        st.write(f"**MA20 > MA50**：{'✅ 是' if tech['ma20_above_ma50'] else '❌ 否'}")
        st.write(f"**布林帶位置**：{tech['bb_position']}（寬度 {tech['bb_width']}%）")
        st.markdown(f'<div class="read-text">布林帶判讀：{tech["bb_read"]}</div>', unsafe_allow_html=True)
        if tech["vol_ratio"] is not None: st.write(f"**成交量比（5日/20日）**：{tech['vol_ratio']}")
        if tech["atr_pct"] is not None: st.write(f"**ATR 波動率**：{tech['atr_pct']}%")
        st.write(f"**距 20 日高 / 低**：{tech['dist_high20']}% / {tech['dist_low20']}%")
        st.markdown(f'<div class="read-text">區間判讀：{tech["range_read"]}</div>', unsafe_allow_html=True)

    with c2:
        st.markdown("### 🎯 期權情緒")
        if isinstance(opt, dict) and "error" not in opt:
            st.metric("Put/Call 成交量比率", opt["volume_put_call_ratio"], help="Put 成交量 ÷ Call 成交量。基數為 Call；<1 偏多，>1 偏空")
            st.metric("Put/Call 持倉量比率", opt["oi_put_call_ratio"], help="Put 未平倉 ÷ Call 未平倉。基數為 Call；<1 偏多，>1 偏空")
            st.write(f"**情緒判斷**：{opt['sentiment_zh']}")
            st.markdown(f'<div class="read-text">判讀：{opt["sentiment_read"]}</div>', unsafe_allow_html=True)
            if opt.get("divergence"):
                st.warning(f"⚠️ {opt['divergence']}")
            if opt.get("put_iv") is not None and opt.get("call_iv") is not None:
                st.write(f"**Put IV / Call IV**：{opt['put_iv']}% / {opt['call_iv']}%")
                if opt.get("iv_skew_pct") is not None:
                    st.write(f"**IV 偏斜**：{opt['iv_skew_pct']}%")
                    st.markdown(f'<div class="read-text">判讀：{opt["iv_read"]}</div>', unsafe_allow_html=True)
                iv_df = pd.DataFrame({"IV": [opt["put_iv"], opt["call_iv"]]}, index=["Put IV", "Call IV"])
                st.bar_chart(iv_df, height=150)
        else:
            st.error(opt.get("error", "期權數據不可用"))

    with c3:
        st.markdown("### 📉 沽空數據")
        if isinstance(short, dict) and "error" not in short:
            spf = short["short_pct_float"]
            st.metric("Short % of Float", f"{spf}%" if spf is not None else "N/A", help="空頭股數佔流通股比例")
            d2c = short["short_ratio_days_to_cover"]
            st.metric("Days to Cover", d2c if d2c is not None else "N/A", help="空頭回補所需日數")
            st.write(f"**空頭持倉**：{short['shares_short']}")
            st.write(f"**上期持倉**：{short['shares_short_prior']}")
            st.markdown(f'<div class="read-text">持倉變化：{short["change_read"]}</div>', unsafe_allow_html=True)
            if short.get("daily_short_volume_read"):
                st.write("**每日沽空量趨勢**")
                st.markdown(f'<div class="read-text">{short["daily_short_volume_read"]}</div>', unsafe_allow_html=True)
            st.write("**信號**")
            for sig in short["signals"]:
                st.write(f"- {SIGNAL_ZH.get(sig, sig)}")
        else:
            st.error(short.get("error", "沽空數據不可用"))

    # ============================================================
    # 期權到期日明細
    # ============================================================
    if isinstance(opt, dict) and "error" not in opt and opt.get("expiry_details"):
        st.divider()
        st.markdown("### 📅 期權到期日明細（Call vs Put）")
        exp_df = pd.DataFrame(opt["expiry_details"])
        
        for idx, row in exp_df.iterrows():
            call_pct = int(row['Call 佔比'])
            put_pct = 100 - call_pct
            st.write(f"**{row['到期日']}**  ──  Call {call_pct}% : {put_pct}% Put")
            st.progress(row['Call 佔比'] / 100)
            
        with st.expander("查看詳細張數數據表"):
            st.dataframe(exp_df.set_index("到期日"), use_container_width=True)

    st.divider()
    st.markdown("## 📈 綜合分析結果")

    r1, r2, r3 = st.columns(3)
    with r1:
        st.metric("🟢 買入概率", f"{result['buy_probability']}%")
        st.progress(result["buy_probability"] / 100)
    with r2:
        st.metric("🟡 賣出概率", f"{result['sell_probability']}%")
        st.progress(result["sell_probability"] / 100)
    with r3:
        st.metric("🔴 沽空概率", f"{result['short_probability']}%")
        st.progress(result["short_probability"] / 100)

    st.success(f"➡️ 概率最高方向：**{result['top_direction']}**")
    st.caption(f"原始分數：{result['raw_scores']}")

    if result.get("details"):
        st.markdown("### 🔍 信號明細")
        det_df = pd.DataFrame(
            result["details"],
            columns=["維度", "信號", "買入加分", "賣出加分", "沽空加分"],
        )
        st.dataframe(det_df, use_container_width=True, hide_index=True)

    with st.expander("📋 查看完整數據（JSON）"):
        st.json({
            "ticker": ticker,
            "technicals": tech,
            "options": opt,
            "short": short,
            "result": result,
        })

st.divider()
st.caption("⚠️ 本工具僅供參考，不構成任何投資建議。期權及沽空數據可能延遲或不完整，實際交易前請自行核實。")
