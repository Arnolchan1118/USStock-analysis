import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np

# ---------- 可選依賴，缺失時降級 ----------
try:
    from openbb import obb
    obb.user.preferences.output_type = "dataframe"
    OPENBB_OK = True
except Exception:
    OPENBB_OK = False

try:
    from xoomar import Xoomar, XoomarRateLimited
    XOOMAR_OK = True
except Exception:
    XOOMAR_OK = False

st.set_page_config(
    page_title="美股分析工具",
    page_icon="📈",
    layout="wide",
)

@st.cache_resource
def get_xoomar():
    return Xoomar() if XOOMAR_OK else None

xoomar_client = get_xoomar()


# ============================================================
# 技術面
# ============================================================

@st.cache_data(ttl=3600)
def fetch_technicals(ticker: str):
    tk = yf.Ticker(ticker)
    hist = tk.history(period="6mo", auto_adjust=False)
    if hist.empty or "Close" not in hist.columns:
        return {"error": "無法獲取歷史數據，請檢查美股代碼"}
    close = hist["Close"].dropna()
    if len(close) < 60:
        return {"error": "歷史數據不足，至少需要約 60 個交易日"}

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

    chart_df = pd.DataFrame({
        "Close": close,
        "MA20": ma20,
        "MA50": ma50,
    }).dropna()

    return {
        "price": round(current_price, 2),
        "rsi": round(current_rsi, 2),
        "rsi_signal": (
            "overbought" if current_rsi > 70
            else "oversold" if current_rsi < 30
            else "neutral"
        ),
        "macd_bullish_cross": bool(macd_hist.iloc[-1] > 0 and macd_hist.iloc[-2] <= 0),
        "macd_bearish_cross": bool(macd_hist.iloc[-1] < 0 and macd_hist.iloc[-2] >= 0),
        "price_above_ma20": current_price > float(ma20.iloc[-1]),
        "ma20_above_ma50": float(ma20.iloc[-1]) > float(ma50.iloc[-1]),
        "chart_df": chart_df,
    }


# ============================================================
# 期權情緒
# ============================================================

def _option_result(put_vol, call_vol, put_oi, call_oi):
    vr = put_vol / call_vol if call_vol > 0 else float("inf")
    oir = put_oi / call_oi if call_oi > 0 else float("inf")
    if vr < 0.8 and oir < 0.8:
        sentiment = "bullish"
    elif vr < 1.0 and oir < 1.0:
        sentiment = "mildly_bullish"
    elif vr > 1.2 and oir > 1.2:
        sentiment = "bearish"
    elif vr > 1.0 and oir > 1.0:
        sentiment = "mildly_bearish"
    else:
        sentiment = "neutral"
    return {
        "put_volume": int(put_vol), "call_volume": int(call_vol),
        "put_oi": int(put_oi), "call_oi": int(call_oi),
        "volume_put_call_ratio": round(vr, 2) if np.isfinite(vr) else "N/A",
        "oi_put_call_ratio": round(oir, 2) if np.isfinite(oir) else "N/A",
        "sentiment": sentiment,
    }


@st.cache_data(ttl=300)
def fetch_option_sentiment(ticker: str, max_expiries: int = 6):
    if not OPENBB_OK:
        return {"error": "OpenBB 未安裝，跳過期權維度"}
    try:
        chain_df = obb.derivatives.options.chains(
            symbol=ticker, provider="cboe"
        ).to_df()
    except Exception as e:
        return {"error": f"期權抓取失敗：{e}"}

    if chain_df is None or chain_df.empty:
        return {"error": "期權鏈為空"}

    cm = {c.lower(): c for c in chain_df.columns}
    tcol = cm.get("option_type") or cm.get("type")
    vcol = cm.get("volume")
    ocol = cm.get("open_interest") or cm.get("openinterest")
    ecol = cm.get("expiration")
    if not all([tcol, vcol, ocol]):
        return {"error": "期權鏈欄位缺失"}

    df = chain_df.copy()
    df[vcol] = pd.to_numeric(df[vcol], errors="coerce").fillna(0)
    df[ocol] = pd.to_numeric(df[ocol], errors="coerce").fillna(0)

    used = []
    if ecol and max_expiries:
        exps = sorted(df[ecol].dropna().astype(str).unique())[:max_expiries]
        df = df[df[ecol].astype(str).isin(exps)]
        used = exps

    puts = df[df[tcol].astype(str).str.lower().str.startswith("put")]
    calls = df[df[tcol].astype(str).str.lower().str.startswith("call")]
    res = _option_result(
        puts[vcol].sum(), calls[vcol].sum(),
        puts[ocol].sum(), calls[ocol].sum(),
    )
    res["used_expiries"] = used
    return res


# ============================================================
# 沽空數據
# ============================================================

def _f(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def _build_short_result(spf, d2c, ss, ssp):
    rising = ss is not None and ssp is not None and ss > ssp * 1.05
    falling = ss is not None and ssp is not None and ss < ssp * 0.95
    sig = []
    if rising and (spf is None or spf >= 5):
        sig.append("short_interest_rising_bearish")
    if rising:
        sig.append("short_interest_rising")
    if falling and spf is not None and spf >= 10:
        sig.append("short_covering_bullish")
    if spf is not None and spf >= 15 and falling:
        sig.append("squeeze_potential")
    if spf is not None and spf < 3 and not rising:
        sig.append("low_short_pressure")
    if d2c is not None and d2c >= 5 and rising:
        sig.append("days_to_cover_high_rising")
    if not sig:
        sig.append("neutral")
    return {
        "short_pct_float": round(spf, 2) if spf is not None else None,
        "short_ratio_days_to_cover": round(d2c, 2) if d2c is not None else None,
        "shares_short": int(ss) if ss is not None else None,
        "shares_short_prior": int(ssp) if ssp is not None else None,
        "short_interest_rising": rising,
        "short_interest_falling": falling,
        "signals": sig,
    }


@st.cache_data(ttl=600)
def fetch_short_sentiment(ticker: str):
    if not XOOMAR_OK or xoomar_client is None:
        return {"error": "xoomar 未安裝，跳過沽空維度"}
    try:
        si_list = xoomar_client.short_interest(ticker)
        si = si_list[0] if si_list else {}
        sv = xoomar_client.short_volume(ticker, days=30)
    except XoomarRateLimited as e:
        return {"error": f"XOOMAR 限流，{e.retry_after}s 後再試"}
    except Exception as e:
        return {"error": f"沽空抓取失敗：{e}"}

    spf = _f(si.get("short_percent_float") or si.get("shortPercentFloat") or si.get("short_pct_float"))
    if spf is not None and spf <= 1:
        spf *= 100
    ss = _f(si.get("shares_short") or si.get("sharesShort") or si.get("short_shares"))
    ssp = _f(si.get("shares_short_prior") or si.get("sharesShortPriorMonth") or si.get("previous_shares_short"))
    d2c = _f(si.get("days_to_cover") or si.get("daysToCover"))

    res = _build_short_result(spf, d2c, ss, ssp)

    if isinstance(sv, list) and len(sv) >= 5:
        df = pd.DataFrame(sv)
        svc = next(
            (c for c in df.columns if "short" in c.lower() and "volume" in c.lower()),
            None,
        )
        if svc:
            vals = pd.to_numeric(df[svc], errors="coerce").dropna()
            if len(vals) >= 5:
                recent = vals.iloc[:5].mean()
                earlier = vals.iloc[5:10].mean() if len(vals) >= 10 else vals.mean()
                trend = (
                    "rising" if recent > earlier * 1.1
                    else "falling" if recent < earlier * 0.9
                    else "flat"
                )
                res["daily_short_volume_trend"] = trend
                if trend == "rising":
                    res["signals"].append("daily_short_volume_rising")
                elif trend == "falling":
                    res["signals"].append("daily_short_volume_falling")
    return res


# ============================================================
# 綜合評分
# ============================================================

def score_and_decide(tech, opt, short):
    b = s = sh = 0

    if tech["rsi_signal"] == "oversold":
        b += 2
    elif tech["rsi_signal"] == "overbought":
        s += 2
        sh += 1
    if tech["macd_bullish_cross"]:
        b += 2
    elif tech["macd_bearish_cross"]:
        s += 2
        sh += 1
    if tech["price_above_ma20"] and tech["ma20_above_ma50"]:
        b += 2
    elif not tech["price_above_ma20"] and not tech["ma20_above_ma50"]:
        s += 1
        sh += 1

    if isinstance(opt, dict) and "error" not in opt:
        m = {
            "bullish": (3, 0, 0),
            "mildly_bullish": (2, 0, 0),
            "bearish": (0, 0, 3),
            "mildly_bearish": (0, 1, 2),
        }
        db, ds, dsh = m.get(opt["sentiment"], (0, 0, 0))
        b += db; s += ds; sh += dsh

    if isinstance(short, dict) and "error" not in short:
        sig = short.get("signals", [])
        if "short_interest_rising_bearish" in sig:
            sh += 3; s += 1
        if "short_interest_rising" in sig:
            sh += 1
        if "short_covering_bullish" in sig:
            b += 2
        if "squeeze_potential" in sig:
            b += 3
        if "low_short_pressure" in sig:
            b += 1
        if "days_to_cover_high_rising" in sig:
            sh += 1
        if "daily_short_volume_rising" in sig:
            sh += 1
        if "daily_short_volume_falling" in sig:
            b += 1

    total = b + s + sh
    if total == 0:
        bp = sp = shp = 33.3
    else:
        bp = round(b / total * 100, 1)
        sp = round(s / total * 100, 1)
        shp = round(sh / total * 100, 1)

    probs = {"買入": bp, "賣出": sp, "沽空": shp}
    return {
        "buy_probability": bp,
        "sell_probability": sp,
        "short_probability": shp,
        "top_direction": max(probs, key=probs.get),
        "raw_scores": {"buy": b, "sell": s, "short": sh},
    }


# ============================================================
# 介面
# ============================================================

st.title("📈 美股分析工具")
st.caption("技術面 · 期權情緒 · 沽空數據 · 綜合概率")

with st.form("analysis_form"):
    col1, col2 = st.columns([2, 1])
    with col1:
        ticker = st.text_input(
            "美股代碼", value="AAPL",
            placeholder="AAPL / TSLA / NVDA / MSFT",
        ).upper().strip()
    with col2:
        max_expiries = st.slider("期權到期日數量", 1, 12, 6)
    submitted = st.form_submit_button("🚀 開始分析", type="primary", use_container_width=True)

if submitted and ticker:
    with st.spinner("分析中，請稍候…"):
        tech = fetch_technicals(ticker)

    if "error" in tech:
        st.error(f"❌ {tech['error']}")
        st.stop()

    with st.spinner("抓取期權及沽空數據…"):
        opt = fetch_option_sentiment(ticker, max_expiries)
        short = fetch_short_sentiment(ticker)

    result = score_and_decide(tech, opt, short)

    # ---------- 股價圖 ----------
    st.subheader(f"📊 {ticker} 股價走勢（6 個月）")
    st.line_chart(tech["chart_df"], height=300)

    # ---------- 三大維度 ----------
    c1, c2, c3 = st.columns(3)

    with c1:
        st.markdown("### 📊 技術面")
        st.metric("現價", tech["price"])
        st.metric("RSI(14)", tech["rsi"], help=tech["rsi_signal"])
        st.write(f"**RSI 狀態**：{tech['rsi_signal']}")
        st.write(f"**MACD 金叉**：{tech['macd_bullish_cross']}")
        st.write(f"**MACD 死叉**：{tech['macd_bearish_cross']}")
        st.write(f"**價格 > MA20**：{tech['price_above_ma20']}")
        st.write(f"**MA20 > MA50**：{tech['ma20_above_ma50']}")

    with c2:
        st.markdown("### 🎯 期權情緒")
        if isinstance(opt, dict) and "error" not in opt:
            st.metric("Put/Call 成交量比率", opt["volume_put_call_ratio"])
            st.metric("Put/Call 持倉量比率", opt["oi_put_call_ratio"])
            st.write(f"**Put / Call 成交量**：{opt['put_volume']} / {opt['call_volume']}")
            st.write(f"**Put / Call 未平倉**：{opt['put_oi']} / {opt['call_oi']}")
            st.write(f"**情緒判斷**：`{opt['sentiment']}`")
            if opt.get("used_expiries"):
                st.caption(f"已汇总到期日：{', '.join(opt['used_expiries'])}")
        else:
            st.warning(opt.get("error", "期權數據不可用"))

    with c3:
        st.markdown("### 📉 沽空數據")
        if isinstance(short, dict) and "error" not in short:
            spf = short["short_pct_float"]
            st.metric("Short % of Float", f"{spf}%" if spf is not None else "N/A")
            d2c = short["short_ratio_days_to_cover"]
            st.metric("Days to Cover", d2c if d2c is not None else "N/A")
            st.write(f"**空頭持倉**：{short['shares_short']}")
            st.write(f"**上期持倉**：{short['shares_short_prior']}")
            if short.get("daily_short_volume_trend"):
                st.write(f"**每日沽空量趨勢**：{short['daily_short_volume_trend']}")
            st.write(f"**信號**：`{', '.join(short['signals'])}`")
        else:
            st.warning(short.get("error", "沽空數據不可用"))

    # ---------- 綜合結果 ----------
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

    with st.expander("📋 查看完整數據（JSON）"):
        st.json({
            "ticker": ticker,
            "technicals": {k: v for k, v in tech.items() if k != "chart_df"},
            "options": opt,
            "short": short,
            "result": result,
        })

st.divider()
st.caption("⚠️ 本工具僅供參考，不構成任何投資建議。期權及沽空數據可能延遲或不完整，實際交易前請自行核實。")
