import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np

try:
    from xoomar import Xoomar, XoomarRateLimited
    XOOMAR_OK = True
except Exception:
    XOOMAR_OK = False

st.set_page_config(page_title="美股分析工具 Pro", page_icon="📈", layout="wide")

@st.cache_resource
def get_xoomar():
    return Xoomar() if XOOMAR_OK else None

xoomar_client = get_xoomar()


# ============================================================
# 技术面（增强）
# ============================================================

@st.cache_data(ttl=3600)
def fetch_technicals(ticker: str):
    tk = yf.Ticker(ticker)
    hist = tk.history(period="6mo", auto_adjust=False)
    if hist.empty or "Close" not in hist.columns:
        return {"error": "无法获取历史数据，请检查美股代码"}
    close = hist["Close"].dropna()
    volume = hist["Volume"].dropna() if "Volume" in hist.columns else None
    if len(close) < 60:
        return {"error": "历史数据不足"}

    # RSI
    delta = close.diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    rs = gain / loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    current_rsi = float(rsi.iloc[-1])

    # MACD
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    macd_line = ema12 - ema26
    signal_line = macd_line.ewm(span=9, adjust=False).mean()
    macd_hist = macd_line - signal_line

    # MA
    ma20 = close.rolling(20).mean()
    ma50 = close.rolling(50).mean()
    current_price = float(close.iloc[-1])

    # 布林带（20日，2倍标准差）
    std20 = close.rolling(20).std()
    bb_upper = ma20 + 2 * std20
    bb_lower = ma20 - 2 * std20
    bb_width = float((bb_upper.iloc[-1] - bb_lower.iloc[-1]) / ma20.iloc[-1] * 100)
    if current_price > float(bb_upper.iloc[-1]):
        bb_position = "above_upper"      # 突破上轨
    elif current_price < float(bb_lower.iloc[-1]):
        bb_position = "below_lower"      # 跌破下轨
    elif current_price > float(ma20.iloc[-1]):
        bb_position = "upper_half"
    else:
        bb_position = "lower_half"

    # 成交量比（5日/20日）
    vol_ratio = None
    if volume is not None and len(volume) >= 20:
        v5 = float(volume.iloc[-5:].mean())
        v20 = float(volume.iloc[-20:].mean())
        vol_ratio = round(v5 / v20, 2) if v20 > 0 else None

    # ATR%（14日）
    high = hist["High"].dropna()
    low = hist["Low"].dropna()
    tr = pd.concat([
        high - low,
        (high - close.shift()).abs(),
        (low - close.shift()).abs(),
    ], axis=1).max(axis=1)
    atr = tr.rolling(14).mean()
    atr_pct = round(float(atr.iloc[-1] / current_price * 100), 2) if current_price > 0 else None

    # 距20日高低点
    high20 = float(close.iloc[-20:].max())
    low20 = float(close.iloc[-20:].min())
    dist_high20 = round((current_price / high20 - 1) * 100, 2)
    dist_low20 = round((current_price / low20 - 1) * 100, 2)

    chart_df = pd.DataFrame({
        "Close": close, "MA20": ma20, "MA50": ma50,
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
        "bb_position": bb_position,
        "bb_width": bb_width,
        "vol_ratio": vol_ratio,
        "atr_pct": atr_pct,
        "dist_high20": dist_high20,
        "dist_low20": dist_low20,
        "chart_df": chart_df,
    }


# ============================================================
# 期权情绪（yfinance + IV）
# ============================================================

def _option_result(put_vol, call_vol, put_oi, call_oi, put_iv=None, call_iv=None):
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

    # 量仓背离检测
    divergence = None
    if (vr < 0.8 and oir > 1.2) or (vr > 1.2 and oir < 0.8):
        divergence = "volume_oi_divergence"

    iv_skew = None
    if put_iv is not None and call_iv is not None and call_iv > 0:
        iv_skew = round((put_iv - call_iv) / call_iv * 100, 2)

    return {
        "put_volume": int(put_vol), "call_volume": int(call_vol),
        "put_oi": int(put_oi), "call_oi": int(call_oi),
        "volume_put_call_ratio": round(vr, 2) if np.isfinite(vr) else None,
        "oi_put_call_ratio": round(oir, 2) if np.isfinite(oir) else None,
        "sentiment": sentiment,
        "divergence": divergence,
        "put_iv": round(put_iv, 2) if put_iv is not None else None,
        "call_iv": round(call_iv, 2) if call_iv is not None else None,
        "iv_skew_pct": iv_skew,
    }


@st.cache_data(ttl=300)
def fetch_option_sentiment(ticker: str, max_expiries: int = 6):
    tk = yf.Ticker(ticker)
    try:
        all_expiries = list(tk.options)
    except Exception as e:
        return {"error": f"无法读取期权到期日：{e}"}

    if not all_expiries:
        return {"error": "此股票无期权数据（可能未上市期权或代码错误）"}

    expiries = all_expiries[:max_expiries]
    total_put_vol = total_call_vol = 0
    total_put_oi = total_call_oi = 0
    put_ivs, call_ivs = [], []
    used = []
    errors = []

    for exp in expiries:
        try:
            chain = tk.option_chain(exp)
            puts = chain.puts
            calls = chain.calls
            total_put_vol += pd.to_numeric(puts.get("volume", pd.Series(dtype=float)), errors="coerce").fillna(0).sum()
            total_call_vol += pd.to_numeric(calls.get("volume", pd.Series(dtype=float)), errors="coerce").fillna(0).sum()
            total_put_oi += pd.to_numeric(puts.get("openInterest", pd.Series(dtype=float)), errors="coerce").fillna(0).sum()
            total_call_oi += pd.to_numeric(calls.get("openInterest", pd.Series(dtype=float)), errors="coerce").fillna(0).sum()

            p_iv = pd.to_numeric(puts.get("impliedVolatility", pd.Series(dtype=float)), errors="coerce").dropna()
            c_iv = pd.to_numeric(calls.get("impliedVolatility", pd.Series(dtype=float)), errors="coerce").dropna()
            if len(p_iv) > 0:
                put_ivs.append(float(p_iv.mean()) * 100)
            if len(c_iv) > 0:
                call_ivs.append(float(c_iv.mean()) * 100)
            used.append(exp)
        except Exception as e:
            errors.append(f"{exp}: {str(e)[:60]}")
            continue

    if total_call_vol == 0 and total_call_oi == 0:
        return {"error": f"期权链数据为空。到期日：{expiries}。可能原因：期权流动性低或数据延迟。", "raw_errors": errors}

    put_iv = np.mean(put_ivs) if put_ivs else None
    call_iv = np.mean(call_ivs) if call_ivs else None
    res = _option_result(total_put_vol, total_call_vol, total_put_oi, total_call_oi, put_iv, call_iv)
    res["used_expiries"] = used
    res["source"] = "yfinance"
    if errors:
        res["warnings"] = errors
    return res


# ============================================================
# 沽空数据（xoomar + yfinance info 双源）
# ============================================================

def _f(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def _build_short_result(spf, d2c, ss, ssp, source):
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
        "source": source,
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
    # 第一优先：xoomar
    if XOOMAR_OK and xoomar_client is not None:
        try:
            si_list = xoomar_client.short_interest(ticker)
            if si_list:
                si = si_list[0]
                si_prior = si_list[1] if len(si_list) > 1 else {}

                spf = _f(si.get("short_percent_float") or si.get("shortPercentFloat") or si.get("short_pct_float") or si.get("percent_float"))
                if spf is not None and spf <= 1:
                    spf *= 100

                ss = _f(si.get("shares_short") or si.get("short_interest") or si.get("shortInterest") or si.get("sharesShort"))
                ssp = _f(si.get("shares_short_prior") or si.get("sharesShortPriorMonth") or si.get("previous_shares_short")
                         or si_prior.get("short_interest") or si_prior.get("shortInterest") or si_prior.get("shares_short"))
                d2c = _f(si.get("days_to_cover") or si.get("daysToCover"))

                if any(v is not None for v in [spf, ss, d2c]):
                    res = _build_short_result(spf, d2c, ss, ssp, "xoomar")

                    # 每日沽空量趋势
                    try:
                        sv = xoomar_client.short_volume(ticker, days=30)
                        if isinstance(sv, list) and len(sv) >= 5:
                            df_sv = pd.DataFrame(sv)
                            sv_col = next((c for c in df_sv.columns if "short" in c.lower() and "volume" in c.lower()), None)
                            if sv_col:
                                vals = pd.to_numeric(df_sv[sv_col], errors="coerce").dropna()
                                if len(vals) >= 5:
                                    recent = vals.iloc[:5].mean()
                                    earlier = vals.iloc[5:10].mean() if len(vals) >= 10 else vals.mean()
                                    trend = "rising" if recent > earlier * 1.1 else "falling" if recent < earlier * 0.9 else "flat"
                                    res["daily_short_volume_trend"] = trend
                                    if trend == "rising":
                                        res["signals"].append("daily_short_volume_rising")
                                    elif trend == "falling":
                                        res["signals"].append("daily_short_volume_falling")
                    except Exception:
                        pass
                    return res
        except XoomarRateLimited:
            pass
        except Exception:
            pass

    # 第二优先：yfinance info
    try:
        info = yf.Ticker(ticker).info or {}
        spf = _f(info.get("shortPercentOfFloat"))
        if spf is not None and spf <= 1:
            spf *= 100
        ss = _f(info.get("sharesShort"))
        ssp = _f(info.get("sharesShortPriorMonth"))
        d2c = _f(info.get("shortRatio"))
        if any(v is not None for v in [spf, ss, d2c]):
            res = _build_short_result(spf, d2c, ss, ssp, "yfinance_info")
            res["note"] = "数据来自 yfinance info（双周更新，可能滞后）"
            return res
    except Exception as e:
        pass

    return {"error": "xoomar 与 yfinance 均无法获取沽空数据", "source": None}


# ============================================================
# 综合评分（含明细）
# ============================================================

def score_and_decide(tech, opt, short):
    b = s = sh = 0
    details = []  # (维度, 信号, 买入, 卖出, 沽空)

    # 技术面
    if tech["rsi_signal"] == "oversold":
        b += 2; details.append(("技术面", "RSI 超卖", 2, 0, 0))
    elif tech["rsi_signal"] == "overbought":
        s += 2; sh += 1; details.append(("技术面", "RSI 超买", 0, 2, 1))

    if tech["macd_bullish_cross"]:
        b += 2; details.append(("技术面", "MACD 金叉", 2, 0, 0))
    elif tech["macd_bearish_cross"]:
        s += 2; sh += 1; details.append(("技术面", "MACD 死叉", 0, 2, 1))

    if tech["price_above_ma20"] and tech["ma20_above_ma50"]:
        b += 2; details.append(("技术面", "价格 > MA20 > MA50", 2, 0, 0))
    elif not tech["price_above_ma20"] and not tech["ma20_above_ma50"]:
        s += 1; sh += 1; details.append(("技术面", "价格 < MA20 < MA50", 0, 1, 1))

    if tech["bb_position"] == "below_lower":
        b += 2; details.append(("技术面", "跌破布林下轨（超卖）", 2, 0, 0))
    elif tech["bb_position"] == "above_upper":
        s += 2; sh += 1; details.append(("技术面", "突破布林上轨（超买）", 0, 2, 1))

    if tech["vol_ratio"] is not None:
        if tech["vol_ratio"] > 1.5 and tech["rsi_signal"] != "neutral":
            sh += 1; details.append(("技术面", "成交量放大 + 方向明确", 0, 0, 1))

    # 期权
    if isinstance(opt, dict) and "error" not in opt:
        m = {"bullish": (3, 0, 0), "mildly_bullish": (2, 0, 0), "bearish": (0, 0, 3), "mildly_bearish": (0, 1, 2)}
        db, ds, dsh = m.get(opt["sentiment"], (0, 0, 0))
        b += db; s += ds; sh += dsh
        details.append(("期权", f"情绪：{opt['sentiment']}", db, ds, dsh))

        if opt.get("divergence"):
            sh += 1; details.append(("期权", "成交量与持仓量背离", 0, 0, 1))

        if opt.get("iv_skew_pct") is not None:
            if opt["iv_skew_pct"] > 10:
                sh += 1; details.append(("期权", f"Put IV 偏斜 +{opt['iv_skew_pct']}%", 0, 0, 1))
            elif opt["iv_skew_pct"] < -10:
                b += 1; details.append(("期权", f"Call IV 偏斜 {opt['iv_skew_pct']}%", 1, 0, 0))

    # 沽空
    if isinstance(short, dict) and "error" not in short:
        sig = short.get("signals", [])
        if "short_interest_rising_bearish" in sig:
            sh += 3; s += 1; details.append(("沽空", "空头持仓上升 + 比例偏高", 0, 1, 3))
        if "short_interest_rising" in sig:
            sh += 1; details.append(("沽空", "空头持仓上升", 0, 0, 1))
        if "short_covering_bullish" in sig:
            b += 2; details.append(("沽空", "空头回补", 2, 0, 0))
        if "squeeze_potential" in sig:
            b += 3; details.append(("沽空", "潜在逼空", 3, 0, 0))
        if "low_short_pressure" in sig:
            b += 1; details.append(("沽空", "沽空压力低", 1, 0, 0))
        if "days_to_cover_high_rising" in sig:
            sh += 1; details.append(("沽空", "Days to Cover 高 + 上升", 0, 0, 1))
        if "daily_short_volume_rising" in sig:
            sh += 1; details.append(("沽空", "每日沽空量上升", 0, 0, 1))
        if "daily_short_volume_falling" in sig:
            b += 1; details.append(("沽空", "每日沽空量下降", 1, 0, 0))

    total = b + s + sh
    if total == 0:
        bp = sp = shp = 33.3
    else:
        bp = round(b / total * 100, 1)
        sp = round(s / total * 100, 1)
        shp = round(sh / total * 100, 1)

    probs = {"买入": bp, "卖出": sp, "沽空": shp}
    return {
        "buy_probability": bp,
        "sell_probability": sp,
        "short_probability": shp,
        "top_direction": max(probs, key=probs.get),
        "raw_scores": {"buy": b, "sell": s, "short": sh},
        "details": details,
    }


# ============================================================
# 界面
# ============================================================

st.title("📈 美股分析工具 Pro")
st.caption("技术面 · 期权情绪 · 沽空数据 · 综合概率 · 信号明细")

with st.form("analysis_form"):
    col1, col2 = st.columns([2, 1])
    with col1:
        ticker = st.text_input("美股代码", value="MU", placeholder="AAPL / TSLA / NVDA / MSFT").upper().strip()
    with col2:
        max_expiries = st.slider("期权到期日数量", 1, 12, 6, help="汇总最近 N 个到期日的 Put/Call 数据")
    submitted = st.form_submit_button("🚀 开始分析", type="primary", use_container_width=True)

if submitted and ticker:
    with st.spinner("分析中…"):
        tech = fetch_technicals(ticker)
    if "error" in tech:
        st.error(f"❌ {tech['error']}")
        st.stop()

    with st.spinner("抓取期权及沽空数据…"):
        opt = fetch_option_sentiment(ticker, max_expiries)
        short = fetch_short_sentiment(ticker)

    result = score_and_decide(tech, opt, short)

    st.subheader(f"📊 {ticker} 股价走势（6 个月）")
    st.line_chart(tech["chart_df"], height=280)

    c1, c2, c3 = st.columns(3)

    with c1:
        st.markdown("### 📊 技术面")
        st.metric("现价", tech["price"])
        st.metric("RSI(14)", tech["rsi"], help=tech["rsi_signal"])
        st.write(f"**RSI 状态**：{tech['rsi_signal']}")
        st.write(f"**MACD 金叉 / 死叉**：{tech['macd_bullish_cross']} / {tech['macd_bearish_cross']}")
        st.write(f"**价格 > MA20**：{tech['price_above_ma20']}")
        st.write(f"**MA20 > MA50**：{tech['ma20_above_ma50']}")
        st.write(f"**布林带位置**：{tech['bb_position']}（宽度 {tech['bb_width']}%）")
        if tech["vol_ratio"] is not None:
            st.write(f"**成交量比（5日/20日）**：{tech['vol_ratio']}")
        if tech["atr_pct"] is not None:
            st.write(f"**ATR 波动率**：{tech['atr_pct']}%")
        st.write(f"**距 20 日高 / 低**：{tech['dist_high20']}% / {tech['dist_low20']}%")

    with c2:
        st.markdown("### 🎯 期权情绪")
        if isinstance(opt, dict) and "error" not in opt:
            st.metric("Put/Call 成交量比率", opt["volume_put_call_ratio"])
            st.metric("Put/Call 持仓量比率", opt["oi_put_call_ratio"])
            st.write(f"**Put / Call 成交量**：{opt['put_volume']} / {opt['call_volume']}")
            st.write(f"**Put / Call 未平仓**：{opt['put_oi']} / {opt['call_oi']}")
            st.write(f"**情绪判断**：`{opt['sentiment']}`")
            if opt.get("divergence"):
                st.warning(f"⚠️ {opt['divergence']}")
            if opt.get("put_iv") is not None and opt.get("call_iv") is not None:
                st.write(f"**Put IV / Call IV**：{opt['put_iv']}% / {opt['call_iv']}%")
                if opt.get("iv_skew_pct") is not None:
                    st.write(f"**IV 偏斜**：{opt['iv_skew_pct']}%")
                iv_df = pd.DataFrame({
                    "IV": [opt["put_iv"], opt["call_iv"]],
                }, index=["Put IV", "Call IV"])
                st.bar_chart(iv_df, height=150)
            if opt.get("used_expiries"):
                st.caption(f"已汇总到期日：{', '.join(opt['used_expiries'])}")
        else:
            st.error(opt.get("error", "期权数据不可用"))
            if opt.get("raw_errors"):
                with st.expander("详细错误"):
                    for e in opt["raw_errors"]:
                        st.text(e)

    with c3:
        st.markdown("### 📉 沽空数据")
        if isinstance(short, dict) and "error" not in short:
            spf = short["short_pct_float"]
            st.metric("Short % of Float", f"{spf}%" if spf is not None else "N/A")
            d2c = short["short_ratio_days_to_cover"]
            st.metric("Days to Cover", d2c if d2c is not None else "N/A")
            st.write(f"**空头持仓**：{short['shares_short']}")
            st.write(f"**上期持仓**：{short['shares_short_prior']}")
            st.write(f"**持仓上升 / 下降**：{short['short_interest_rising']} / {short['short_interest_falling']}")
            if short.get("daily_short_volume_trend"):
                st.write(f"**每日沽空量趋势**：{short['daily_short_volume_trend']}")
            st.write(f"**信号**：`{', '.join(short['signals'])}`")
            st.caption(f"数据源：{short.get('source')}")
            if short.get("note"):
                st.info(short["note"])
        else:
            st.error(short.get("error", "沽空数据不可用"))

    st.divider()
    st.markdown("## 📈 综合分析结果")

    r1, r2, r3 = st.columns(3)
    with r1:
        st.metric("🟢 买入概率", f"{result['buy_probability']}%")
        st.progress(result["buy_probability"] / 100)
    with r2:
        st.metric("🟡 卖出概率", f"{result['sell_probability']}%")
        st.progress(result["sell_probability"] / 100)
    with r3:
        st.metric("🔴 沽空概率", f"{result['short_probability']}%")
        st.progress(result["short_probability"] / 100)

    st.success(f"➡️ 概率最高方向：**{result['top_direction']}**")
    st.caption(f"原始分数：{result['raw_scores']}")

    if result.get("details"):
        st.markdown("### 🔍 信号明细")
        det_df = pd.DataFrame(
            result["details"],
            columns=["维度", "信号", "买入加分", "卖出加分", "沽空加分"],
        )
        st.dataframe(det_df, use_container_width=True, hide_index=True)

    with st.expander("📋 查看完整数据（JSON）"):
        st.json({
            "ticker": ticker,
            "technicals": {k: v for k, v in tech.items() if k != "chart_df"},
            "options": opt,
            "short": short,
            "result": result,
        })

st.divider()
st.caption("⚠️ 本工具仅供参考，不构成任何投资建议。期权及沽空数据可能延迟或不完整，实际交易前请自行核实。")
