
import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
import datetime as dt

# Optional online fetch (yfinance); app will still work without internet by asking for CSV upload.
try:
    import yfinance as yf
    YF_OK = True
except Exception:
    YF_OK = False

st.set_page_config(page_title="K-Market Research Engine v0.3", layout="wide")
st.title("📈 K-Market Research Engine v0.3")
st.caption("GitHub+Streamlit에서 바로 사용: 종목 & 테마만 입력하면 특징 생성 → 학습 → 예측까지")

# ===== Config =====
FEATURES = [
    "foreign_net20","inst_net20",
    "candle_psych_prev","ai_risk_level",
    "theme_strength",
    "kospi_ret","usdkrw_change","wti_change",
    "news_industry","news_economy","news_breaking",
    "news_policy","news_global","news_finance","news_company",
]
LABEL = "target_next_1d_up"
MODEL_PATH = "model.pkl"

# ===== Helpers =====
def build_from_prices(df_prices, theme_strength_value=3, macros=None, news=None):
    """Create feature frame from OHLC dataframe with index=Date and columns: Open,High,Low,Close, Volume"""
    df = df_prices.copy().reset_index().rename(columns=str.lower)
    df["date"] = pd.to_datetime(df["date"]).dt.tz_localize(None)
    df["ticker"] = st.session_state.get("ticker","TICKER")

    # Foreign / Inst flows are optional. If not provided, set to 0.
    df["foreign_net"] = 0.0
    df["inst_net"] = 0.0
    # 20-day sums
    df["foreign_net20"] = df["foreign_net"].rolling(20).sum()
    df["inst_net20"]    = df["inst_net"].rolling(20).sum()

    # Candle psychology (very simple)
    body = (df["close"] - df["open"]).abs()
    rng  = (df["high"] - df["low"]).replace(0, np.nan)
    ratio = (body / rng).fillna(0)
    cond_up = df["close"] > df["open"]
    df["candle_psych_prev"] = np.where((cond_up) & (ratio>0.6), 5,
                                np.where((cond_up), 4,
                                np.where(ratio<0.2, 3,
                                np.where(~cond_up, 2, 3)))).astype(int)

    # AI risk: rolling 10d realized volatility percentile
    ret = df["close"].pct_change()
    vol10 = ret.rolling(10).std()
    rank = vol10.rank(pct=True)
    df["ai_risk_level"] = pd.cut(rank, bins=[0,.2,.4,.6,.8,1.0], labels=[1,2,3,4,5], include_lowest=True).astype("Int64").fillna(3)

    # Theme strength (manual input or simple proxy vs market)
    df["theme_strength"] = int(theme_strength_value)

    # Macros
    if macros is not None:
        m = macros.copy()
        for c in ["kospi","usdkrw","wti"]:
            if c not in m.columns: m[c] = np.nan
        df = df.merge(m[["date","kospi","usdkrw","wti"]], on="date", how="left")
        df["kospi_ret"]     = df["kospi"].pct_change()
        df["usdkrw_change"] = df["usdkrw"].pct_change()
        df["wti_change"]    = df["wti"].pct_change()
    else:
        df["kospi_ret"]     = 0.0
        df["usdkrw_change"] = 0.0
        df["wti_change"]    = 0.0

    # News (set to 0 unless provided)
    for k in ["industry","economy","breaking","policy","global","finance","company"]:
        df[f"news_{k}"] = 0.0

    # label
    df[LABEL] = (df["close"].shift(-1) > df["close"]).astype(int)

    base_cols = ["date","ticker"] + FEATURES
    train_cols = base_cols + [LABEL]
    train = df[train_cols].dropna().copy()
    infer = df[base_cols].dropna().copy()
    return train, infer

def fetch_prices_yf(ticker, period="6mo"):
    obj = yf.Ticker(ticker)
    df = obj.history(period=period, interval="1d")
    df = df.reset_index().rename(columns={"Date":"date","Open":"open","High":"high","Low":"low","Close":"close","Volume":"volume"})
    df["date"] = pd.to_datetime(df["date"]).dt.tz_localize(None)
    df = df.set_index("date")[["open","high","low","close","volume"]]
    return df

def fetch_macros_yf(period="6mo"):
    # KOSPI (^KS11), USD/KRW (KRW=X), WTI (CL=F)
    macros = {}
    for sym, name in {"^KS11":"kospi", "KRW=X":"usdkrw", "CL=F":"wti"}.items():
        try:
            df = yf.Ticker(sym).history(period=period, interval="1d")[["Close"]].rename(columns={"Close":name})
            df.index = pd.to_datetime(df.index).tz_localize(None)
            macros[name] = df
        except Exception:
            macros[name] = pd.DataFrame(columns=[name])
    out = None
    for name,df in macros.items():
        df = df.copy()
        df["date"] = df.index
        out = df if out is None else out.merge(df, on="date", how="outer")
    if out is None:
        return None
    out = out.sort_values("date").fillna(method="ffill")
    return out

# ===== Sidebar Inputs =====
with st.sidebar:
    st.header("입력")
    st.session_state["ticker"] = st.text_input("종목 코드/심볼", value="005930")
    theme_name = st.text_input("테마 이름(선택)", value="")
    theme_strength_value = st.slider("테마 강도(1~5)", 1, 5, 3)
    period = st.selectbox("데이터 기간", ["3mo","6mo","1y","2y"], index=1)
    auto_train = st.checkbox("모델 없으면 자동 학습", value=True)

st.subheader("1) 특징 생성 (자동)")

online_ok = YF_OK
if online_ok:
    try:
        prices = fetch_prices_yf(st.session_state["ticker"], period=period)
    except Exception as e:
        online_ok = False
        st.warning(f"온라인 가격 수집 실패: {e}")
else:
    st.info("yfinance 미설치 또는 네트워크 제한으로 온라인 수집 비활성화. CSV 업로드를 이용하세요.")

if online_ok:
    st.success("✅ 가격 데이터 수집 완료")
    try:
        macros = fetch_macros_yf(period=period)
        st.info("거시(KOSPI/환율/WTI) 동기화 완료")
    except Exception as e:
        macros = None
        st.warning(f"거시 지표 수집 실패: {e}")
    tr, inf = build_from_prices(prices, theme_strength_value=theme_strength_value, macros=macros)
else:
    # fallback: ask upload OHLC CSV
    st.subheader("수동 업로드 (대안)")
    up = st.file_uploader("OHLC CSV 업로드 (date, open, high, low, close 필수)", type=["csv"])
    tr=inf=None
    if up is not None:
        df = pd.read_csv(up)
        df["date"] = pd.to_datetime(df["date"])
        df = df.set_index("date")[["open","high","low","close"]]
        tr, inf = build_from_prices(df, theme_strength_value=theme_strength_value, macros=None)

if tr is not None and inf is not None:
    st.write("**학습/예측용 미리보기**")
    st.dataframe(inf.tail(10))
    # Save temp
    Path("data").mkdir(exist_ok=True)
    tr.to_csv("data/_auto_train.csv", index=False, encoding="utf-8-sig")
    inf.to_csv("data/_auto_infer.csv", index=False, encoding="utf-8-sig")

    # ===== Model =====
    from sklearn.ensemble import RandomForestClassifier
    import joblib

    if Path(MODEL_PATH).exists():
        model = joblib.load(MODEL_PATH)
    elif auto_train:
        X = tr[FEATURES].astype(float); y = tr[LABEL].astype(int)
        model = RandomForestClassifier(
            n_estimators=400, min_samples_split=4, min_samples_leaf=2, n_jobs=-1, random_state=42,
            class_weight="balanced_subsample"
        ).fit(X,y)
        joblib.dump(model, MODEL_PATH)
        st.success("✅ 자동 학습 완료 (model.pkl 저장)")
    else:
        model=None
        st.error("모델이 없습니다. 자동 학습을 켜거나 model.pkl을 업로드하세요.")

    # optional: model upload
    with st.expander("모델 업로드/관리"):
        uploaded_model = st.file_uploader("model.pkl 업로드(선택)", type=["pkl"], key="model_up")
        if uploaded_model is not None:
            with open(MODEL_PATH, "wb") as f:
                f.write(uploaded_model.read())
            st.info("업로드된 model.pkl 저장 완료. 페이지를 다시 실행하면 반영됩니다.")

    # ===== Predict =====
    if model is not None:
        st.subheader("2) 예측")
        X_inf = inf[FEATURES].astype(float)
        prob = model.predict_proba(X_inf)[:,1]
        out = inf.copy()
        out["prob_up"] = prob
        def prob_to_signal(p):
            if p >= 0.62: return "BUY"
            if p >= 0.50: return "HOLD"
            return "AVOID"
        out["signal"] = [prob_to_signal(p) for p in prob]
        st.success("✅ 예측 완료")
        st.dataframe(out.tail(20))
        st.download_button("CSV 다운로드", data=out.to_csv(index=False).encode("utf-8-sig"),
                           file_name=f"{st.session_state['ticker']}_predictions.csv", mime="text/csv")
else:
    st.warning("데이터를 수집/업로드하여 특징을 생성하세요.")
