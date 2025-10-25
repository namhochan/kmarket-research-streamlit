
import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
import datetime as dt

# optional online fetch
try:
    import yfinance as yf
    YF_OK = True
except Exception:
    YF_OK = False

st.set_page_config(page_title="K-Market Research Engine v0.3.1", layout="wide")
st.title("📈 K-Market Research Engine v0.3.1")
st.caption("종목/테마만 입력 → 특징 생성 → 학습/예측. (결측/표본/라벨 검사 강화판)")

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

def build_from_prices(df_prices, theme_strength_value=3, macros=None, news=None, ticker="TICKER"):
    df = df_prices.copy().reset_index().rename(columns=str.lower)
    if "date" not in df.columns:
        df.rename(columns={"index":"date"}, inplace=True)
    df["date"] = pd.to_datetime(df["date"]).dt.tz_localize(None)
    df["ticker"] = ticker

    # flows default 0
    df["foreign_net"] = 0.0
    df["inst_net"] = 0.0
    df["foreign_net20"] = pd.Series(df["foreign_net"]).rolling(20).sum()
    df["inst_net20"]    = pd.Series(df["inst_net"]).rolling(20).sum()

    # candle psychology
    body = (df["close"] - df["open"]).abs()
    rng  = (df["high"] - df["low"]).replace(0, np.nan)
    ratio = (body / rng).fillna(0)
    cond_up = df["close"] > df["open"]
    df["candle_psych_prev"] = np.where((cond_up) & (ratio>0.6), 5,
                                np.where((cond_up), 4,
                                np.where(ratio<0.2, 3,
                                np.where(~cond_up, 2, 3)))).astype(int)

    # AI risk
    ret = df["close"].pct_change()
    vol10 = ret.rolling(10).std()
    rank = vol10.rank(pct=True)
    df["ai_risk_level"] = pd.cut(rank, bins=[0,.2,.4,.6,.8,1.0], labels=[1,2,3,4,5], include_lowest=True).astype("Int64").fillna(3)

    # theme
    df["theme_strength"] = int(theme_strength_value)

    # macros
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

    # news defaults
    for k in ["industry","economy","breaking","policy","global","finance","company"]:
        df[f"news_{k}"] = 0.0

    # label
    df[LABEL] = (df["close"].shift(-1) > df["close"]).astype(int)

    base_cols = ["date","ticker"] + FEATURES
    train_cols = base_cols + [LABEL]
    train = df[train_cols].copy()
    infer = df[base_cols].copy()

    # safety: replace inf/nan
    train = train.replace([np.inf,-np.inf], np.nan).fillna(0.0)
    infer = infer.replace([np.inf,-np.inf], np.nan).fillna(0.0)
    # drop first few rows with incomplete rolling
    train = train.iloc[25:].reset_index(drop=True)
    infer = infer.iloc[25:].reset_index(drop=True)
    return train, infer

def fetch_prices_yf(ticker, period="6mo"):
    obj = yf.Ticker(ticker)
    df = obj.history(period=period, interval="1d")
    if df is None or len(df)==0:
        raise ValueError("가격 데이터를 찾지 못했습니다. 심볼을 확인하세요 (예: 005930.KS / 247660.KQ).")
    df = df.reset_index().rename(columns={"Date":"date","Open":"open","High":"high","Low":"low","Close":"close","Volume":"volume"})
    df["date"] = pd.to_datetime(df["date"]).dt.tz_localize(None)
    return df.set_index("date")[["open","high","low","close","volume"]]

def fetch_macros_yf(period="6mo"):
    d = {}
    for sym, name in {"^KS11":"kospi","KRW=X":"usdkrw","CL=F":"wti"}.items():
        try:
            tmp = yf.Ticker(sym).history(period=period, interval="1d")[["Close"]].rename(columns={"Close":name})
            tmp.index = pd.to_datetime(tmp.index).tz_localize(None)
            d[name] = tmp
        except Exception:
            d[name] = pd.DataFrame(columns=[name])
    out=None
    for name, df in d.items():
        df=df.copy(); df["date"]=df.index
        out = df if out is None else out.merge(df, on="date", how="outer")
    if out is None: return None
    out = out.sort_values("date").fillna(method="ffill")
    return out

with st.sidebar:
    st.header("입력")
    ticker = st.text_input("종목 코드/심볼", value="005930.KS")
    theme_strength_value = st.slider("테마 강도(1~5)", 1, 5, 3)
    period = st.selectbox("데이터 기간", ["3mo","6mo","1y","2y"], index=2)
    auto_train = st.checkbox("모델 없으면 자동 학습", value=True)

st.subheader("1) 특징 생성 (자동)")

if YF_OK:
    try:
        prices = fetch_prices_yf(ticker, period=period)
        st.success("✅ 가격 데이터 수집 완료")
        macros = fetch_macros_yf(period=period)
        st.info("거시(KOSPI/환율/WTI) 동기화")
        tr, inf = build_from_prices(prices, theme_strength_value=theme_strength_value, macros=macros, ticker=ticker)
    except Exception as e:
        st.error(f"온라인 수집 실패: {e}")
        tr=inf=None
else:
    st.warning("yfinance 미설치/네트워크 제한. CSV 업로드로 진행하세요.")
    tr=inf=None

# CSV 대안 업로드
if tr is None or inf is None:
    up = st.file_uploader("OHLC CSV 업로드 (date, open, high, low, close)", type=["csv"])
    if up is not None:
        df = pd.read_csv(up)
        df["date"] = pd.to_datetime(df["date"])
        df = df.set_index("date")[["open","high","low","close"]]
        tr, inf = build_from_prices(df, theme_strength_value=theme_strength_value, macros=None, ticker=ticker)

if tr is not None and inf is not None and len(tr)>0 and len(inf)>0:
    st.write("**학습/예측용 미리보기 (하위 10행)**")
    st.dataframe(inf.tail(10))
    Path("data").mkdir(exist_ok=True)
    tr.to_csv("data/_auto_train.csv", index=False, encoding="utf-8-sig")
    inf.to_csv("data/_auto_infer.csv", index=False, encoding="utf-8-sig")

    from sklearn.ensemble import RandomForestClassifier
    import joblib

    model=None
    if Path(MODEL_PATH).exists():
        model = joblib.load(MODEL_PATH)
        st.success("저장된 모델 로드 완료")
    elif auto_train:
        X = tr[FEATURES].astype(float).replace([np.inf,-np.inf], np.nan).fillna(0.0)
        y = tr[LABEL].astype(int)

        n_samples = len(X)
        n_classes = y.nunique()

        if n_samples < 120:
            st.error(f"학습 표본이 부족합니다({n_samples}행). 사이드바 기간을 1y 이상으로 늘려주세요.")
        elif n_classes < 2:
            st.error("라벨 클래스가 하나뿐입니다(전부 0 또는 전부 1). 기간을 늘리거나 구간을 바꿔주세요.")
        else:
            model = RandomForestClassifier(
                n_estimators=500, min_samples_split=4, min_samples_leaf=2,
                n_jobs=-1, random_state=42, class_weight="balanced_subsample"
            ).fit(X,y)
            joblib.dump(model, MODEL_PATH)
            st.success("✅ 자동 학습 완료 (model.pkl 저장)")

    with st.expander("모델 업로드/관리"):
        uploaded_model = st.file_uploader("model.pkl 업로드(선택)", type=["pkl"], key="model_up_v031")
        if uploaded_model is not None:
            with open(MODEL_PATH, "wb") as f:
                f.write(uploaded_model.read())
            st.info("업로드된 model.pkl 저장 완료. 페이지를 다시 실행하면 반영됩니다.")

    if model is not None:
        st.subheader("2) 예측")
        X_inf = inf[FEATURES].astype(float).replace([np.inf,-np.inf], np.nan).fillna(0.0)
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
                           file_name=f"{ticker}_predictions.csv", mime="text/csv")
else:
    st.warning("데이터 수집/업로드 후 다시 시도하세요.")
