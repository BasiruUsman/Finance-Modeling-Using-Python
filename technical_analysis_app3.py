#Standard libs
import datetime
import io

#Third-party
import requests
import pandas as pd
import yfinance as yf
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# =========================
# Utility / Cache functions
# =========================
@st.cache_data(ttl=24*60*60)  # cache for 1 day
def get_sp500_components():
    """Download S&P 500 constituents (Symbol, Security) from Wikipedia."""
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/124.0 Safari/537.36"
        )
    }
    resp = requests.get(url, headers=headers, timeout=30)
    resp.raise_for_status()

    tables = pd.read_html(io.StringIO(resp.text), attrs={"id": "constituents"})
    df = tables[0]

    # Normalize tickers for Yahoo Finance (e.g., BRK.B -> BRK-B)
    df["Symbol"] = (
        df["Symbol"].astype(str)
        .str.replace(".", "-", regex=False)
        .str.strip()
    )
    df["Security"] = df["Security"].astype(str).str.strip()

    tickers = df["Symbol"].to_list()
    tickers_companies_dict = dict(zip(df["Symbol"], df["Security"]))
    return tickers, tickers_companies_dict


@st.cache_data(ttl=24*60*60)
def load_data(symbol: str, start: datetime.date, end: datetime.date) -> pd.DataFrame:
    """Download OHLCV for a single ticker via yfinance."""
    df = yf.download(symbol, start, end, auto_adjust=False)
    # Flatten possible MultiIndex columns
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    # Standardize columns
    keep = [c for c in ["Open", "High", "Low", "Close", "Volume"] if c in df.columns]
    return df[keep].dropna(how="any")


@st.cache_data
def convert_df_to_csv(df: pd.DataFrame) -> bytes:
    return df.to_csv().encode("utf-8")


# =========================
# Indicator calculations
# =========================
def sma(series: pd.Series, window: int) -> pd.Series:
    return series.rolling(window=window, min_periods=window).mean()

def bollinger_bands(series: pd.Series, window: int = 20, num_std: int = 2):
    ma = sma(series, window)
    sd = series.rolling(window=window, min_periods=window).std()
    upper = ma + num_std * sd
    lower = ma - num_std * sd
    return ma, upper, lower

def rsi(series: pd.Series, window: int = 14) -> pd.Series:
    # Classic Wilder's RSI
    delta = series.diff()
    up = delta.clip(lower=0)
    down = -delta.clip(upper=0)

    # Wilder's smoothing
    gain = up.ewm(alpha=1/window, adjust=False).mean()
    loss = down.ewm(alpha=1/window, adjust=False).mean()

    rs = gain / loss
    rsi_ = 100 - (100 / (1 + rs))
    return rsi_


# ============
# Streamlit UI
# ============
st.set_page_config(page_title="Technical Analysis App", layout="wide")
st.title("A simple web app for technical analysis")
st.write("""
### User manual
* Choose any company from the S&P 500 constituents.
* Toggle indicators in the sidebar.
""")

# Sidebar: stock + date range
st.sidebar.header("Stock Parameters")
available_tickers, tickers_companies_dict = get_sp500_components()
ticker = st.sidebar.selectbox(
    "Ticker",
    available_tickers,
    format_func=tickers_companies_dict.get,
    index=available_tickers.index("AAPL") if "AAPL" in available_tickers else 0
)
start_date = st.sidebar.date_input("Start date", datetime.date(2019, 1, 1))
end_date = st.sidebar.date_input("End date", datetime.date.today())
if start_date > end_date:
    st.sidebar.error("The end date must fall after the start date")

# Sidebar: TA params
st.sidebar.header("Technical Analysis Parameters")
volume_flag = st.sidebar.checkbox(label="Add volume", value=True)

exp_sma = st.sidebar.expander("SMA", expanded=False)
sma_flag = exp_sma.checkbox(label="Add SMA", value=True)
sma_periods = exp_sma.number_input("SMA Periods", 1, 200, 20, 1)

exp_bb = st.sidebar.expander("Bollinger Bands", expanded=False)
bb_flag = exp_bb.checkbox(label="Add Bollinger Bands", value=False)
bb_periods = exp_bb.number_input("BB Periods", 1, 200, 20, 1)
bb_std = exp_bb.number_input("# of standard deviations", 1, 4, 2, 1)

exp_rsi = st.sidebar.expander("Relative Strength Index", expanded=False)
rsi_flag = exp_rsi.checkbox(label="Add RSI", value=False)
rsi_periods = exp_rsi.number_input("RSI Periods", 1, 200, 14, 1)
rsi_upper = exp_rsi.number_input("RSI Upper", 50, 90, 70, 1)
rsi_lower = exp_rsi.number_input("RSI Lower", 10, 50, 30, 1)

# Load data
df = load_data(ticker, start_date, end_date)

# Data preview + download
data_exp = st.expander("Preview data", expanded=False)
if df.empty:
    data_exp.info("No data for the selected parameters.")
else:
    available_cols = df.columns.tolist()
    columns_to_show = data_exp.multiselect(
        "Columns",
        available_cols,
        default=available_cols
    )
    data_exp.dataframe(df[columns_to_show])
    csv_file = convert_df_to_csv(df[columns_to_show])
    data_exp.download_button(
        label="Download selected as CSV",
        data=csv_file,
        file_name=f"{ticker}_stock_prices.csv",
        mime="text/csv",
    )

# Guard: ensure OHLC present
required = {"Open", "High", "Low", "Close"}
if not required.issubset(df.columns):
    st.warning("Could not find required OHLC columns in the downloaded data. Try a different date range or ticker.")
    st.stop()

# =========================
# Build Plotly figure
# =========================
# Determine if we need 2 or 3 rows (RSI makes a 3rd)
rows = 3 if rsi_flag else 2
row_heights = [0.65, 0.35] if rows == 2 else [0.55, 0.20, 0.25]

fig = make_subplots(
    rows=rows,
    cols=1,
    shared_xaxes=True,
    vertical_spacing=0.03,
    row_heights=row_heights,
    specs=[[{"secondary_y": True}], [{"secondary_y": False}]] + ([[{"secondary_y": False}]] if rows == 3 else [])
)

# --- Price (candlestick) ---
fig.add_trace(
    go.Candlestick(
        x=df.index,
        open=df["Open"], high=df["High"], low=df["Low"], close=df["Close"],
        name="Price"
    ),
    row=1, col=1, secondary_y=False
)

# --- SMA ---
if sma_flag and sma_periods > 0:
    df[f"SMA_{sma_periods}"] = sma(df["Close"], int(sma_periods))
    fig.add_trace(
        go.Scatter(
            x=df.index, y=df[f"SMA_{sma_periods}"],
            name=f"SMA {int(sma_periods)}",
            mode="lines"
        ),
        row=1, col=1, secondary_y=False
    )

# --- Bollinger Bands ---
if bb_flag and bb_periods > 0:
    ma, upper, lower = bollinger_bands(df["Close"], int(bb_periods), int(bb_std))
    fig.add_trace(
        go.Scatter(x=df.index, y=upper, name=f"BB Upper ({int(bb_periods)},{int(bb_std)}σ)", mode="lines"),
        row=1, col=1, secondary_y=False
    )
    fig.add_trace(
        go.Scatter(x=df.index, y=ma, name=f"BB Middle ({int(bb_periods)})", mode="lines"),
        row=1, col=1, secondary_y=False
    )
    fig.add_trace(
        go.Scatter(x=df.index, y=lower, name=f"BB Lower ({int(bb_periods)},{int(bb_std)}σ)", mode="lines", fill=None),
        row=1, col=1, secondary_y=False
    )

# --- Volume (bars) ---
if volume_flag and "Volume" in df.columns:
    fig.add_trace(
        go.Bar(x=df.index, y=df["Volume"], name="Volume", opacity=0.6),
        row=2, col=1, secondary_y=False
    )

# --- RSI (own panel) ---
if rsi_flag:
    df[f"RSI_{rsi_periods}"] = rsi(df["Close"], int(rsi_periods))
    fig.add_trace(
        go.Scatter(
            x=df.index, y=df[f"RSI_{rsi_periods}"],
            name=f"RSI {int(rsi_periods)}", mode="lines"
        ),
        row=3, col=1
    )
    # Upper/lower bands as dashed lines
    fig.add_trace(
        go.Scatter(x=df.index, y=[float(rsi_upper)]*len(df), name=f"RSI Upper {int(rsi_upper)}",
                   mode="lines", line=dict(dash="dash")),
        row=3, col=1
    )
    fig.add_trace(
        go.Scatter(x=df.index, y=[float(rsi_lower)]*len(df), name=f"RSI Lower {int(rsi_lower)}",
                   mode="lines", line=dict(dash="dash")),
        row=3, col=1
    )

# Layout
company_name = tickers_companies_dict.get(ticker, ticker)
fig.update_layout(
    title=f"{company_name} ({ticker})",
    xaxis=dict(rangeslider=dict(visible=False)),
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
    margin=dict(l=10, r=10, t=60, b=10),
    hovermode="x unified"
)

# Axis titles
fig.update_yaxes(title_text="Price", row=1, col=1)
if volume_flag:
    fig.update_yaxes(title_text="Volume", row=2, col=1)
if rsi_flag:
    fig.update_yaxes(title_text="RSI", range=[0, 100], row=3, col=1)

st.plotly_chart(fig, use_container_width=True)
