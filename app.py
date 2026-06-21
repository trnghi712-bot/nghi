import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
import time
import random
import os
import io

# Setup page configuration
st.set_page_config(
    page_title="LSTM-GRU Portfolio Optimization",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom styling for a premium look
st.markdown("""
<style>
    .metric-card {
        background-color: #1e293b;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        border: 1px solid #334155;
        text-align: center;
    }
    .metric-value {
        font-size: 24px;
        font-weight: bold;
        color: #38bdf8;
    }
    .metric-label {
        font-size: 14px;
        color: #94a3b8;
        margin-top: 5px;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 24px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        background-color: #0f172a;
        border-radius: 4px 4px 0px 0px;
        gap: 1px;
        padding-top: 10px;
        padding-bottom: 10px;
        font-weight: 600;
        font-size: 16px;
    }
    .stTabs [aria-selected="true"] {
        background-color: #1e293b !important;
        border-bottom: 3px solid #38bdf8 !important;
    }
</style>
""", unsafe_allow_html=True)

# Try to import TensorFlow and Scikit-Learn
try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Input, LSTM, GRU, Dense, Dropout
    from tensorflow.keras.optimizers import Adam
    import tensorflow.keras.backend as K
    from sklearn.preprocessing import StandardScaler
    tf_installed = True
except ImportError:
    tf_installed = False

# Import tickers list
try:
    from industry_tickers import INDUSTRY_TICKERS
except ImportError:
    st.error("Không tìm thấy file industry_tickers.py trong thư mục làm việc!")
    st.stop()

from vnstock import Vnstock

# Title and description
st.title("📈 Tối ưu hóa danh mục đầu tư bằng mạng Neural LSTM-GRU")
st.markdown(
    "Ứng dụng này sử dụng học máy (Deep Learning) để tối ưu hóa trọng số của danh mục đầu tư cổ phiếu "
    "tại thị trường Việt Nam dựa trên hàm mục tiêu Sharpe cải tiến (Sharpe Loss với Entropy Regularization)."
)

if not tf_installed:
    st.error("Các thư viện TensorFlow hoặc Scikit-Learn chưa được cài đặt đầy đủ. Vui lòng cài đặt chúng trước khi sử dụng mô hình học máy.")
    st.stop()

# --- SIDEBAR CONFIGURATIONS ---
st.sidebar.header("⚙️ Cấu hình hệ thống")

# Industry Selection
all_industries = list(INDUSTRY_TICKERS.keys())
selected_industries = st.sidebar.multiselect(
    "1. Chọn nhóm ngành",
    all_industries,
    default=["Thép"]
)

# Risk-free rate
rf_annual = st.sidebar.number_input(
    "2. Tỷ suất sinh lời phi rủi ro năm (Rf)",
    min_value=0.0,
    max_value=0.20,
    value=0.045,
    step=0.005,
    format="%.3f"
)
trading_days = st.sidebar.number_input(
    "3. Số ngày giao dịch trong năm",
    min_value=100,
    max_value=365,
    value=252,
    step=1
)

# Date Selection
st.sidebar.subheader("📅 Khoảng thời gian")
train_start = st.sidebar.date_input("Huấn luyện từ ngày", pd.to_datetime("2015-01-01"))
train_end = st.sidebar.date_input("Huấn luyện đến ngày", pd.to_datetime("2024-12-31"))
test_start = st.sidebar.date_input("Kiểm thử từ ngày", pd.to_datetime("2025-01-01"))
test_end = st.sidebar.date_input("Kiểm thử đến ngày", pd.to_datetime("2025-12-31"))

# Model hyperparameters
st.sidebar.subheader("🧠 Cấu hình mạng Neural")
top_n = st.sidebar.slider("Số cổ phiếu tối ưu (Top N Sharpe)", 3, 20, 10)
window_size = st.sidebar.slider("Kích thước cửa sổ dữ liệu (Lookback Window)", 5, 60, 30)
horizon = st.sidebar.slider("Chỉ số dự báo tương lai (Horizon)", 1, 10, 5)
epochs = st.sidebar.number_input("Số epoch huấn luyện", min_value=5, max_value=200, value=20, step=5)
batch_size = st.sidebar.selectbox("Batch Size", [16, 32, 64, 128], index=1)
entropy_lambda = st.sidebar.number_input("Hệ số Entropy (Regularization)", min_value=0.0, max_value=0.5, value=0.01, step=0.005, format="%.3f")

# Seeds configuration
selected_seeds = st.sidebar.multiselect(
    "Danh sách seed để tối ưu",
    [7, 21, 42, 99, 123],
    default=[7, 21, 42, 99, 123]
)

# Initialize Session States
if "raw_data" not in st.session_state:
    st.session_state["raw_data"] = None
if "selected_tickers" not in st.session_state:
    st.session_state["selected_tickers"] = []
if "best_model" not in st.session_state:
    st.session_state["best_model"] = None
if "best_seed" not in st.session_state:
    st.session_state["best_seed"] = None
if "optimized_weights" not in st.session_state:
    st.session_state["optimized_weights"] = None
if "test_metrics" not in st.session_state:
    st.session_state["test_metrics"] = None
if "results_runs_df" not in st.session_state:
    st.session_state["results_runs_df"] = None
if "top_symbols" not in st.session_state:
    st.session_state["top_symbols"] = []
if "eval_dates" not in st.session_state:
    st.session_state["eval_dates"] = None
if "eval_returns_lstm" not in st.session_state:
    st.session_state["eval_returns_lstm"] = None
if "eval_returns_baseline" not in st.session_state:
    st.session_state["eval_returns_baseline"] = None
if "y_test_df" not in st.session_state:
    st.session_state["y_test_df"] = None

# Custom loss function mapping
def make_sharpe_loss(rf_annual_val, trading_days_val, entropy_lambda_val):
    def loss_fn(y_true, y_pred):
        # y_true: (batch_size, n_assets), y_pred: (batch_size, n_assets)
        portfolio_returns = tf.reduce_sum(y_true * y_pred, axis=1)
        rf_daily = rf_annual_val / trading_days_val
        portfolio_returns = portfolio_returns - rf_daily

        mean_returns = tf.reduce_mean(portfolio_returns)
        std_returns = tf.math.reduce_std(portfolio_returns)

        sharpe = mean_returns / (std_returns + 1e-9)
        entropy = -tf.reduce_sum(y_pred * tf.math.log(y_pred + 1e-9), axis=1)
        entropy = tf.reduce_mean(entropy)

        return -sharpe - entropy_lambda_val * entropy
    return loss_fn

def build_lstm_gru_model(timesteps, n_features, n_assets):
    model = Sequential([
        Input(shape=(timesteps, n_features)),
        LSTM(96, return_sequences=True, activation="tanh", recurrent_activation="sigmoid"),
        Dropout(0.2),
        GRU(48, return_sequences=False, activation="tanh", recurrent_activation="sigmoid"),
        Dropout(0.2),
        Dense(64, activation="relu"),
        Dropout(0.1),
        Dense(n_assets, activation="softmax")
    ])
    return model

def compute_rsi(price_df, period=14):
    delta = price_df.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(period, min_periods=period).mean()
    avg_loss = loss.rolling(period, min_periods=period).mean()
    rs = avg_gain / (avg_loss + 1e-9)
    rsi = 100 - (100 / (1 + rs))
    return rsi

def build_features(price_df, return_df):
    common_idx = price_df.index.intersection(return_df.index)
    price_df = price_df.loc[common_idx].copy()
    return_df = return_df.loc[common_idx].copy()

    price_df = price_df.replace([np.inf, -np.inf], np.nan).ffill().bfill()
    return_df = return_df.replace([np.inf, -np.inf], np.nan).fillna(0)

    feat_list = []
    
    # 1. ret1
    ret_1 = return_df.copy()
    ret_1.columns = [f"{c}_ret1" for c in ret_1.columns]
    feat_list.append(ret_1)

    # 2. ret5
    ret_5 = price_df.pct_change(5)
    ret_5.columns = [f"{c}_ret5" for c in ret_5.columns]
    feat_list.append(ret_5)

    # 3. ret10
    ret_10 = price_df.pct_change(10)
    ret_10.columns = [f"{c}_ret10" for c in ret_10.columns]
    feat_list.append(ret_10)

    # 4. ma5_ratio
    ma5_ratio = price_df / (price_df.rolling(5, min_periods=5).mean() + 1e-9) - 1
    ma5_ratio.columns = [f"{c}_ma5_ratio" for c in ma5_ratio.columns]
    feat_list.append(ma5_ratio)

    # 5. ma10_ratio
    ma10_ratio = price_df / (price_df.rolling(10, min_periods=10).mean() + 1e-9) - 1
    ma10_ratio.columns = [f"{c}_ma10_ratio" for c in ma10_ratio.columns]
    feat_list.append(ma10_ratio)

    # 6. vol5
    vol5 = return_df.rolling(5, min_periods=5).std()
    vol5.columns = [f"{c}_vol5" for c in vol5.columns]
    feat_list.append(vol5)

    # 7. vol10
    vol10 = return_df.rolling(10, min_periods=10).std()
    vol10.columns = [f"{c}_vol10" for c in vol10.columns]
    feat_list.append(vol10)

    # 8. mom5
    mom5 = price_df.pct_change(5)
    mom5.columns = [f"{c}_mom5" for c in mom5.columns]
    feat_list.append(mom5)

    # 9. rsi14
    rsi14 = compute_rsi(price_df, period=14) / 100.0
    rsi14.columns = [f"{c}_rsi14" for c in rsi14.columns]
    feat_list.append(rsi14)

    features = pd.concat(feat_list, axis=1)
    features = features.replace([np.inf, -np.inf], np.nan)
    features = features.dropna(axis=0, how="any")

    return features

def create_sequences_and_targets(features_df, target_returns_df, window_size_val, horizon_val=5):
    X, y, dates = [], [], []
    feat_values = features_df.values.astype(np.float32)
    target_values = target_returns_df.values.astype(np.float32)
    idx = features_df.index

    for i in range(len(features_df) - window_size_val - horizon_val + 1):
        X.append(feat_values[i:i + window_size_val])
        y.append(target_values[i + window_size_val:i + window_size_val + horizon_val].mean(axis=0))
        dates.append(idx[i + window_size_val + horizon_val - 1])

    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32), pd.Index(dates)

# Cache data loading function
@st.cache_data(show_spinner=False)
def load_data_for_tickers(tickers, start, end):
    all_data = []
    failed = []
    
    # vnstock initialization
    vn = Vnstock()
    
    download_progress = st.progress(0)
    status_text = st.empty()
    
    req_count = 0
    window_start = time.time()
    
    for i, t in enumerate(tickers, start=1):
        status_text.write(f"⏳ Đang tải dữ liệu cho {t}... ({i}/{len(tickers)})")
        success = False
        retries = 0
        max_retries = 4
        
        while not success and retries < max_retries:
            try:
                stock_obj = vn.stock(symbol=t, source="KBS")
                df = stock_obj.quote.history(start=start, end=end, interval="1D")
                
                if df is not None and not df.empty:
                    df = df.copy()
                    df["ticker"] = t
                    
                    if "time" not in df.columns:
                        if "date" in df.columns:
                            df["time"] = df["date"]
                        elif "datetime" in df.columns:
                            df["time"] = df["datetime"]
                        else:
                            df["time"] = df.index
                            
                    keep_cols = [c for c in ["time", "open", "high", "low", "close", "volume", "ticker"] if c in df.columns]
                    df = df[keep_cols]
                    all_data.append(df)
                    success = True
                else:
                    retries += 1
                    time.sleep(1.5)
            except Exception as e:
                retries += 1
                msg = str(e).lower()
                if "rate limit" in msg or "429" in msg or "forbidden" in msg:
                    time.sleep(10)
                else:
                    time.sleep(2)
        
        if not success:
            failed.append(t)
            
        download_progress.progress(i / len(tickers))
        
        # Rate limit prevention (sleep 3s)
        time.sleep(3.2)
        req_count += 1
        
        if req_count >= 18:
            elapsed = time.time() - window_start
            if elapsed < 60:
                sleep_time = 60 - elapsed
                for s in range(int(sleep_time), 0, -1):
                    status_text.write(f"⏳ Tạm dừng {s}s để tránh rate limit của API...")
                    time.sleep(1)
            req_count = 0
            window_start = time.time()
            
    status_text.empty()
    download_progress.empty()
    
    if all_data:
        raw_df = pd.concat(all_data, ignore_index=True)
        return raw_df, failed
    else:
        return None, failed

# Keras training callback for Streamlit progress
class StreamlitCallback(tf.keras.callbacks.Callback):
    def __init__(self, progress_bar, status_text, max_epochs, seed, run_info_text):
        self.progress_bar = progress_bar
        self.status_text = status_text
        self.max_epochs = max_epochs
        self.seed = seed
        self.run_info = run_info_text

    def on_epoch_end(self, epoch, logs=None):
        loss = logs.get("loss", 0.0)
        val_loss = logs.get("val_loss", 0.0)
        pct = (epoch + 1) / self.max_epochs
        self.progress_bar.progress(pct)
        self.status_text.text(f"Seed {self.seed} - Epoch {epoch+1}/{self.max_epochs}: Loss = {loss:.4f}, Val Loss = {val_loss:.4f}")

# Main Tabs Setup
tab1, tab2, tab3 = st.tabs([
    "📂 1. Thu thập & Khám phá dữ liệu",
    "🧠 2. Huấn luyện Mô hình LSTM-GRU",
    "📊 3. Tối ưu hóa & Đánh giá danh mục"
])

# --- TAB 1: DATA EXPLORATION ---
with tab1:
    st.header("📂 Thu thập và Phân tích Dữ liệu")
    if not selected_industries:
        st.warning("Vui lòng chọn ít nhất một nhóm ngành trong thanh cấu hình bên trái.")
    else:
        # Get all tickers in selected industries
        tickers = []
        for ind in selected_industries:
            tickers.extend(INDUSTRY_TICKERS[ind])
        tickers = sorted(list(set(tickers)))
        
        st.info(f"Đã chọn nhóm ngành: **{', '.join(selected_industries)}**. Tổng số mã: **{len(tickers)}**.")
        
        if st.button("📥 Tải dữ liệu từ vnstock", key="btn_download"):
            with st.spinner("Đang tải dữ liệu giá từ KBS. Vui lòng chờ..."):
                raw_df, failed = load_data_for_tickers(tickers, train_start.strftime("%Y-%m-%d"), test_end.strftime("%Y-%m-%d"))
                
            if raw_df is not None:
                st.session_state["raw_data"] = raw_df
                st.session_state["selected_tickers"] = tickers
                st.success(f"Tải thành công dữ liệu cho {len(tickers) - len(failed)} mã. Lỗi: {len(failed)} mã.")
                if failed:
                    st.warning(f"Mã lỗi hoặc không có dữ liệu: {', '.join(failed)}")
            else:
                st.error("Không tải được dữ liệu cho bất kỳ mã nào. Vui lòng thử lại sau.")

        if st.session_state["raw_data"] is not None:
            raw_data = st.session_state["raw_data"]
            
            st.subheader("Bảng dữ liệu thô (Raw Data)")
            st.dataframe(raw_data.head(10), use_container_width=True)
            
            # Pivot table
            pivot_df_clean = raw_data.pivot_table(
                index="time",
                columns="ticker",
                values="close",
                aggfunc="last"
            ).sort_index()
            pivot_df_clean.index = pd.to_datetime(pivot_df_clean.index)
            
            # Calculate Sharpe ratios
            price_filled = pivot_df_clean.ffill().bfill()
            returns_df = price_filled.pct_change().replace([np.inf, -np.inf], np.nan).dropna(how="any")
            
            rf_daily = rf_annual / trading_days
            mean_ret = returns_df.mean()
            std_ret = returns_df.std().replace(0, np.nan)
            sharpe_ratio = ((mean_ret - rf_daily) / std_ret).dropna().sort_values(ascending=False)
            
            # Save top symbols
            top_symbols = sharpe_ratio.head(top_n).index.tolist()
            st.session_state["top_symbols"] = top_symbols
            
            st.subheader("📊 Tỷ số Sharpe Ratio lịch sử của các mã")
            
            # Bar chart of Sharpe ratios
            fig_sharpe = px.bar(
                x=sharpe_ratio.index,
                y=sharpe_ratio.values,
                labels={"x": "Mã cổ phiếu", "y": "Tỷ số Sharpe Ratio"},
                title="Sharpe Ratio lịch sử của toàn bộ mã trong ngành",
                color=sharpe_ratio.values,
                color_continuous_scale="Viridis"
            )
            fig_sharpe.update_layout(xaxis_tickangle=-45)
            st.plotly_chart(fig_sharpe, use_container_width=True)
            
            # Highlight selected Top N
            st.markdown(f"**Top {top_n} mã có Sharpe Ratio cao nhất được chọn để huấn luyện mô hình:**")
            st.write(", ".join(top_symbols))
            
            # Line chart of normalized prices
            st.subheader("📈 Giá chuẩn hóa (Normalized Price) của Top cổ phiếu đã chọn")
            top_prices = price_filled[top_symbols]
            normalized_prices = top_prices / top_prices.iloc[0]
            
            fig_prices = px.line(
                normalized_prices,
                x=normalized_prices.index,
                y=normalized_prices.columns,
                labels={"time": "Ngày", "value": "Giá chuẩn hóa (Base = 1)"},
                title=f"Diễn biến giá chuẩn hóa của Top {top_n} mã"
            )
            st.plotly_chart(fig_prices, use_container_width=True)

# --- TAB 2: MODEL TRAINING ---
with tab2:
    st.header("🧠 Huấn luyện Mô hình Học Máy LSTM-GRU")
    
    if st.session_state["raw_data"] is None or not st.session_state["top_symbols"]:
        st.warning("Vui lòng tải dữ liệu ở Tab 1 trước khi sang Tab này.")
    else:
        top_symbols = st.session_state["top_symbols"]
        raw_data = st.session_state["raw_data"]
        
        # Prepare datasets
        pivot_df_clean = raw_data.pivot_table(
            index="time",
            columns="ticker",
            values="close",
            aggfunc="last"
        ).sort_index()
        pivot_df_clean.index = pd.to_datetime(pivot_df_clean.index)
        
        price_top10 = pivot_df_clean[top_symbols].copy()
        
        # Split train & test
        train_prices = price_top10.loc[train_start.strftime("%Y-%m-%d"):train_end.strftime("%Y-%m-%d")].copy()
        test_prices  = price_top10.loc[test_start.strftime("%Y-%m-%d"):test_end.strftime("%Y-%m-%d")].copy()
        
        # Check validation dates size
        if len(test_prices) < (window_size + horizon):
            st.error(f"Tập kiểm thử quá ngắn ({len(test_prices)} ngày). Cần tối thiểu {window_size + horizon} ngày để chạy kiểm thử.")
            st.stop()
            
        train_prices = train_prices.sort_index().ffill().bfill()
        test_prices  = test_prices.sort_index().ffill().bfill()
        
        train_returns = train_prices.pct_change().replace([np.inf, -np.inf], np.nan).dropna(how="any")
        test_returns  = test_prices.pct_change().replace([np.inf, -np.inf], np.nan).dropna(how="any")
        
        if st.button("🚀 Bắt đầu huấn luyện mô hình", key="btn_train"):
            st.info("Đang tạo đặc trưng kỹ thuật & chuẩn hóa dữ liệu...")
            
            # Generate features
            train_features = build_features(train_prices, train_returns)
            test_features = build_features(test_prices, test_returns)
            
            # Scale features
            scaler = StandardScaler()
            train_features_scaled = pd.DataFrame(
                scaler.fit_transform(train_features),
                index=train_features.index,
                columns=train_features.columns
            )
            test_features_scaled = pd.DataFrame(
                scaler.transform(test_features),
                index=test_features.index,
                columns=test_features.columns
            )
            
            train_target_returns = train_returns.loc[train_features_scaled.index].copy()
            test_target_returns = test_returns.loc[test_features_scaled.index].copy()
            
            # Create sequences
            X_train, y_train_target, train_seq_dates = create_sequences_and_targets(
                train_features_scaled, train_target_returns, window_size, horizon=horizon
            )
            X_test, y_test_target, test_seq_dates = create_sequences_and_targets(
                test_features_scaled, test_target_returns, window_size, horizon=horizon
            )
            
            st.write(f"Kích thước tập Train: {X_train.shape[0]} sequences. Kích thước tập Test: {X_test.shape[0]} sequences.")
            
            # Training loop across seeds
            results_runs = []
            best_model = None
            best_sharpe = -1e9
            best_seed_val = None
            best_history_loss = []
            best_history_val_loss = []
            best_portfolio_returns = None
            best_pred_weights_test = None
            
            # Progress controls
            seed_progress_bar = st.progress(0)
            seed_status_text = st.empty()
            
            epoch_progress_bar = st.progress(0)
            epoch_status_text = st.empty()
            
            loss_plot_placeholder = st.empty()
            
            for s_idx, seed in enumerate(selected_seeds):
                seed_status_text.text(f"🔄 Đang huấn luyện với Seed: {seed} ({s_idx+1}/{len(selected_seeds)})...")
                seed_progress_bar.progress(s_idx / len(selected_seeds))
                
                # Set seeds for reproducibility
                os.environ["PYTHONHASHSEED"] = str(seed)
                random.seed(seed)
                np.random.seed(seed)
                tf.random.set_seed(seed)
                
                # Build model
                model = build_lstm_gru_model(
                    timesteps=X_train.shape[1],
                    n_features=X_train.shape[2],
                    n_assets=y_train_target.shape[1]
                )
                
                sharpe_loss_fn = make_sharpe_loss(rf_annual, trading_days, entropy_lambda)
                model.compile(
                    optimizer=Adam(learning_rate=0.0005),
                    loss=sharpe_loss_fn
                )
                
                callbacks = [
                    tf.keras.callbacks.EarlyStopping(
                        monitor="val_loss",
                        patience=10,
                        restore_best_weights=True
                    ),
                    tf.keras.callbacks.ReduceLROnPlateau(
                        monitor="val_loss",
                        factor=0.5,
                        patience=5,
                        min_lr=1e-5
                    ),
                    StreamlitCallback(
                        epoch_progress_bar,
                        epoch_status_text,
                        epochs,
                        seed,
                        seed_status_text
                    )
                ]
                
                # Train
                history = model.fit(
                    X_train,
                    y_train_target,
                    epochs=epochs,
                    batch_size=batch_size,
                    shuffle=False,
                    verbose=0,
                    validation_split=0.2,
                    callbacks=callbacks
                )
                
                # Predict & Evaluate on Test Set
                pred_weights_test = model.predict(X_test, verbose=0)
                
                weights_test_df = pd.DataFrame(
                    pred_weights_test,
                    index=test_seq_dates,
                    columns=top_symbols
                )
                y_test_df = pd.DataFrame(
                    y_test_target,
                    index=test_seq_dates,
                    columns=top_symbols
                )
                
                # Calculate return
                portfolio_returns = (weights_test_df * y_test_df).sum(axis=1)
                
                run_er = portfolio_returns.mean() * trading_days
                run_std = portfolio_returns.std() * np.sqrt(trading_days)
                run_sharpe = (run_er - rf_annual) / (run_std + 1e-12)
                
                results_runs.append({
                    "Seed": seed,
                    "Tỉ suất sinh lời năm": run_er,
                    "Độ lệch chuẩn năm": run_std,
                    "Tỉ số Sharpe": run_sharpe
                })
                
                if run_sharpe > best_sharpe:
                    best_sharpe = run_sharpe
                    best_seed_val = seed
                    best_model = model
                    best_history_loss = history.history["loss"]
                    best_history_val_loss = history.history["val_loss"]
                    best_portfolio_returns = portfolio_returns
                    best_pred_weights_test = pred_weights_test
                    st.session_state["y_test_df"] = y_test_df
            
            # Clean up widgets
            seed_progress_bar.empty()
            seed_status_text.empty()
            epoch_progress_bar.empty()
            epoch_status_text.empty()
            
            # Save results to session state
            results_runs_df = pd.DataFrame(results_runs).sort_values("Tỉ số Sharpe", ascending=False).reset_index(drop=True)
            st.session_state["results_runs_df"] = results_runs_df
            st.session_state["best_seed"] = best_seed_val
            st.session_state["best_model"] = best_model
            st.session_state["eval_dates"] = test_seq_dates
            st.session_state["eval_returns_lstm"] = best_portfolio_returns
            
            # Baseline portfolio (Equal Weighted)
            baseline_returns = y_test_df.mean(axis=1)
            st.session_state["eval_returns_baseline"] = baseline_returns
            
            # Compute weights average
            weights_avg = best_pred_weights_test.mean(axis=0)
            weights_df = pd.DataFrame({
                "Asset": top_symbols,
                "Weight": weights_avg
            }).sort_values("Weight", ascending=False).reset_index(drop=True)
            st.session_state["optimized_weights"] = weights_df
            
            # Test set baseline metrics
            b_er = baseline_returns.mean() * trading_days
            b_std = baseline_returns.std() * np.sqrt(trading_days)
            b_sharpe = (b_er - rf_annual) / (b_std + 1e-12)
            
            st.session_state["test_metrics"] = {
                "lstm": {"er": best_sharpe * best_portfolio_returns.std() * np.sqrt(trading_days) + rf_annual, "vol": best_portfolio_returns.std() * np.sqrt(trading_days), "sharpe": best_sharpe},
                "baseline": {"er": b_er, "vol": b_std, "sharpe": b_sharpe}
            }
            
            st.success(f"Huấn luyện hoàn tất! Seed tốt nhất: {best_seed_val} (Tỉ số Sharpe test = {best_sharpe:.4f})")
            
            # Display loss plot
            fig_loss = go.Figure()
            fig_loss.add_trace(go.Scatter(y=best_history_loss, mode="lines", name="Train Loss"))
            fig_loss.add_trace(go.Scatter(y=best_history_val_loss, mode="lines", name="Val Loss"))
            fig_loss.update_layout(
                title=f"Đường cong huấn luyện (Loss Curve) - Seed {best_seed_val}",
                xaxis_title="Epoch",
                yaxis_title="Loss",
                template="plotly_dark"
            )
            st.plotly_chart(fig_loss, use_container_width=True)

        # Show runs results table if exists
        if st.session_state["results_runs_df"] is not None:
            st.subheader("📊 Kết quả kiểm thử các Seed")
            st.dataframe(st.session_state["results_runs_df"].style.highlight_max(subset=["Tỉ số Sharpe"], color="#1e3a8a"), use_container_width=True)

# --- TAB 3: OPTIMIZATION RESULTS ---
with tab3:
    st.header("📊 Tối ưu hóa và Đánh giá danh mục đầu tư")
    
    if st.session_state["optimized_weights"] is None:
        st.warning("Vui lòng huấn luyện mô hình ở Tab 2 trước khi xem kết quả tối ưu hóa.")
    else:
        weights_df = st.session_state["optimized_weights"]
        metrics = st.session_state["test_metrics"]
        
        # Display KPIs
        st.subheader("💡 Chỉ số hiệu suất trên tập kiểm thử (Test Set)")
        col1, col2 = st.columns(2)
        
        # Optimized Metrics Card
        with col1:
            st.markdown(
                f"""
                <div class="metric-card">
                    <h3>Danh mục Tối ưu LSTM-GRU</h3>
                    <div class="metric-value">{metrics['lstm']['er']*100:.2f}%</div>
                    <div class="metric-label">Tỉ suất sinh lời mong đợi năm</div>
                    <div style="height: 15px;"></div>
                    <div class="metric-value">{metrics['lstm']['vol']*100:.2f}%</div>
                    <div class="metric-label">Độ lệch chuẩn năm (Volatility)</div>
                    <div style="height: 15px;"></div>
                    <div class="metric-value" style="color: #4ade80;">{metrics['lstm']['sharpe']:.4f}</div>
                    <div class="metric-label">Tỉ số Sharpe Ratio</div>
                </div>
                """,
                unsafe_allow_html=True
            )
            
        # Baseline Metrics Card
        with col2:
            st.markdown(
                f"""
                <div class="metric-card">
                    <h3>Danh mục Phân bổ đều (1/N)</h3>
                    <div class="metric-value">{metrics['baseline']['er']*100:.2f}%</div>
                    <div class="metric-label">Tỉ suất sinh lời mong đợi năm</div>
                    <div style="height: 15px;"></div>
                    <div class="metric-value">{metrics['baseline']['vol']*100:.2f}%</div>
                    <div class="metric-label">Độ lệch chuẩn năm (Volatility)</div>
                    <div style="height: 15px;"></div>
                    <div class="metric-value" style="color: #f87171;">{metrics['baseline']['sharpe']:.4f}</div>
                    <div class="metric-label">Tỉ số Sharpe Ratio</div>
                </div>
                """,
                unsafe_allow_html=True
            )
            
        # Visualizing Weights
        st.subheader("🍕 Phân bổ trọng số của Danh mục tối ưu")
        col_w1, col_w2 = st.columns([1, 1])
        
        with col_w1:
            fig_pie = px.pie(
                weights_df,
                values="Weight",
                names="Asset",
                title="Biểu đồ tròn phân bổ trọng số danh mục tối ưu",
                color_discrete_sequence=px.colors.sequential.Plotly3
            )
            st.plotly_chart(fig_pie, use_container_width=True)
            
        with col_w2:
            fig_bar_w = px.bar(
                weights_df,
                x="Asset",
                y="Weight",
                labels={"Asset": "Mã cổ phiếu", "Weight": "Trọng số"},
                title="Biểu đồ cột trọng số danh mục tối ưu",
                color="Weight",
                color_continuous_scale="Blues"
            )
            st.plotly_chart(fig_bar_w, use_container_width=True)
            
        # Show Weights Table
        st.dataframe(weights_df, use_container_width=True)
        
        # Download weights option
        csv_buffer = io.StringIO()
        weights_df.to_csv(csv_buffer, index=False)
        csv_data = csv_buffer.getvalue()
        
        st.download_button(
            label="📥 Tải trọng số danh mục tối ưu (.csv)",
            data=csv_data,
            file_name="optimized_portfolio_weights.csv",
            mime="text/csv"
        )
        
        # Cumulative returns comparison
        st.subheader("📈 Hiệu suất đầu tư tích lũy (Cumulative Returns) trên tập kiểm thử")
        
        dates = st.session_state["eval_dates"]
        lstm_ret = st.session_state["eval_returns_lstm"]
        baseline_ret = st.session_state["eval_returns_baseline"]
        
        # Calculate cumulative returns (1 + r).cumprod()
        lstm_cum = (1 + lstm_ret).cumprod() - 1
        baseline_cum = (1 + baseline_ret).cumprod() - 1
        
        # Cumulative returns dataframe
        cum_df = pd.DataFrame({
            "LSTM-GRU Portfolio": lstm_cum.values,
            "Baseline Equal-Weighted Portfolio": baseline_cum.values
        }, index=dates)
        
        # Individual stock returns for context
        y_test_df = st.session_state["y_test_df"]
        for col in y_test_df.columns:
            cum_df[f"{col} (Giá trị gốc)"] = (1 + y_test_df[col]).cumprod() - 1
            
        fig_cum = go.Figure()
        
        # Add optimized returns line
        fig_cum.add_trace(go.Scatter(
            x=cum_df.index,
            y=cum_df["LSTM-GRU Portfolio"] * 100,
            mode="lines",
            name="Danh mục Tối ưu LSTM-GRU",
            line=dict(color="#38bdf8", width=3)
        ))
        
        # Add equal weights line
        fig_cum.add_trace(go.Scatter(
            x=cum_df.index,
            y=cum_df["Baseline Equal-Weighted Portfolio"] * 100,
            mode="lines",
            name="Danh mục Phân bổ đều (1/N)",
            line=dict(color="#fb7185", width=2, dash="dash")
        ))
        
        # Add individual assets
        for col in y_test_df.columns:
            fig_cum.add_trace(go.Scatter(
                x=cum_df.index,
                y=cum_df[f"{col} (Giá trị gốc)"] * 100,
                mode="lines",
                name=col,
                opacity=0.3,
                line=dict(width=1)
            ))
            
        fig_cum.update_layout(
            title="Lợi nhuận tích lũy trên tập kiểm thử (Cumulative Return %)",
            xaxis_title="Thời gian",
            yaxis_title="Lợi nhuận tích lũy (%)",
            hovermode="x unified",
            template="plotly_dark"
        )
        
        st.plotly_chart(fig_cum, use_container_width=True)
