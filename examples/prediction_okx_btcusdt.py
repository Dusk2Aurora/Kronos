import sys
from pathlib import Path
# 将项目根目录加入 Python 模块搜索路径，确保能够 import model
sys.path.insert(0, str(Path(__file__).parent.parent))

import argparse
import numpy as np        # 新增：用于生成不同颜色
import ccxt
import pandas as pd
import matplotlib.pyplot as plt
from model import Kronos, KronosTokenizer, KronosPredictor

# --- 添加中文字体支持，避免缺失字形警告 ---
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
# -------------------------------------------

# 支持的 timeframes
TIMEFRAMES = ["5m", "15m", "1h", "4h", "1d"]

def parse_time_delta(timeframe: str) -> pd.Timedelta:
    unit = timeframe[-1]
    value = int(timeframe[:-1])
    if unit == "m":
        return pd.Timedelta(minutes=value)
    if unit == "h":
        return pd.Timedelta(hours=value)
    if unit == "d":
        return pd.Timedelta(days=value)
    raise ValueError(f"Unsupported timeframe: {timeframe}")

def fetch_okx_ohlcv(symbol: str, timeframe: str, limit: int, proxy=None):
    """用 ccxt 从 OKX 拉取 OHLCV 数据"""
    exchange_config = {"enableRateLimit": True}
    if proxy:
        exchange_config["proxies"] = {"http": proxy, "https": proxy}
    exchange = ccxt.okx(exchange_config)
    data = exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
    df = pd.DataFrame(data, columns=["timestamp", "open", "high", "low", "close", "volume"])
    df["timestamps"] = pd.to_datetime(df["timestamp"], unit="ms")
    return df[["timestamps", "open", "high", "low", "close", "volume"]]

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="基于Kronos模型的BTC/USDT OHLCV预测")
    parser.add_argument("--timeframe", choices=TIMEFRAMES, default="5m", help="预测时间段")
    parser.add_argument("--lookback",   type=int, default=400,   help="回溯步数")
    parser.add_argument("--pred-len",   type=int, default=80,   help="预测步数")
    parser.add_argument("--num-samples",type=int, default=4,     help="采样次数")
    parser.add_argument("--proxy", default=None, help="Optional HTTP/HTTPS proxy URL")
    parser.add_argument("--device", default=None, help="Inference device; defaults to automatic selection")
    args = parser.parse_args()

    lookback   = args.lookback
    pred_len   = args.pred_len
    timeframe  = args.timeframe
    num_samples= args.num_samples

    # 计算时间间隔
    time_delta = parse_time_delta(timeframe)

    # 生成多条预测
    colors = plt.cm.tab10(np.linspace(0, 1, num_samples))

    # 预加载模型与 tokenizer
    tokenizer = KronosTokenizer.from_pretrained("NeoQuasar/Kronos-Tokenizer-base")
    model     = Kronos.from_pretrained("NeoQuasar/Kronos-small")
    predictor = KronosPredictor(model, tokenizer, device=args.device, max_context=512)

    # 拉取数据，根据 --timeframe 参数动态设置
    limit = lookback + pred_len
    df = fetch_okx_ohlcv(
        symbol="BTC/USDT",
        timeframe=timeframe,   # 使用传入的时间段
        limit=limit,
        proxy=args.proxy
    )
    x_df  = df.iloc[:lookback][["open", "high", "low", "close", "volume"]]
    x_ts  = df.iloc[:lookback]["timestamps"].reset_index(drop=True)
    y_start = x_ts.iloc[-1] + time_delta
    y_ts    = pd.date_range(start=y_start, periods=pred_len, freq=time_delta)

    T     = 0.9
    top_p = 0.9

    # 多次采样预测
    all_preds = []
    for i in range(num_samples):
        pred_df = predictor.predict(
            df=x_df.reset_index(drop=True),
            x_timestamp=x_ts,
            y_timestamp=pd.Series(y_ts, name="timestamps"),
            pred_len=pred_len,
            T=T,
            top_p=top_p,
            sample_count=1,
            verbose=True,     # 开启进度条显示
        )
        all_preds.append(pred_df)

    # 绘图
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(df["timestamps"][:lookback], df["close"][:lookback],
            label="历史 Close", color="blue")
    for idx, pred in enumerate(all_preds):
        ax.plot(pred.index, pred["close"],
                label=f"预测 #{idx+1}",
                color=colors[idx], linestyle="--", alpha=0.8)
    ax.set_xlabel("时间")
    ax.set_ylabel("价格")
    ax.legend()
    ax.grid(True)
    plt.tight_layout()
    plt.show()
