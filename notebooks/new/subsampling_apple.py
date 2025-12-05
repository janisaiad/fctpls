# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.16.1
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Mid Price Subsampling
# 
# we subsample mid prices at different time intervals and visualize the results

# %%
import os
import polars as pl
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from datetime import timedelta
from tqdm import tqdm

# %%
folder_path = "/home/janis/HFT/HFT/data/DB_MBP_10/"

# %%
def curate_mid_price(stock, file, folder_path=folder_path):
    """
    we curate mid price data by filtering publishers, trading hours, and calculating mid prices
    based on logic from curating.py
    """
    # we read the parquet file
    df = pl.read_parquet(f"{folder_path}/{stock}/{file}")
    
    # we check publisher_id distribution and filter
    num_entries_by_publisher = df.group_by("publisher_id").len().sort("len", descending=True)
    if len(num_entries_by_publisher) > 1:
        # we prefer publisher 41 if multiple publishers exist
        df = df.filter(pl.col("publisher_id") == 41)
    else:
        # we use publisher 2 if only one publisher exists
        df = df.filter(pl.col("publisher_id") == 2)
    
    # we filter by trading hours based on stock
    if stock == "GOOGL":
        # we filter GOOGL: 13:00-20:00
        df = df.filter(pl.col("ts_event").dt.hour() >= 13)
        df = df.filter(pl.col("ts_event").dt.hour() <= 20)
    else:
        # we filter other stocks: 9:30-16:00
        df = df.filter(
            (
                (pl.col("ts_event").dt.hour() == 9) & (pl.col("ts_event").dt.minute() >= 30) |
                (pl.col("ts_event").dt.hour() > 9) & (pl.col("ts_event").dt.hour() < 16) |
                (pl.col("ts_event").dt.hour() == 16) & (pl.col("ts_event").dt.minute() == 0)
            )
        )
    
    # we calculate mid price
    mid_price = (df["ask_px_00"] + df["bid_px_00"]) / 2
    
    # we manage nans, infs, and nulls with preceding value filling
    # we replace inf with null first, then fill all nulls and nans
    shifted = mid_price.shift(1)
    mid_price = mid_price.replace([float('inf'), float('-inf')], None)  # we replace inf with null
    mid_price = mid_price.fill_null(shifted)  # we fill nulls (including those from inf)
    mid_price = mid_price.fill_nan(shifted)  # we fill nans
    
    # we add mid_price column to dataframe
    df = df.with_columns(mid_price=mid_price)
    
    return df

# we use only INTC for testing
STOCKS = ["INTC"]

# time intervals in microseconds: 5min, 1min, 5sec, 1sec, 500ms, 100ms, 50ms, 10ms, 5ms, 1ms, 500us, 100us, 50us, 10us, 5us
INTERVALS_US = [
    5 * 60 * 1_000_000,      # 5 minutes
    1 * 60 * 1_000_000,      # 1 minute
    5 * 1_000_000,           # 5 seconds
    1 * 1_000_000,           # 1 second
    500_000,                 # 500 milliseconds
    100_000,                 # 100 milliseconds
    50_000,                  # 50 milliseconds
    10_000,                  # 10 milliseconds
    5_000,                   # 5 milliseconds
    1_000,                   # 1 millisecond
    500,                     # 500 microseconds
    100,                     # 100 microseconds
    50,                      # 50 microseconds
    10,                      # 10 microseconds
    5,                       # 5 microseconds
]

# interval names for display
INTERVAL_NAMES = [
    "5min", "1min", "5sec", "1sec", "500ms", "100ms", "50ms", "10ms", 
    "5ms", "1ms", "500us", "100us", "50us", "10us", "5us"
]

# %%
def extract_mid_price_parquet(stock, file, folder_path=folder_path, output_path=None):
    """
    we extract mid price data and save it as a parquet file with just ts_event and mid_price
    """
    # we curate the data
    df = curate_mid_price(stock, file, folder_path)
    
    # we keep only ts_event and mid_price
    mid_price_df = df.select(["ts_event", "mid_price"])
    
    # we save to parquet if output_path is provided
    if output_path:
        mid_price_df.write_parquet(output_path)
        print(f"saved mid price data to {output_path} ({len(mid_price_df)} points)")
    
    return mid_price_df

# %%
def subsample_mid_prices(df, interval_us):
    """
    we subsample mid prices at a given interval in microseconds
    returns a dataframe with ts_event and mid_price columns
    we use group_by_dynamic with proper duration format
    """
    if len(df) == 0:
        return pl.DataFrame({"ts_event": pl.Series([], dtype=pl.Datetime), "mid_price": pl.Series([], dtype=pl.Float64)})
    
    # we ensure the dataframe is sorted by timestamp
    df = df.sort("ts_event")
    
    # we convert interval_us to a duration string for group_by_dynamic
    # polars supports: "ns", "us", "ms", "s", "m", "h", "d"
    if interval_us >= 1_000_000_000:
        # seconds or more
        duration_sec = interval_us / 1_000_000
        if duration_sec >= 60:
            duration_min = duration_sec / 60
            duration = f"{int(duration_min)}m"
        else:
            duration = f"{int(duration_sec)}s"
    elif interval_us >= 1_000_000:
        # milliseconds
        duration_ms = interval_us / 1_000
        duration = f"{int(duration_ms)}ms"
    elif interval_us >= 1_000:
        # microseconds
        duration = f"{int(interval_us)}us"
    else:
        # nanoseconds
        duration = f"{int(interval_us * 1000)}ns"
    
    try:
        # we use group_by_dynamic to resample at the specified interval
        # we take the first value in each interval
        result = df.group_by_dynamic(
            "ts_event",
            every=duration,
            closed="left"
        ).agg(
            pl.col("mid_price").first().alias("mid_price")
        )
    except Exception as e:
        # if group_by_dynamic fails, we fall back to a simpler method
        print(f"warning: group_by_dynamic failed with duration {duration}, using fallback method: {e}")
        # we create a simple index-based subsampling
        step = max(1, len(df) // (df["ts_event"].max() - df["ts_event"].min()).total_seconds() * 1_000_000 / interval_us)
        result = df[::int(step)].select(["ts_event", "mid_price"])
    
    return result

# %%
def plot_subsampling_comparison(stock, file, intervals_to_plot=None):
    """
    we plot subsampled mid prices at different intervals for verification
    intervals_to_plot: list of interval indices to plot (default: first 5 intervals)
    """
    # we extract mid price data first
    df = extract_mid_price_parquet(stock, file, folder_path)
    
    if len(df) == 0:
        print(f"no data for {stock}/{file}")
        return
    
    # we select intervals to plot (default: first 5 for readability)
    if intervals_to_plot is None:
        intervals_to_plot = list(range(min(5, len(INTERVALS_US))))
    
    # we filter intervals_to_plot to only include valid indices
    max_available = len(INTERVALS_US)
    intervals_to_plot = [i for i in intervals_to_plot if i < max_available]
    
    if len(intervals_to_plot) == 0:
        print(f"no valid intervals to plot (only {max_available} intervals available)")
        return
    
    # we create subplots
    n_plots = len(intervals_to_plot)
    fig = make_subplots(
        rows=n_plots, 
        cols=1,
        subplot_titles=[f"{INTERVAL_NAMES[i]} ({INTERVALS_US[i]}us)" for i in intervals_to_plot],
        vertical_spacing=0.05,
        shared_xaxes=True
    )
    
    # we get original data for reference (subsampled at 1ms for visibility)
    original_subsampled = subsample_mid_prices(df, 1_000)  # 1ms for reference
    
    for idx, interval_idx in enumerate(intervals_to_plot):
        interval_us = INTERVALS_US[interval_idx]
        interval_name = INTERVAL_NAMES[interval_idx]
        
        # we subsample at this interval
        subsampled_df = subsample_mid_prices(df, interval_us)
        
        if len(subsampled_df) == 0:
            continue
        
        # we convert timestamps to numeric for plotting
        timestamps = subsampled_df["ts_event"].to_list()
        mid_prices = subsampled_df["mid_price"].to_list()
        
        # we plot the subsampled data
        fig.add_trace(
            go.Scatter(
                x=timestamps,
                y=mid_prices,
                mode='lines',
                name=f"{interval_name}",
                line=dict(width=1),
                showlegend=False
            ),
            row=idx+1, col=1
        )
        
        # we add original data as reference (only for first subplot)
        if idx == 0 and len(original_subsampled) > 0:
            orig_timestamps = original_subsampled["ts_event"].to_list()
            orig_prices = original_subsampled["mid_price"].to_list()
            fig.add_trace(
                go.Scatter(
                    x=orig_timestamps,
                    y=orig_prices,
                    mode='lines',
                    name='original (1ms)',
                    line=dict(width=0.5, color='red', dash='dash'),
                    opacity=0.5,
                    showlegend=True
                ),
                row=idx+1, col=1
            )
    
    # we update layout
    fig.update_layout(
        title=f"Mid Price Subsampling Comparison - {stock} - {file}",
        height=300 * n_plots,
        showlegend=True
    )
    
    fig.update_xaxes(title_text="Time", row=n_plots, col=1)
    fig.update_yaxes(title_text="Mid Price", col=1)
    
    fig.show()

# %%
def plot_all_intervals_for_stock(stock, file):
    """
    we plot all intervals for a single stock/day combination
    """
    # we extract mid price data first
    df = extract_mid_price_parquet(stock, file, folder_path)
    
    if len(df) == 0:
        print(f"no data for {stock}/{file}")
        return
    
    # we create a single plot with all intervals overlaid (with different opacities)
    fig = go.Figure()
    
    # we plot intervals from coarsest to finest
    colors = ['red', 'orange', 'yellow', 'green', 'blue', 'indigo', 'violet']
    
    for idx, (interval_us, interval_name) in enumerate(zip(INTERVALS_US, INTERVAL_NAMES)):
        # we subsample at this interval
        subsampled_df = subsample_mid_prices(df, interval_us)
        
        if len(subsampled_df) == 0:
            continue
        
        timestamps = subsampled_df["ts_event"].to_list()
        mid_prices = subsampled_df["mid_price"].to_list()
        
        # we calculate opacity based on interval (finer intervals more transparent)
        opacity = max(0.1, 1.0 - (idx / len(INTERVALS_US)) * 0.8)
        color_idx = idx % len(colors)
        
        fig.add_trace(
            go.Scatter(
                x=timestamps,
                y=mid_prices,
                mode='lines',
                name=f"{interval_name} ({len(subsampled_df)} points)",
                line=dict(width=1, color=colors[color_idx]),
                opacity=opacity
            )
        )
    
    fig.update_layout(
        title=f"All Intervals Overlay - {stock} - {file}",
        xaxis_title="Time",
        yaxis_title="Mid Price",
        height=600,
        hovermode='x unified'
    )
    
    fig.show()

# %%
def verify_subsampling_for_sample():
    """
    we verify subsampling works by extracting mid price, then subsampling and plotting
    """
    stock = "INTC"
    stock_path = os.path.join(folder_path, stock)
    
    if not os.path.exists(stock_path):
        print(f"path does not exist: {stock_path}")
        return
    
    parquet_files = [f for f in os.listdir(stock_path) if f.endswith('.parquet') and not f.endswith('_curated.parquet')]
    
    if not parquet_files:
        print(f"no parquet files found for {stock}")
        return
    
    # we use the first file
    file = parquet_files[0]
    print(f"verifying subsampling for {stock}/{file}")
    
    # step 1: extract mid price to parquet
    print("step 1: extracting mid price data...")
    mid_price_df = extract_mid_price_parquet(stock, file, folder_path)
    print(f"extracted {len(mid_price_df)} mid price points")
    
    # step 2: test subsampling with a few intervals
    print("step 2: testing subsampling...")
    test_intervals = [0, 1, 2, 3, 4]  # first 5 intervals
    test_intervals = [i for i in test_intervals if i < len(INTERVALS_US)]
    
    for idx in test_intervals:
        interval_us = INTERVALS_US[idx]
        interval_name = INTERVAL_NAMES[idx]
        subsampled = subsample_mid_prices(mid_price_df, interval_us)
        print(f"  {interval_name}: {len(subsampled)} points")
    
    # step 3: plot comparison
    print("step 3: plotting comparison...")
    plot_subsampling_comparison(stock, file, intervals_to_plot=test_intervals)
    
    # step 4: plot all intervals overlaid
    print("step 4: plotting all intervals overlaid...")
    plot_all_intervals_for_stock(stock, file)

# %%
def test_all_intervals():
    """
    we test subsampling for INTC with all time intervals
    """
    stock = "INTC"
    stock_path = os.path.join(folder_path, stock)
    
    if not os.path.exists(stock_path):
        print(f"path does not exist: {stock_path}")
        return
    
    parquet_files = [f for f in os.listdir(stock_path) if f.endswith('.parquet') and not f.endswith('_curated.parquet')]
    
    if not parquet_files:
        print(f"no parquet files found for {stock}")
        return
    
    # we use the first file
    file = parquet_files[0]
    print(f"testing all intervals for {stock}/{file}")
    print(f"total intervals to test: {len(INTERVALS_US)}")
    
    # we plot all intervals overlaid
    plot_all_intervals_for_stock(stock, file)
    
    # we also plot comparison of all intervals
    all_intervals = list(range(len(INTERVALS_US)))
    plot_subsampling_comparison(stock, file, intervals_to_plot=all_intervals)

# %%
# we run verification
verify_subsampling_for_sample()

# %%
def process_and_plot_all_stocks():
    """
    we process all stocks and create plots for verification
    """
    for stock in tqdm(STOCKS, desc="processing stocks"):
        stock_path = os.path.join(folder_path, stock)
        
        if not os.path.exists(stock_path):
            print(f"warning: path does not exist for {stock}: {stock_path}")
            continue
        
        parquet_files = [f for f in os.listdir(stock_path) if f.endswith('.parquet')]
        
        if not parquet_files:
            print(f"warning: no parquet files found for {stock}")
            continue
        
        # we process first file for each stock as a sample
        file = parquet_files[0]
        print(f"\nprocessing {stock}/{file}")
        
        # we plot comparison of first 5 intervals
        plot_subsampling_comparison(stock, file, intervals_to_plot=list(range(5)))

# %%
# we test all intervals for INTC
test_all_intervals()