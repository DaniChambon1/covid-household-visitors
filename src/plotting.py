import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.cm import ScalarMappable
from matplotlib.colors import ListedColormap


def risk_plot(df_risk, show_title=True):
    """
    Plot the estimated per contact risk over time.
    """
    # Only plot the dates where the full window was used
    df_risk_clean = df_risk[df_risk['full_window_used'] == 1]

    # Shift T0 dates by 11 days earlier
    df_risk_clean['shifted_T0_date'] = df_risk_clean['T0_date'] - pd.to_timedelta(11, unit='D')

    plt.figure(figsize=(12, 6))

    # Plot epsilon values
    plt.plot(df_risk_clean['shifted_T0_date'], df_risk_clean['estimated_epsilon'], label='Estimated Epsilon (shifted)', c="tab:blue")

    # Overlay gamma values in the same plot
    plt.plot(df_risk_clean['shifted_T0_date'], df_risk_clean['estimated_gamma'], label='Estimated Gamma (shifted)', c="tab:orange")

    plt.xlim(df_risk_clean['shifted_T0_date'].min(), df_risk_clean['shifted_T0_date'].max())
    plt.ylim(-0.006, 0.03)
    ax = plt.gca()
    ax.xaxis.set_major_locator(mdates.MonthLocator())
    
    plt.xticks(rotation=45, ha='right')
    if show_title:
        plt.title('Estimated Epsilon and Gamma over Time', size=16)
    plt.xlabel('Date', size=12)
    plt.ylabel('Estimated Values', size=12)
    plt.legend(loc="upper left", fontsize=11)
    plt.grid(True)

    plt.show()

def plot_cause_split(df_pred, window_size=35, save_path=None, show_title=True):
    """
    Plot the predicted positive tests per day, split by cause (visitors vs others), along with the actual positive tests (moving average).
    """
    daily = (
        df_pred.assign(
            T0_date=pd.to_datetime(df_pred["T0_date"]).dt.normalize()
        )
        .set_index("T0_date")[
            ["predicted_infections", "vis_ratio_split", "oth_ratio_split", "positivetest"]
        ]
        .resample("D")
        .sum()
    )

    # Apply a moving average (centered) to the data
    predicted_ma = daily["predicted_infections"].rolling(
        window=window_size, center=True, min_periods=window_size
    ).mean()
    vis_ma = daily["vis_ratio_split"].rolling(
        window=window_size, center=True, min_periods=window_size
    ).mean()
    oth_ma = daily["oth_ratio_split"].rolling(
        window=window_size, center=True, min_periods=window_size
    ).mean()
    actual_ma = daily["positivetest"].rolling(
        window=window_size, center=True, min_periods=window_size
    ).mean()

    # Remove NaN values from the moving averages
    vis_ma_clean = vis_ma.dropna()
    oth_ma_clean = oth_ma.dropna()
    actual_ma_clean = actual_ma.dropna()

    # Create the plot
    plt.figure(figsize=(12, 6))

    # Plot the stacked area for infections caused by visitors
    plt.fill_between(
        vis_ma_clean.index,
        0,
        vis_ma_clean.values,
        label="Predicted positive tests caused by visitors",
        color="lightblue",
        alpha=0.7,
    )

    # Plot the stacked area for infections caused by other contacts
    plt.fill_between(
        oth_ma_clean.index,
        vis_ma_clean.values,
        vis_ma_clean.values + oth_ma_clean.values,
        label="Predicted positive tests caused by others",
        color="orange",
        alpha=0.7,
    )

    # Plot the actual infections (moving average) as a dashed line
    plt.plot(
        actual_ma_clean.index,
        actual_ma_clean.values,
        label=f"Actual positive tests ({window_size}-day MA)",
        color="black",
        linewidth=1.5,
        linestyle="--",
    )
   
    plt.xlim(actual_ma_clean.index.min(), actual_ma_clean.index.max())
    plt.ylim(0, 1.6)

    if show_title:
        plt.title(f"""{window_size}-day moving average of predicted positive tests per day by cause""", fontsize=16)
    ax = plt.gca()
    ax.xaxis.set_major_locator(mdates.MonthLocator())
    
    plt.xticks(rotation=45, ha='right')
    
    plt.xlabel("Date", size=12)
    plt.ylabel("Number of positive tests", size=12)
    plt.legend(loc="upper left", fontsize=11)
    plt.grid(True, linestyle="--", alpha=0.5)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=200, bbox_inches='tight')
        
    plt.show()

def plot_cause_split_proportions(df_pred, window_size=35, save_path=None, show_title=True):
    """
    Plot the predicted positive tests per day, split by cause (visitors vs others), as proportions of total predicted infections.
    """
    daily = (
        df_pred.assign(
            T0_date=pd.to_datetime(df_pred["T0_date"]).dt.normalize()
        )
        .set_index("T0_date")[["vis_ratio_split", "oth_ratio_split"]]
        .resample("D")
        .sum()
    )

    # Apply a moving average (centered) to the data
    vis_ma = daily["vis_ratio_split"].rolling(
        window=window_size, center=True, min_periods=window_size
    ).mean()
    oth_ma = daily["oth_ratio_split"].rolling(
        window=window_size, center=True, min_periods=window_size
    ).mean()

    # Compute the total infections and percentage contributions
    total_ma = vis_ma + oth_ma
    vis_percent = (vis_ma / total_ma) * 100
    oth_percent = (oth_ma / total_ma) * 100

    # Remove NaN values caused by the rolling average
    vis_percent_clean = vis_percent.dropna()
    oth_percent_clean = oth_percent.dropna()

    # Create the plot
    plt.figure(figsize=(12, 6))

    # Plot the stacked area chart normalized to 100%
    plt.fill_between(
        vis_percent_clean.index,
        0,
        vis_percent_clean.values,
        label="Visitors",
        color="lightblue",
        alpha=0.7,
    )

    plt.fill_between(
        oth_percent_clean.index,
        vis_percent_clean.values,
        vis_percent_clean.values + oth_percent_clean.values,
        label="Others",
        color="orange",
        alpha=0.7,
    )

    plt.xlim(vis_percent_clean.index.min(), vis_percent_clean.index.max())
    plt.ylim(0, 100)

    if show_title:
        plt.title(f"""Contribution of {window_size}-day moving average of predicted positive tests per day by cause""", size=16)

    ax = plt.gca()
    ax.xaxis.set_major_locator(mdates.MonthLocator())
    
    plt.xticks(rotation=45, ha='right')
    plt.xlabel("Date", size=12)
    plt.ylabel("Contribution (%)", size=12)
    plt.legend(loc="upper left", fontsize=11)
    plt.grid(True, linestyle="--", alpha=0.5)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.show()

def smoothed_predictions_plot(df, window_size, show_title=True):

    daily = (
        df.assign(T0_date=pd.to_datetime(df["T0_date"]).dt.normalize())
        .set_index("T0_date")[["predicted_infections", "positivetest"]]
        .resample("D")
        .sum()
    )

    # Step 3: Apply moving average (centered)
    smoothed = daily.rolling(
        window=window_size, center=True, min_periods=window_size
    ).mean()

    # Step 4: Drop NaNs caused by the smoothing window
    smoothed = smoothed.dropna()

    # Step 5: Plot
    plt.figure(figsize=(12, 6))
    plt.plot(smoothed.index, smoothed["positivetest"],
             label=f"Actual Positive Tests ({window_size}-day MA)", color='tab:blue')
    plt.plot(smoothed.index, smoothed["predicted_infections"],
             label=f"Predicted Positive Tests ({window_size}-day MA)", color='tab:orange')

    ax = plt.gca()
    ax.xaxis.set_major_locator(mdates.MonthLocator())
    
    if show_title:
        plt.title("21-Day Moving Average of Actual vs Predicted Positive Tests Per Day", size=16)
    plt.xlim(smoothed["predicted_infections"].index.min(), smoothed["predicted_infections"].index.max())
    plt.ylim(0, 1.6)
    plt.xlabel("Date", size=12)
    plt.ylabel("Number of Positive Tests", size=12)
    plt.xticks(rotation=45, ha='right')
    plt.legend(loc="upper left", fontsize=11)
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def smoothed_predictions_plot_CI(df_ci_predict, window_size, save_path=None, show_title=True):
    # Plot the base estimates bundled with the bootstrap intervals.
    df_plot = df_ci_predict.copy()
    df_plot["T0_date"] = pd.to_datetime(df_plot["T0_date"]).dt.normalize()
    df_plot = df_plot.set_index("T0_date").sort_index()
    has_complete_ci = df_plot[
        ["vis_lower", "vis_upper", "oth_lower", "oth_upper"]
    ].notna().all(axis=1)
    df_plot = df_plot.loc[has_complete_ci]

    plt.figure(figsize=(12, 6))
    plt.plot(df_plot.index, df_plot["vis_ma"],
             label=f"Predicted positive tests caused by visitors", color='tab:blue')
    plt.fill_between(df_plot.index, df_plot["vis_lower"], df_plot["vis_upper"],
                 color="tab:blue", alpha=0.2, label=r"_95% CI for positive tests caused by visitors")
    plt.plot(df_plot.index, df_plot["oth_ma"],
             label=f"Predicted positive tests caused by others", color='tab:orange')
    plt.fill_between(df_plot.index, df_plot["oth_lower"], df_plot["oth_upper"],
                 color="tab:orange", alpha=0.2, label=r"_95% CI for positive tests caused by others")

    correlation_data = df_plot[["actual_ma", "predicted_ma"]].dropna()
    print(
        'Correlation is',
        correlation_data["actual_ma"].corr(correlation_data["predicted_ma"]),
    )
    
    ax = plt.gca()
    ax.xaxis.set_major_locator(mdates.MonthLocator())
    
    if show_title:
        plt.title(f"{window_size}-day moving average of predicted positive tests per day by cause", size=16)
    plt.xlim(df_plot.index.min(), df_plot.index.max())
    plt.ylim(0, 1.6)
    plt.xlabel("Date", size=12)
    plt.ylabel("Number of positive tests", size=12)
    plt.xticks(rotation=45, ha='right')
    plt.legend(loc="upper left", fontsize=11)
    plt.grid(True)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.show()
