import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.cm import ScalarMappable
from matplotlib.colors import ListedColormap


def risk_plot(df_risk):
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
    plt.title('Estimated Epsilon and Gamma over Time', size=16)
    plt.xlabel('Date', size=12)
    plt.ylabel('Estimated Values', size=12)
    plt.legend(loc="upper left", fontsize=11)
    plt.grid(True)

    plt.show()

def plot_cause_split(df_pred, window_size=35, save_path=None):
    """
    Plot the predicted positive tests per day, split by cause (visitors vs others), along with the actual positive tests (moving average).
    """
    df_pred = df_pred.copy()
    df_pred["T0_date"] = pd.to_datetime(df_pred["T0_date"]).dt.normalize()

    # Fill missing dates so each rolling window covers consecutive calendar days.
    daily = df_pred.groupby("T0_date")[
        ["predicted_infections", "vis_ratio_split", "oth_ratio_split", "positivetest"]
    ].sum()
    calendar = pd.date_range(daily.index.min(), daily.index.max(), freq="D")
    daily = daily.reindex(calendar, fill_value=0)

    # Apply a moving average (centered) to the data
    predicted_ma = daily["predicted_infections"].rolling(window=window_size, center=True).mean()
    vis_ma = daily["vis_ratio_split"].rolling(window=window_size, center=True).mean()
    oth_ma = daily["oth_ratio_split"].rolling(window=window_size, center=True).mean()
    actual_ma = daily["positivetest"].rolling(window=window_size, center=True).mean()

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
    plt.ylim(0, 1.5)

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

def plot_cause_split_proportions(df_pred, window_size=35, save_path=None):
    """
    Plot the predicted positive tests per day, split by cause (visitors vs others), as proportions of total predicted infections.
    """
    df_pred.loc[:, "T0_date"] = pd.to_datetime(df_pred["T0_date"])

    # Compute the total predicted infections per source per day
    daily_vis_infections = df_pred.groupby("T0_date")["vis_ratio_split"].sum()
    daily_oth_infections = df_pred.groupby("T0_date")["oth_ratio_split"].sum()

    # Apply a moving average (centered) to the data
    vis_ma = daily_vis_infections.rolling(window=window_size, center=True).mean()
    oth_ma = daily_oth_infections.rolling(window=window_size, center=True).mean()

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

def smoothed_predictions_plot(df, window_size):
    
    df["T0_date"] = pd.to_datetime(df["T0_date"])

    # Step 1: Aggregate daily counts
    daily = df.groupby("T0_date")[["predicted_infections", "positivetest"]].sum()

    # Step 3: Apply moving average (centered)
    smoothed = daily.rolling(window=window_size, center=True).mean()

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
    
    plt.title("21-Day Moving Average of Actual vs Predicted Positive Tests Per Day", size=16)
    plt.xlim(smoothed["predicted_infections"].index.min(), smoothed["predicted_infections"].index.max())
    plt.ylim(0, 1.5)
    plt.xlabel("Date", size=12)
    plt.ylabel("Number of Positive Tests", size=12)
    plt.xticks(rotation=45, ha='right')
    plt.legend(loc="upper left", fontsize=11)
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def smoothed_predictions_plot_CI(df, df_ci, df_ci_predict, window_size, save_path=None):
    # Only if df_ci_predict contains dates
    df["T0_date"] = pd.to_datetime(df["T0_date"])
    df_ci_predict['T0_date'] = pd.to_datetime(df_ci_predict['T0_date'])

    # Step 1: Aggregate daily counts
    daily = df.groupby("T0_date")[[
        "predicted_infections",
        "positivetest",
        "vis_ratio_split",
        "oth_ratio_split",
    ]].sum()

    # Step 3: Apply moving average (centered)
    smoothed = daily.rolling(window=window_size, center=True).mean()

    # Step 4: Drop NaNs caused by the smoothing window
    smoothed = smoothed.dropna()

    # Match bootstrap intervals to the exact dates used by the point estimates.
    df_plot = df_ci_predict.set_index("T0_date").reindex(smoothed.index)
    df_plot = df_plot.dropna(subset=["vis_lower", "vis_upper", "oth_lower", "oth_upper"])

    # Step 5: Plot
    plt.figure(figsize=(12, 6))
    plt.plot(smoothed.index, smoothed["vis_ratio_split"],
             label=f"Predicted positive tests caused by visitors", color='tab:blue')
    plt.fill_between(df_plot.index, df_plot["vis_lower"], df_plot["vis_upper"],
                 color="tab:blue", alpha=0.2, label=r"_95% CI for positive tests caused by visitors")
    plt.plot(smoothed.index, smoothed["oth_ratio_split"],
             label=f"Predicted positive tests caused by others", color='tab:orange')
    plt.fill_between(df_plot.index, df_plot["oth_lower"], df_plot["oth_upper"],
                 color="tab:orange", alpha=0.2, label=r"_95% CI for positive tests caused by others")
    #plt.plot(df_plot['shifted_T0_date'],df_plot['actual_ma'])
    
    #Quick fix for paper: calculate correlation coefficient:
    print('Correlation is', smoothed['positivetest'].corr(smoothed["predicted_infections"]))
    
    ax = plt.gca()
    ax.xaxis.set_major_locator(mdates.MonthLocator())
    
    plt.title(f"{window_size}-day moving average of predicted positive tests per day by cause", size=16)
    plt.xlim(smoothed["predicted_infections"].index.min(), smoothed["predicted_infections"].index.max())
    plt.ylim(0, 1.5)
    plt.xlabel("Date", size=12)
    plt.ylabel("Number of positive tests", size=12)
    plt.xticks(rotation=45, ha='right')
    plt.legend(loc="upper left", fontsize=11)
    plt.grid(True)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.show()
