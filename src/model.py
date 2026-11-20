import ast
import json
from datetime import timedelta

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.metrics import brier_score_loss
from sklearn.model_selection import KFold


def calibration_metrics(y_true, y_prob):
    """Calculate weighted adaptive calibration error from quantile bins."""
    y_true_arr = np.asarray(y_true, dtype=float)
    y_prob_arr = np.asarray(y_prob, dtype=float)

    if len(y_true_arr) == 0:
        print("No valid samples for calibration.")
        return np.nan, np.array([]), np.array([]), 0

    bin_edges = np.unique(
        np.quantile(y_prob_arr, np.linspace(0.0, 1.0, 11))
    )

    if len(bin_edges) <= 1:
        fraction_of_positives = np.array([np.mean(y_true_arr)])
        mean_predicted_value = np.array([np.mean(y_prob_arr)])
        ace = np.abs(
            fraction_of_positives[0] - mean_predicted_value[0]
        )
        return ace, fraction_of_positives, mean_predicted_value, 1

    bin_indices = np.digitize(
        y_prob_arr, bin_edges[1:-1], right=True
    )
    fraction_of_positives = []
    mean_predicted_value = []
    bin_weights = []

    for bin_number in range(len(bin_edges) - 1):
        mask = bin_indices == bin_number
        count = np.sum(mask)
        if count > 0:
            fraction_of_positives.append(np.mean(y_true_arr[mask]))
            mean_predicted_value.append(np.mean(y_prob_arr[mask]))
            bin_weights.append(count / len(y_prob_arr))

    fraction_of_positives = np.asarray(fraction_of_positives)
    mean_predicted_value = np.asarray(mean_predicted_value)
    bin_weights = np.asarray(bin_weights)
    ace = np.sum(
        bin_weights
        * np.abs(fraction_of_positives - mean_predicted_value)
    )

    return (
        ace,
        fraction_of_positives,
        mean_predicted_value,
        len(fraction_of_positives),
    )


def log_likelihood(params, I_t_input, n_vis, n_oth, weights):
    """
        Calculate the log-likelihood of the model given the parameters and data.
    """
    epsilon, gamma = params
    I_t_input = np.asarray(I_t_input, dtype=float)
    n_vis = np.asarray(n_vis, dtype=float)
    n_oth = np.asarray(n_oth, dtype=float)
    weights = np.asarray(weights, dtype=float)

    P_t = 1.0 - np.power(1.0 - epsilon, n_vis) * np.power(1.0 - gamma, n_oth)
    P_t = np.clip(P_t, 1e-10, 1.0 - 1e-10)
    return -np.sum(
        weights
        * (I_t_input * np.log(P_t) + (1.0 - I_t_input) * np.log(1.0 - P_t))
    )

def risk_mle_adaptive_triangle_only_pos(
    df_behavior, min_pos_tests, max_window_size, Poisson_contacts=False
):
  """
    Perform maximum likelihood estimation using an adaptive triangular window to estimate per contact risk.
    """
  results = []

  df_behavior = df_behavior.copy()
  df_behavior["T0_date"] = pd.to_datetime(df_behavior["T0_date"])
  df_behavior = df_behavior.sort_values(by="T0_date")

  min_date = df_behavior["T0_date"].min()
  max_date = df_behavior["T0_date"].max()

  # Generate full date range, not just the ones with data
  full_date_range = pd.date_range(start=min_date, end=max_date)

  for date in full_date_range:  # , desc="Fitting adaptive MLE"):
    left_offset = 0
    right_offset = 0
    pos_tests = 0
    max_hit = False

    # Try expanding window until enough positive tests or limits are hit
    while pos_tests < min_pos_tests:
      start_date = date - timedelta(days=left_offset)
      end_date = date + timedelta(days=right_offset)

      current_window_size = (end_date - start_date).days + 1

      # Stop if trying to look outside the available date range
      if start_date < min_date or end_date > max_date:
        break

      if current_window_size > max_window_size:
        max_hit = True
        break

      df_window = df_behavior[
          (df_behavior["T0_date"] >= start_date)
          & (df_behavior["T0_date"] <= end_date)
      ].copy()

      pos_tests = len(df_window[df_window["positivetest"] == 1])

      left_offset += 1
      right_offset += 1

    # Skip if no data in window
    if "df_window" not in locals() or df_window.empty:
      continue

    # Recompute final window (undo one offset step)
    final_start = date - timedelta(days=left_offset - 1)
    final_end = date + timedelta(days=right_offset - 1)

    df_window = df_behavior[
        (df_behavior["T0_date"] >= final_start)
        & (df_behavior["T0_date"] <= final_end)
    ].copy()

    # If still no data (e.g. entire window is empty), skip
    if df_window.empty:
      continue

    df_window["day_offset"] = (df_window["T0_date"] - date).dt.days
    max_offset = max(
        abs(df_window["day_offset"].min()), abs(df_window["day_offset"].max())
    )
    window_size_days = 2 * max_offset + 1

    df_window["day_weight"] = (
        window_size_days + 1
    ) / 2 - df_window["day_offset"].abs()

    I_t = np.array(df_window["positivetest"])
    final_pos_tests = int((I_t == 1).sum())
    total_tests_in_window = int(len(I_t))

    if Poisson_contacts == False:
      n_vis = np.array(df_window["sum_visitors_t1_t11_interpolated"])
      n_oth = np.array(df_window["sum_others_t1_t11_interpolated"])
    else:
      n_vis = np.random.poisson(df_window["sum_visitors_t1_t11_interpolated"])
      n_oth = np.random.poisson(df_window["sum_others_t1_t11_interpolated"])
    weights = np.array(df_window["day_weight"])

    initial_guess = [0.001, 0.001]
    result = minimize(
        log_likelihood,
        initial_guess,
        args=(I_t, n_vis, n_oth, weights),
        bounds=((1e-10, 1 - 1e-10), (1e-10, 1 - 1e-10)),
        method="L-BFGS-B",
    )

    epsilon_hat, gamma_hat = result.x

    results.append({
        "T0_date": date,
        "estimated_epsilon": epsilon_hat,
        "estimated_gamma": gamma_hat,
        "window_size_days": window_size_days,
        "pos_tests_in_window": final_pos_tests,
        "total_tests_in_window": total_tests_in_window,
        "max_window_reached": max_hit,
        "full_window_used": (
            (final_start >= min_date)
            and (final_end <= max_date)
            and ((final_pos_tests >= min_pos_tests) or max_hit)
        ),
    })

  return pd.DataFrame(results).sort_values("T0_date").reset_index(drop=True)

def predict_postest(df_risk, df_behavior):
    """
    Predict the probability of infection for each test in df_behavior based on the estimated per contact risk values from df_risk.
    """
    # Initialize lists to store intermediate results
    predictions = []
    P_vis_pred = []
    P_oth_pred = []
    epsilon_lst = []
    gamma_lst = []
    vis_only_lst = []
    oth_only_lst = []

    # Iterate over all tests
    for _, row in tqdm(df_behavior.iterrows()):
        # Retrieve the estimated epsilon and gamma for the date of this test
        date = row["T0_date"]
        epsilon_gamma = df_risk[df_risk["T0_date"] == date]
        
        if epsilon_gamma.empty:
            predictions.append(np.nan)
            P_vis_pred.append(np.nan)
            P_oth_pred.append(np.nan)
            epsilon_lst.append(np.nan)
            gamma_lst.append(np.nan)
            vis_only_lst.append(np.nan)
            oth_only_lst.append(np.nan)
            continue
        
        epsilon = epsilon_gamma["estimated_epsilon"].values[0]
        gamma = epsilon_gamma["estimated_gamma"].values[0]
        
        # Calculate the probability of infection
        n_vis = row["sum_visitors_t1_t11_interpolated"]
        n_oth = row["sum_others_t1_t11_interpolated"]
        P_t = 1 - (1 - epsilon)**n_vis * (1 - gamma)**n_oth
        P_vis = 1 - (1 - epsilon)**n_vis
        P_oth = 1 - (1 - gamma)**n_oth
        
        # Calculate conditional probabilities
        vis_only = (1 - (1 - epsilon)**n_vis) * (1 - gamma)**n_oth
        oth_only = (1 - (1 - gamma)**n_oth) * (1 - epsilon)**n_vis
        
        # Append results
        predictions.append(P_t)
        P_vis_pred.append(P_vis)
        P_oth_pred.append(P_oth)
        epsilon_lst.append(epsilon)
        gamma_lst.append(gamma)
        vis_only_lst.append(vis_only)
        oth_only_lst.append(oth_only)
    
    df_pred = df_behavior.copy()

    # Add predictions and components
    df_pred["predicted_infections"] = predictions
    df_pred["P_vis_pred"] = P_vis_pred
    df_pred["P_oth_pred"] = P_oth_pred
    df_pred["epsilon"] = epsilon_lst
    df_pred["gamma"] = gamma_lst
    df_pred["vis_only"] = vis_only_lst
    df_pred["oth_only"] = oth_only_lst

    # Proportional splits
    df_pred["vis_prop"] = df_pred["P_vis_pred"] / (df_pred["P_oth_pred"] + df_pred["P_vis_pred"])
    df_pred["oth_prop"] = df_pred["P_oth_pred"] / (df_pred["P_oth_pred"] + df_pred["P_vis_pred"])
    df_pred["vis_ratio_split"] = df_pred["vis_prop"] * df_pred["predicted_infections"]
    df_pred["oth_ratio_split"] = df_pred["oth_prop"] * df_pred["predicted_infections"]

    # Compute overlap
    df_pred["overlap"] = df_pred["predicted_infections"] - df_pred["vis_only"] - df_pred["oth_only"]

    # Keep only rows for which the model was fully fitted (based on df_risk full_window_used flag)
    df_pred["T0_date"] = pd.to_datetime(df_pred["T0_date"])
    valid_dates = pd.to_datetime(df_risk[df_risk["full_window_used"] == True]["T0_date"]).dt.normalize()
    df_pred = df_pred[df_pred["T0_date"].dt.normalize().isin(valid_dates)].reset_index(drop=True)

    return df_pred

def cross_validation_adaptive_window_pos(df, min_pos_tests, max_window_size_days):
    
    kf = KFold(n_splits=10, shuffle=True, random_state=1809)
    df = df.copy()
    df["T0_date"] = pd.to_datetime(df["T0_date"])

    all_predictions = []
    all_actuals = []
    all_window_sizes = []
    max_window_hit_flags = []
    pos_tests_used = []

    fold = 1

    for train_idx, val_idx in kf.split(df):
        print(f"Processing Fold {fold}")
        train_df = df.iloc[train_idx].copy()
        val_df = df.iloc[val_idx].copy()
        train_results = []

        all_dates = train_df["T0_date"].sort_values().unique()
        min_date = all_dates.min()
        max_date = all_dates.max()

        # Generate full date range, not just the ones with data
        full_date_range = pd.date_range(start=min_date, end=max_date)
        
        for date in full_date_range:
            date = pd.Timestamp(date)
            left_offset = 0
            right_offset = 0
            total_tests = 0
            pos_tests = 0
            boundary_hit = False
            max_hit = False

            while pos_tests < min_pos_tests:
                start_date = date - timedelta(days=left_offset)
                end_date = date + timedelta(days=right_offset)

                current_window_size = (end_date - start_date).days + 1

                if start_date < min_date or end_date > max_date:
                    boundary_hit = True
                    break

                if current_window_size >= max_window_size_days:
                    max_hit = True
                    break

                window_data = train_df[
                    (train_df["T0_date"] >= start_date) & (train_df["T0_date"] <= end_date)
                ].copy()

                total_tests = len(window_data)
                pos_tests = len(window_data[window_data["positivetest"] == 1])
                left_offset += 1
                right_offset += 1

            if total_tests == 0:
                continue  # Skip if no data even after expansion

            current_window_size = (end_date - start_date).days + 1

            window_data["day_offset"] = (pd.to_datetime(window_data["T0_date"]) - pd.to_datetime(date)).dt.days
            max_offset = max(abs(window_data["day_offset"].min()), abs(window_data["day_offset"].max()))
            window_size = 2 * max_offset + 1
            window_data["day_weight"] = (window_size + 1) / 2 - window_data["day_offset"].abs()

            weights = np.array(window_data["day_weight"])
            I_t = np.array(window_data["positivetest"])
            n_vis = np.array(window_data["sum_visitors_t1_t11_interpolated"])
            n_oth = np.array(window_data["sum_others_t1_t11_interpolated"])

            initial_guess = [0.001, 0.001]

            result = minimize(
                log_likelihood,
                initial_guess,
                args=(I_t, n_vis, n_oth, weights),
                bounds=((1e-10, 1 - 1e-10), (1e-10, 1 - 1e-10))
            )

            epsilon_hat, gamma_hat = result.x
            train_results.append({
                "T0_date": date,
                "estimated_epsilon": epsilon_hat,
                "estimated_gamma": gamma_hat,
                "window_size": current_window_size,
                "full_window_used": not boundary_hit  # only False if dataset bounds were hit
            })

        train_estimates = pd.DataFrame(train_results)

        predictions = []
        actuals = []
        val_window_sizes = []
        val_max_flags = []
        val_pos_tests = []

        for _, row in val_df.iterrows():
            date = row["T0_date"]
            epsilon_gamma = train_estimates[
                (train_estimates["T0_date"] == date) & (train_estimates["full_window_used"] == True)
            ]

            if epsilon_gamma.empty:
                predictions.append(np.nan)
                actuals.append(np.nan)
                val_window_sizes.append(np.nan)
                val_max_flags.append(np.nan)
                val_pos_tests.append(np.nan)
                continue

            epsilon = epsilon_gamma["estimated_epsilon"].values[0]
            gamma = epsilon_gamma["estimated_gamma"].values[0]
            n_vis = row["sum_visitors_t1_t11_interpolated"]
            n_oth = row["sum_others_t1_t11_interpolated"]
            P_t = 1 - (1 - epsilon) ** n_vis * (1 - gamma) ** n_oth

            predictions.append(P_t)
            actuals.append(row["positivetest"])

            current_size = epsilon_gamma["window_size"].values[0]
            val_window_sizes.append(current_size)
            val_max_flags.append(current_size >= max_window_size_days)
            val_pos_tests.append(min_pos_tests)  # We stopped once this was met

        val_df["predicted_infections"] = predictions
        val_actuals = np.array(val_df["positivetest"])
        val_predictions = np.array(val_df["predicted_infections"])

        valid_mask = ~np.isnan(val_actuals) & ~np.isnan(val_predictions)
        val_actuals = val_actuals[valid_mask]
        val_predictions = val_predictions[valid_mask]
        val_window_sizes = np.array(val_window_sizes)[valid_mask]
        val_max_flags = np.array(val_max_flags)[valid_mask]
        val_pos_tests = np.array(val_pos_tests)[valid_mask]

        all_predictions.extend(val_predictions)
        all_actuals.extend(val_actuals)
        all_window_sizes.extend(val_window_sizes)
        max_window_hit_flags.extend(val_max_flags)
        pos_tests_used.extend(val_pos_tests)

        df.loc[val_df.index[valid_mask], 'predicted_infections'] = val_predictions

        fold += 1

    df_clean = df[df["predicted_infections"].notna()]
    y_true = df_clean['positivetest']
    y_prob = df_clean['predicted_infections']

    valid_mask = ~np.isnan(y_true) & ~np.isnan(y_prob)
    y_true = y_true[valid_mask]
    y_prob = y_prob[valid_mask]

    ace, fraction_of_positives, mean_predicted_value, actual_bins = (
        calibration_metrics(y_true, y_prob)
    )


    brier_score = brier_score_loss(y_true, y_prob)

    return (
        len(y_prob),
        np.array(all_actuals),
        np.array(all_predictions),
        ace,
        brier_score,
        fraction_of_positives,
        mean_predicted_value,
        actual_bins,
        np.array(all_window_sizes),
        np.array(max_window_hit_flags),
        np.array(pos_tests_used)
    )

def bootstrap_confidence_intervals_pred(df_behavior, min_pos_tests, max_window_size, n_bootstraps=100, alpha=0.05):
    """
    Return point estimates and bootstrap confidence intervals for epsilon and gamma per T0_date
    using Poisson bootstrapping and adaptive windowing.

    1. Calculate the base results (This also saves the needed window size per test date)
    2. Loop over all dates and collect all tests which fall within the adaptive window for that date
    3. Repeat "n_bootstrap" times:
    4.      Sample for each test in the window a number of visitors and other contacts based on Poisson distributions using the actual value as the parameter of said Poisson distribution
    5. Calculate and save the 95% CI for this date
    
    """
    np.random.seed(1809)
    
    base_results = risk_mle_adaptive_triangle_only_pos(df_behavior, min_pos_tests, max_window_size)
    ci_results = []
    ci_results_pred = []
    
    df_samples_epsilon = pd.DataFrame()
    df_samples_gamma = pd.DataFrame()
    df_samples_daily_predicted = pd.DataFrame()
    df_samples_vis_infections = pd.DataFrame()
    df_samples_oth_infections = pd.DataFrame()

    for sample_nr in tqdm(range(n_bootstraps), desc = 'Bootstrap iteration'):
        #print('Bootstrap sample', sample_nr)
        df_risk_HELP = risk_mle_adaptive_triangle_only_pos(df_behavior, min_pos_tests, max_window_size, Poisson_contacts = True)
        df_samples_epsilon[str(sample_nr)] = df_risk_HELP['estimated_epsilon']
        df_samples_gamma[str(sample_nr)] = df_risk_HELP['estimated_gamma']
        df_pred_HELP = predict_postest(df_risk_HELP, df_behavior)
        
        daily_predicted_infections = df_pred_HELP.groupby("T0_date")["predicted_infections"].sum()
        daily_vis_infections = df_pred_HELP.groupby("T0_date")["vis_ratio_split"].sum()
        daily_oth_infections = df_pred_HELP.groupby("T0_date")["oth_ratio_split"].sum()
        daily_actual_infections = df_pred_HELP.groupby("T0_date")["positivetest"].sum()
        
        window_size = 21
        predicted_ma = daily_predicted_infections.rolling(window=window_size, center=True).mean()
        vis_ma = daily_vis_infections.rolling(window=window_size, center=True).mean()
        oth_ma = daily_oth_infections.rolling(window=window_size, center=True).mean()
        actual_ma = daily_actual_infections.rolling(window=window_size, center=True).mean()

        predicted_ma = predicted_ma.dropna()
        vis_ma = vis_ma.dropna()
        oth_ma = oth_ma.dropna()
        actual_ma = actual_ma.dropna()
        
        df_samples_daily_predicted[str(sample_nr)] = predicted_ma#daily_predicted_infections
        df_samples_vis_infections[str(sample_nr)] = vis_ma#daily_vis_infections
        df_samples_oth_infections[str(sample_nr)] = oth_ma#daily_oth_infections
        
        

    for _, row in tqdm(base_results.iterrows(), total=len(base_results), desc="Bootstrapping CIs"):
        date = row["T0_date"]
        # Compute confidence interval percentiles
        lower = alpha / 2
        upper = 1 - alpha / 2


        ci_results.append({
                "T0_date": date,
                "estimated_epsilon": row["estimated_epsilon"],
                "estimated_gamma": row["estimated_gamma"],
                "window_size_days": row["window_size_days"],
                "max_window_reached": row["max_window_reached"],
                "full_window_used": row["full_window_used"]
        })
        
    df_ci_results = pd.DataFrame(ci_results).sort_values("T0_date").reset_index(drop=True)
    new_columns = {'epsilon_lower': df_samples_epsilon.quantile(lower,axis = 1), 
                   'epsilon_upper': df_samples_epsilon.quantile(upper, axis = 1), 
                   'gamma_lower': df_samples_gamma.quantile(lower, axis = 1), 
                   'gamma_upper': df_samples_gamma.quantile(upper, axis = 1),
                  }
    print(new_columns)
    df_ci_results = df_ci_results.assign(**new_columns)
    
    pred_columns = {'predicted_lower': df_samples_daily_predicted.quantile(lower, axis = 1),
                    'predicted_upper': df_samples_daily_predicted.quantile(upper, axis = 1),
                    'vis_lower': df_samples_vis_infections.quantile(lower, axis = 1),
                    'vis_upper': df_samples_vis_infections.quantile(upper, axis = 1),
                    'oth_lower': df_samples_oth_infections.quantile(lower, axis = 1),
                    'oth_upper': df_samples_oth_infections.quantile(upper, axis = 1),  
                    'actual': daily_actual_infections,
                    'actual_ma': actual_ma,
                   }
    df_ci_results_predict = pd.DataFrame(pred_columns).sort_values("T0_date").reset_index(drop = False)
                    
        #return pd.DataFrame(ci_results).sort_values("T0_date").reset_index(drop=True)
    return df_ci_results, df_ci_results_predict

def log_likelihood_simple_weighted(params, I_t_input, n_vis, n_oth, weights):
    epsilon = params[0]  # single parameter

    log_likelihood_value = 0

    for t in range(len(I_t_input)):
        I_t = I_t_input[t]
        n_vis_t = n_vis[t]
        n_oth_t = n_oth[t]
        w_t = weights[t]

        # Combined contact effect
        P_t = 1 - (1 - epsilon)**(n_vis_t + n_oth_t)
        P_t = np.clip(P_t, 1e-10, 1 - 1e-10)

        log_likelihood_value += w_t * (I_t * np.log(P_t) + (1 - I_t) * np.log(1 - P_t))

    return -log_likelihood_value

def cross_validation_adaptive_window_simple(df, min_pos_tests, max_window_size_days):
    
    kf = KFold(n_splits=10, shuffle=True, random_state=1809)
    df = df.copy()
    df["T0_date"] = pd.to_datetime(df["T0_date"])

    all_predictions = []
    all_actuals = []
    all_window_sizes = []
    max_window_hit_flags = []
    pos_tests_used = []

    fold = 1

    for train_idx, val_idx in kf.split(df):
        print(f"Processing Fold {fold}")
        train_df = df.iloc[train_idx].copy()
        val_df = df.iloc[val_idx].copy()
        train_results = []

        all_dates = train_df["T0_date"].sort_values().unique()
        min_date = all_dates.min()
        max_date = all_dates.max()

        full_date_range = pd.date_range(start=min_date, end=max_date)

        for date in full_date_range:
            date = pd.Timestamp(date)
            left_offset = 0
            right_offset = 0
            total_tests = 0
            pos_tests = 0
            boundary_hit = False
            max_hit = False

            while pos_tests < min_pos_tests:
                start_date = date - timedelta(days=left_offset)
                end_date = date + timedelta(days=right_offset)

                current_window_size = (end_date - start_date).days + 1

                if start_date < min_date or end_date > max_date:
                    boundary_hit = True
                    break

                if current_window_size >= max_window_size_days:
                    max_hit = True
                    break

                window_data = train_df[
                    (train_df["T0_date"] >= start_date) & (train_df["T0_date"] <= end_date)
                ].copy()

                total_tests = len(window_data)
                pos_tests = len(window_data[window_data["positivetest"] == 1])
                left_offset += 1
                right_offset += 1

            if total_tests == 0:
                continue

            current_window_size = (end_date - start_date).days + 1
            
            window_data["day_offset"] = (pd.to_datetime(window_data["T0_date"]) - pd.to_datetime(date)).dt.days
            max_offset = max(abs(window_data["day_offset"].min()), abs(window_data["day_offset"].max()))
            window_size = 2 * max_offset + 1
            window_data["day_weight"] = (window_size + 1) / 2 - window_data["day_offset"].abs()

            weights = np.array(window_data["day_weight"])
            I_t = np.array(window_data["positivetest"])
            n_vis = np.array(window_data["sum_visitors_t1_t11_interpolated"])
            n_oth = np.array(window_data["sum_others_t1_t11_interpolated"])

            initial_guess = [0.001]
            
            result = minimize(
                log_likelihood_simple_weighted,
                initial_guess,
                args=(I_t, n_vis, n_oth, weights),
                bounds=[(1e-10, 1 - 1e-10)],
            )

            epsilon_hat = result.x[0]

            train_results.append({
                "T0_date": date,
                "estimated_epsilon": epsilon_hat,
                "window_size": current_window_size,
                "full_window_used": not boundary_hit
            })

        train_estimates = pd.DataFrame(train_results)

        predictions = []
        actuals = []
        val_window_sizes = []
        val_max_flags = []
        val_pos_tests = []

        for _, row in val_df.iterrows():
            date = row["T0_date"]
            epsilon_gamma = train_estimates[
                (train_estimates["T0_date"] == date) & (train_estimates["full_window_used"] == True)
            ]

            if epsilon_gamma.empty:
                predictions.append(np.nan)
                actuals.append(np.nan)
                val_window_sizes.append(np.nan)
                val_max_flags.append(np.nan)
                val_pos_tests.append(np.nan)
                continue

            epsilon = epsilon_gamma["estimated_epsilon"].values[0]
            n_vis = row["sum_visitors_t1_t11_interpolated"]
            n_oth = row["sum_others_t1_t11_interpolated"]
            P_t = 1 - (1 - epsilon) ** (n_vis + n_oth)

            predictions.append(P_t)
            actuals.append(row["positivetest"])
            current_size = epsilon_gamma["window_size"].values[0]
            val_window_sizes.append(current_size)
            val_max_flags.append(current_size >= max_window_size_days)
            val_pos_tests.append(min_pos_tests)

        val_df["predicted_infections"] = predictions

        val_actuals = np.array(val_df["positivetest"])
        val_predictions = np.array(val_df["predicted_infections"])

        valid_mask = ~np.isnan(val_actuals) & ~np.isnan(val_predictions)
        val_actuals = val_actuals[valid_mask]
        val_predictions = val_predictions[valid_mask]
        val_window_sizes = np.array(val_window_sizes)[valid_mask]
        val_max_flags = np.array(val_max_flags)[valid_mask]
        val_pos_tests = np.array(val_pos_tests)[valid_mask]

        all_predictions.extend(val_predictions)
        all_actuals.extend(val_actuals)
        all_window_sizes.extend(val_window_sizes)
        max_window_hit_flags.extend(val_max_flags)
        pos_tests_used.extend(val_pos_tests)

        df.loc[val_df.index[valid_mask], 'predicted_infections'] = val_predictions
        fold += 1

    df_clean = df[df["predicted_infections"].notna()]
    y_true = df_clean['positivetest']
    y_prob = df_clean['predicted_infections']

    valid_mask = ~np.isnan(y_true) & ~np.isnan(y_prob)
    y_true = y_true[valid_mask]
    y_prob = y_prob[valid_mask]

    ace, fraction_of_positives, mean_predicted_value, actual_bins = (
        calibration_metrics(y_true, y_prob)
    )

    brier_score = brier_score_loss(y_true, y_prob)

    return (
        len(y_prob),
        np.array(all_actuals),
        np.array(all_predictions),
        ace,
        brier_score,
        fraction_of_positives,
        mean_predicted_value,
        actual_bins,
        np.array(all_window_sizes),
        np.array(max_window_hit_flags),
        np.array(pos_tests_used)
    )
