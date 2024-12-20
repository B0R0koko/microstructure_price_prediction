import json
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import pandas as pd
from catboost import Pool
from scipy.signal import savgol_filter


# Load global best results from disk if they exist
def load_best_global(BEST_GLOBAL_FILE: str) -> str:
    try:
        with open(BEST_GLOBAL_FILE, "r") as f:
            return json.load(f)
    except FileNotFoundError:
        return {"study_name": None, "best_value": float("inf"), "best_params": None, "features": None}
    
# Save global best results to disk
def save_best_global(best_global: str, BEST_GLOBAL_FILE: str) -> None:
    with open(BEST_GLOBAL_FILE, "w") as f:
        json.dump(best_global, f, indent=4)

def prepare_data_for_ranker(X_train: pd.DataFrame, 
                            y_train: pd.Series, 
                            group_id: str,
                            cat_features: list[str]) -> Pool:

    sorted_train_data = X_train.sort_values(by=group_id)
    sorted_labels = y_train.loc[sorted_train_data.index]
    sorted_group_id = sorted_train_data[group_id].astype('category').cat.codes
    sorted_train_data = sorted_train_data.drop(columns=[group_id])

    train_pool = Pool(
        data=sorted_train_data,
        label=sorted_labels,
        group_id=sorted_group_id,
        cat_features=cat_features
    )
    return train_pool

# Function to plot feature importance
def plot_feature_importance(model, feature_names: list[str]) -> None:

    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]
    plt.figure(figsize=(10, 6))
    plt.title("Feature Importances")
    plt.bar(range(len(importances)), importances[indices], align="center")
    plt.xticks(range(len(importances)), [feature_names[i] for i in indices], rotation=45, ha="right")
    plt.tight_layout()
    plt.show()


def permutation_feature_importance(model, X_test, y_test, metrics):
    """
    Evaluates feature importance by shuffling one feature at a time and measuring the impact on model performance.
    
    Parameters:
    - model: Trained model to evaluate.
    - X_test (pl.DataFrame): Test features.
    - y_test (pl.Series): True labels.
    - metrics (list[function]): List of metric functions to evaluate model performance.

    Returns:
    - results (dict): A dictionary where keys are metric names, and values are DataFrames
                      containing features and their corresponding performance scores.
    """
    if isinstance(X_test, pl.DataFrame):
        X_test = pd.DataFrame(X_test, columns=X_test.columns)
        y_test = pd.Series(y_test)
    
    results = {}

    # Iterate over metrics
    for metric in metrics:
        metric_name = metric.__name__
        metric_results = []

        baseline_score = metric(y_test, model.predict(X_test))

        # Iterate over each feature in X_test
        for feature in X_test.columns:
            X_test_shuffled = X_test.copy()

            X_test_shuffled[feature] = np.random.permutation(X_test_shuffled[feature])
            shuffled_score = metric(y_test, model.predict(X_test_shuffled))

            metric_results.append({
                "feature": feature,
                "baseline_score": baseline_score,
                "shuffled_score": shuffled_score,
                "score_difference": baseline_score - shuffled_score
            })

        results[metric_name] = pl.DataFrame(metric_results)

    return results


def plot_feature_premutation_importance(results):
    """
    Plots feature importance based on score differences for each metric.

    Parameters:
    - results (dict): A dictionary where keys are metric names, and values are Polars DataFrames
                      containing features and their corresponding performance scores.
    """
    for metric_name, df in results.items():
        df_sorted = df.sort("score_difference", descending=True)

        # Plot the feature importances
        plt.figure(figsize=(10, 6))
        plt.barh(df_sorted["feature"], df_sorted["score_difference"], color="skyblue")
        plt.xlabel("Score Difference")
        plt.ylabel("Feature")
        plt.title(f"Feature Importance ({metric_name})")
        plt.gca().invert_yaxis()
        plt.tight_layout()
        plt.show()

def permutation_feature_importance_ranker(
    ranker_model, X_test, y_test, group_id, cat_features: list[str], metric="NDCG:type=Base"
):
    """
    Evaluates feature importance for a CatBoostRanker by shuffling one feature at a time
    and measuring the impact on ranking performance (e.g., NDCG).

    Parameters:
    - ranker_model: Trained CatBoostRanker model.
    - X_test (pd.DataFrame): Test feature matrix.
    - y_test (pd.Series): Test labels (returns).
    - group_id (str): Column name representing group structure in the dataset.
    - cat_features (list[str]): List of categorical feature names.
    - metric (str): Ranking metric to evaluate feature importance. Default is "NDCG:type=Base".

    Returns:
    - results (pd.DataFrame): DataFrame containing features and their corresponding performance scores.
    """
    import pandas as pd
    import numpy as np
    from catboost import Pool
    results = {}

    # Sort data by group_id to ensure alignment
    sorted_train_data = X_test.sort_values(by=group_id)
    sorted_labels = y_test.loc[sorted_train_data.index]
    sorted_group_id = sorted_train_data[group_id].astype('category').cat.codes
    sorted_train_data = sorted_train_data.drop(columns=[group_id])

    # Create test pool
    test_pool = Pool(
        data=sorted_train_data,
        label=sorted_labels,
        group_id=sorted_group_id,
        cat_features=cat_features
    )

    # Store baseline metric
    baseline_metrics = ranker_model.eval_metrics(
        data=test_pool,
        metrics=[metric],
        ntree_start=0,
        ntree_end=ranker_model.tree_count_,
        eval_period=1
    )
    baseline_score = np.max(baseline_metrics[metric])  # Best NDCG score

    # Identify features to shuffle
    features_for_shuffle = [x for x in X_test.columns if x not in cat_features and x != group_id]

    feature_importances = []

    # Iterate over each feature
    for feature_name in features_for_shuffle:
        X_test_shuffled = sorted_train_data.copy()

        # Shuffle the feature
        X_test_shuffled[feature_name] = np.random.permutation(X_test_shuffled[feature_name])

        # Create a new test pool with the shuffled feature
        shuffled_test_pool = Pool(
            data=X_test_shuffled,
            label=sorted_labels,
            group_id=sorted_group_id,
            cat_features=cat_features
        )

        # Evaluate metrics after shuffling
        shuffled_metrics = ranker_model.eval_metrics(
            data=shuffled_test_pool,
            metrics=[metric],
            ntree_start=0,
            ntree_end=ranker_model.tree_count_,
            eval_period=1
        )
        shuffled_score = np.max(shuffled_metrics[metric])  # Best NDCG score after shuffling

        # Calculate performance difference
        score_difference = baseline_score - shuffled_score

        feature_importances.append({
            "feature": feature_name,
            "baseline_score": baseline_score,
            "shuffled_score": shuffled_score,
            "score_difference": score_difference
        })

    # Convert results to DataFrame
    results[metric] = pl.DataFrame(feature_importances)

    return results

        
def plot_predictions(y_test, y_pred):
    # Apply smoothing (Savitzky-Golay filter)
    window_length = 51  # Choose an odd number; increase for more smoothing
    poly_order = 3      # Polynomial order for the smoothing
    
    y_test_smooth = savgol_filter(y_test, window_length=window_length, polyorder=poly_order)
    y_pred_smooth = savgol_filter(y_pred, window_length=window_length, polyorder=poly_order)
    
    # Plot smoothed predictions vs true values
    plt.figure(figsize=(12, 6))
    plt.plot(y_test_smooth, label='Smoothed True Values', color='blue', alpha=0.8)
    plt.plot(y_pred_smooth, label='Smoothed Predicted Values', color='red', alpha=0.8, linestyle='--')
    
    plt.xlabel('Index')
    plt.ylabel('Values')
    plt.title('Smoothed True vs Predicted Values')
    plt.legend()
    plt.grid(True)
    plt.show()