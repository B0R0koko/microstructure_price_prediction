import json
import matplotlib.pyplot as plt
import numpy as np

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