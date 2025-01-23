import json
from datetime import datetime
from pathlib import Path

def log_classification_results(song_path: str, category: str, model_probabilities: list, final_probability: float, log_dir: str = "classification_logs"):
    """
    Log classification results for a given category to a JSON file.
    
    Args:
        song_path: Path to the analyzed song
        category: Music category being classified
        model_probabilities: List of probabilities from individual models
        final_probability: Final probability from logistic regression
        log_dir: Directory to store log files
    """
    # Create logs directory if it doesn't exist
    Path(log_dir).mkdir(exist_ok=True)
    
    log_file = Path(log_dir) / f"{category}_results.json"
    
    # Prepare log entry
    entry = {
        "timestamp": datetime.now().isoformat(),
        "song_path": song_path,
        "song_name": Path(song_path).stem,
        "category": category,
        "model_predictions": {
            "knn": model_probabilities[0],
            "lr": model_probabilities[1],
            "nb": model_probabilities[2],
            "rf": model_probabilities[3],
            "svm": model_probabilities[4]
        },
        "final_probability": float(final_probability)
    }
    
    # Load existing logs or create new list
    if log_file.exists():
        with open(log_file) as f:
            logs = json.load(f)
    else:
        logs = []
    
    # Append new entry and save
    logs.append(entry)
    with open(log_file, 'w') as f:
        json.dump(logs, f, indent=2)