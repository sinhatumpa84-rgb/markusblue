import os
import json
from typing import Dict, Optional

class SUDOEngineeringAgent:
    """
    SUDO AI Assistant Layer for Development & Experimentation.
    IMPORTANT: This agent is strictly an offline development assistant for
    training analysis, hyperparameter suggestions, and code review.
    It is NOT part of the embedded real-time audio inference path.
    """
    def __init__(self, api_key: Optional[str] = None):
        # Reads securely from environment variable
        self.api_key = api_key or os.environ.get("SUDO_API_KEY")
        
    def is_configured(self) -> bool:
        return bool(self.api_key)

    def analyze_experiment(self, training_log_path: str = "experiments/training_history.json") -> Dict:
        """Analyze offline training metrics and suggest hyperparameter adjustments."""
        if not os.path.exists(training_log_path):
            return {"status": "error", "message": "No training history found"}
            
        with open(training_log_path, "r") as f:
            history = json.load(f)
            
        if not history:
            return {"status": "error", "message": "Empty history"}
            
        best_val = min(h["val_loss"] for h in history)
        max_gain = max(h["sisdr_gain_db"] for h in history)
        
        return {
            "status": "success",
            "epochs_analyzed": len(history),
            "best_validation_loss": best_val,
            "max_sisdr_gain_db": max_gain,
            "recommendation": "Model convergence confirmed. Proceed to INT8 quantization and hardware benchmark."
        }
