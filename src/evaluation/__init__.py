"""Evaluation metrics, visualizations, and report generation."""
from .metrics import compute_tactical_metrics, evaluate_model_on_split
from .visualizer import TacticalVisualizer
from .report_generator import generate_html_evaluation_report

__all__ = [
    "compute_tactical_metrics",
    "evaluate_model_on_split",
    "TacticalVisualizer",
    "generate_html_evaluation_report"
]
