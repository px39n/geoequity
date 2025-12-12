"""
geoequity: Spatial Equity Assessment for Machine Learning Models
================================================================

Tools to diagnose and visualize spatial performance disparities
in geospatial machine learning models.

Main Components:
- TwoStageModel: Two-stage accuracy prediction (GAM + SVM)
- eval_baseline_comparison: Compare baseline methods
- split_test_train: Geospatial cross-validation strategies
- calculate_density: Data density calculation
"""

from .two_stage import TwoStageModel
from .two_stage.visualization import plot_predicted_accuracy_map, predict_at_locations
from .data import split_test_train, calculate_density
from .evaluation import eval_baseline_comparison
from .models import SpatialRegressor, InterpolationModel
from .visualization import plot_accuracy_map, plot_accuracy_comparison

__version__ = '0.1.0'
__author__ = 'Zhehao Liang'

__all__ = [
    'TwoStageModel',
    'eval_baseline_comparison',
    'SpatialRegressor',
    'InterpolationModel',
    'split_test_train', 
    'calculate_density',
    'plot_predicted_accuracy_map',
    'predict_at_locations',
    'plot_accuracy_map',
    'plot_accuracy_comparison',
]

