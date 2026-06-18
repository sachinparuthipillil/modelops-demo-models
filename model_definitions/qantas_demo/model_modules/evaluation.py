from sklearn.metrics import confusion_matrix, roc_curve, auc
#import matplotlib.pyplot as plt
from teradataml import td_sklearn as osml
from teradataml import(
    DataFrame,
    copy_to_sql,
    get_context,
    get_connection,
    ScaleTransform,
    ConvertTo,
    ClassificationEvaluator,
    ROC,
    td_lightgbm,
    INTEGER
)
from aoa import (
    record_evaluation_stats,
    save_plot,
    aoa_create_context,
    ModelContext
)

import joblib
import json

import numpy as np
import pandas as pd
import os
import warnings
warnings.filterwarnings('ignore')
warnings.simplefilter(action='ignore', category=DeprecationWarning)
warnings.simplefilter(action='ignore', category=UserWarning)
warnings.simplefilter(action='ignore', category=FutureWarning)


def evaluate(context: ModelContext, **kwargs):

    aoa_create_context()
    

    print("Empty Evaluation - All done!")
