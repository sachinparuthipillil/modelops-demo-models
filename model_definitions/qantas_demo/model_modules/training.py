from teradataml import (
    DataFrame,
    ScaleFit,
    ScaleTransform,
)
#from teradataml import td_sklearn as osml
#from teradataml import td_lightgbm

from aoa import (
    record_training_stats,
    aoa_create_context,
    ModelContext
)

#import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import json

from collections import Counter
import warnings
warnings.filterwarnings('ignore')
warnings.simplefilter(action='ignore', category=DeprecationWarning)
warnings.simplefilter(action='ignore', category=UserWarning)
warnings.simplefilter(action='ignore', category=FutureWarning)
    
def train(context: ModelContext, **kwargs):
    aoa_create_context()
    
    
    print("Empty Training - All done!")
