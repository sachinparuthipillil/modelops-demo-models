#from teradataml import td_sklearn as osml
import numpy as np
import lightgbm as lgb
import pickle
from sksurv.ensemble import RandomSurvivalForest

from teradataml import (
    copy_to_sql,
    DataFrame,
    ScaleTransform,
    td_lightgbm,
    INTEGER,
    Antiselect,
 
)
from teradataml import *
from aoa import (
    record_scoring_stats,
    aoa_create_context,
    ModelContext,
    DatasetInfo
)
import pandas as pd

import json
import warnings
warnings.filterwarnings('ignore')
warnings.simplefilter(action='ignore', category=DeprecationWarning)
warnings.simplefilter(action='ignore', category=UserWarning)
warnings.simplefilter(action='ignore', category=FutureWarning)

def predict_points(model, X: pd.DataFrame) -> np.ndarray:
    """Predict points using trained LightGBM model."""
    preds = model.predict(X
                          , num_iteration=model.best_iteration
                          #, predict_disable_shape_check=True
                         )
    
    return np.maximum(0, preds)
    
def predict_median_tenure(model, X: pd.DataFrame, fallback_time: float = 108.0) -> np.ndarray:
    """
    Predict median survival (tenure) in months using the RSF model.
    Returns the time when survival probability <= 0.5.
    If survival never drops below 0.5, returns `fallback_time`.
    """
    chf_funcs = model.predict_cumulative_hazard_function(X)

    median_survival_times = []
    for fn in chf_funcs:
        times = fn.x
        chf = fn.y
        surv_probs = np.exp(-chf)  # S(t) = exp(-H(t))
        below_half = np.where(surv_probs <= 0.5)[0]
        if len(below_half) == 0:
            median_survival_times.append(fallback_time)
        else:
            median_survival_times.append(times[below_half[0]])

    return np.array(median_survival_times)

def batch_predict(model, X, predict_fn, batch_size=50000, model_type="lgb"):
    """
    Predict in batches to handle large datasets efficiently.

    Parameters
    ----------
    model : object
        Trained LightGBM or RSF model.
    X : pd.DataFrame
        Feature matrix.
    predict_fn : callable
        Function to generate predictions for a batch (e.g., predict_points or predict_median_tenure).
    batch_size : int, default=50_000
        Number of rows per batch.
    model_type : str, {"lgb", "rsf"}
        Used to format progress output.
    """
    n = len(X)
    preds = np.zeros(n)
    num_batches = int(np.ceil(n / batch_size))

    print(f"🔄 Predicting {model_type.upper()} model in {num_batches} batches of {batch_size:,} rows each...")

    for i in range(num_batches):
        start, end = i * batch_size, min((i + 1) * batch_size, n)
        X_batch = X.iloc[start:end]
        preds[start:end] = predict_fn(model, X_batch)
        print(f"  ✅ Batch {i+1}/{num_batches} complete ({end:,}/{n:,} rows)")

    print(f"✅ {model_type.upper()} predictions finished.\n")
    return preds



def score(context: ModelContext, **kwargs):
    
    aoa_create_context()
    
    # Extract feature names, target name, and entity key from the context
    feature_names = context.dataset_info.feature_names
    #print(feature_names)
    target_name = context.dataset_info.target_names[0]
    #print("target_name "+target_name)
    entity_key = context.dataset_info.entity_key
    #print("entity_key "+entity_key)
    # Load the test dataset
    test_df = DataFrame.from_query(context.dataset_info.sql)
    
    ## add this step 
    
    copy_to_sql(
        df=test_df,
        schema_name=context.dataset_info.predictions_database,
        table_name='test_df',
        index=False,
        if_exists="replace"
    )
    
    
    # Get features from the model
    print("loading model")
    
    cur = get_context().raw_connection().cursor()
    cur.execute(f"SELECT file_content FROM qa_model_table")
    row = cur.fetchone()
  
    print(row[0][:100])
    content = row[0]
    
    print("writing model to the file")
    with open("temp_file.txt", "w", encoding="utf-8") as f:
        f.write(content)
    print("converting model to LGB")
    lgb_model=lgb.Booster(model_file='temp_file.txt')

    #Light GB features
    feature_names_lgb = lgb_model.feature_name()
    print(feature_names_lgb)
     
    
    df_lgb = test_df.select(feature_names_lgb+[entity_key]).to_pandas(index_column = entity_key, all_rows = True)

    df_lgb["points_score"] = batch_predict(lgb_model, df_lgb, predict_points, batch_size=500_000, model_type="lgb")

    predictions_pdf = pd.DataFrame(df_lgb, columns=[target_name])
    #predictions_pdf[entity_key] = test_df.get([entity_key]).to_pandas().reset_index().values.flatten()
    
    ## add job_id column so we know which execution this is from if appended to predictions table
    predictions_pdf["job_id"] = context.job_id
    predictions_pdf["json_report"] = ""
   # print(predictions_pdf.head())
    
   # predictions_pdf = predictions_pdf[["job_id", entity_key, target_name, "json_report"]]

    print(predictions_pdf.head())
    print(context.dataset_info.predictions_database)
    print(context.dataset_info.predictions_table)
    
    copy_to_sql(
        df=predictions_pdf.reset_index(),
        schema_name=context.dataset_info.predictions_database,
        table_name=context.dataset_info.predictions_table,
        index=False,
        if_exists="replace"
    )
        
    print("Saved predictions in Teradata")
  
