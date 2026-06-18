from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from teradataml import (
    DataFrame,
    copy_to_sql,
    XGBoostPredict,
    ConvertTo,
    ClassificationEvaluator,
    ROC
)
from tmo import (
    record_evaluation_stats,
    tmo_create_context,
    ModelContext
)
from collections import Counter

import matplotlib.pyplot as plt
import json
import pandas as pd
import numpy as np
import os

def traverse_tree(tree, feature_counter):
    if 'split_' in tree and 'attr_' in tree['split_']:
        feature_counter[tree['split_']['attr_']] += 1
    if 'leftChild_' in tree:
        traverse_tree(tree['leftChild_'], feature_counter)
    if 'rightChild_' in tree:
        traverse_tree(tree['rightChild_'], feature_counter)


def compute_feature_importance(trees_json):
    feature_counter = Counter()
    for tree_json in trees_json:
        tree = json.loads(tree_json)
        traverse_tree(tree, feature_counter)
    total_splits = sum(feature_counter.values())
    feature_importance = {
        feature: count / total_splits for feature, count in feature_counter.items()}
    return feature_importance


def plot_feature_importance(fi, img_filename):
    feat_importances = pd.Series(fi)
    feat_importances.nlargest(10).plot(
        kind='barh').set_title('Feature Importance')
    fig = plt.gcf()
    fig.savefig(img_filename, dpi=400)
    plt.clf()

def compute_metrics(actual, predicted, label):
    mae  = mean_absolute_error(actual, predicted)
    rmse = np.sqrt(mean_squared_error(actual, predicted))
    mape = np.mean(np.abs((actual - predicted) / actual.clip(lower=0.1))) * 100
    r2   = r2_score(actual, predicted)
    return {"model": label, "MAE": f"{mae:.2f}", "RMSE": f"{rmse:.2f}",
            "MAPE": f"{mape:.1f}", "Rsquare": f"{r2:.4f}"}

def evaluate(context: ModelContext, **kwargs):

    tmo_create_context()

    print(f"Loading model from table model_{context.model_version}")
    model = DataFrame(f"model_{context.model_version}")

    target_name = context.dataset_info.target_names[0]
    entity_key = context.dataset_info.entity_key

    test_df = DataFrame.from_query(context.dataset_info.sql)

    print("Evaluating...")

    predictions = XGBoostPredict(
    newdata=test_df,
    object=model,
    model_type='Regression',
    id_column=entity_key,
    accumulate=target_name
    )
    

    predicted_data = ConvertTo(
        data=predictions.result,
        target_columns=[target_name, 'Prediction'],
        target_datatype=["DECIMAL"]
    )

    predicted_data = predicted_data.result.to_pandas()

    metrics_pd = pd.DataFrame([
        compute_metrics(predicted_data["total_kwh"], predicted_data["Prediction"], "TD_XGBoost"),
    ])
    
    evaluation = {
        'MAE': '{}'.format(metrics_pd.MAE[0]),
        'RMSE': '{}'.format(metrics_pd.RMSE[0]),
        'MAPE_%': '{}'.format(metrics_pd.MAPE[0]),
        'R-square': '{}'.format(metrics_pd.Rsquare[0]),
    }

    with open(f"{context.artifact_output_path}/metrics.json", "w+") as f:
        json.dump(evaluation, f)


    # Calculate feature importance and generate plot
    try:
        model_pdf = model.result.to_pandas()['regression_tree']
        feature_importance = compute_feature_importance(model_pdf)
        feature_importance_df = pd.DataFrame(
            list(feature_importance.items()), columns=['Feature', 'Importance'])
        plot_feature_importance(
            feature_importance, f"{context.artifact_output_path}/feature_importance")
    except:
        feature_importance = {}
    

    predictions_table = "predictions_tmp"
    copy_to_sql(df=predicted_data, table_name=predictions_table,
                index=False, if_exists="replace", temporary=True)

    # calculate stats if training stats exist
    if os.path.exists(f"{context.artifact_input_path}/data_stats.json"):
        record_evaluation_stats(
            features_df=test_df,
            predicted_df=DataFrame.from_query(
                f"SELECT * FROM {predictions_table}"),
            feature_importance=feature_importance,
            context=context
        )
    

    print("All done!")
