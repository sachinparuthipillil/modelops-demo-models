from teradataml import (
    DataFrame,
    XGBoost
)

from tmo import (
    record_training_stats,
    tmo_create_context,
    ModelContext
)
from collections import Counter

import matplotlib.pyplot as plt
import pandas as pd
import json


def train(context: ModelContext, **kwargs):
    tmo_create_context()

    feature_names = context.dataset_info.feature_names
    target_name = context.dataset_info.target_names[0]
    entity_key = context.dataset_info.entity_key

    # Extract and cast hyperparameters
    #scale_method = str(context.hyperparams["scale_method"])
    #miss_value = str(context.hyperparams["miss_value"])
    #global_scale = str(context.hyperparams["global_scale"]).lower() in ['true', '1']
    #multiplier = str(context.hyperparams["multiplier"])
    #intercept = str(context.hyperparams["intercept"])
    model_type = str(context.hyperparams["model_type"])
    lambda1 = float(context.hyperparams["lambda1"])
    learning_rate = float(context.hyperparams["learning_rate"])
    max_depth = int(context.hyperparams["max_depth"])
    seed = int(context.hyperparams["seed"])

    # read training dataset from Teradata and convert to pandas
    train_df = DataFrame.from_query(context.dataset_info.sql)


    print("Training using InDB Functions...")

    model = XGBoost(
        data=train_df,
        input_columns=feature_names,
        response_column = target_name,
        lambda1 = lambda1,
        model_type=model_type,
        seed=seed,
        shrinkage_factor=learning_rate,
        max_depth=max_depth
    )


    model.result.to_sql(
        f"model_{context.model_version}", if_exists="replace")
    print(f"Saved trained model in table model_{context.model_version}")

    # Calculate feature importance and generate plot

    print("All done!")
