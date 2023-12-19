""" Inference with GraphDRP for drug response prediction.

Required outputs
----------------
All the outputs from this infer script are saved in params["infer_outdir"].

1. Predictions on test data.
   Raw model predictions calcualted using the trained model on test data. The
   predictions are saved in test_y_data_predicted.csv

2. Prediction performance scores on test data.
   The performance scores are calculated using the raw model predictions and
   the true values for performance metrics specified in the metrics_list. The
   scores are saved as json in test_scores.json
"""

import sys
from pathlib import Path
from typing import Dict

import os
import torch
import pickle
import pandas as pd

# [Req] IMPROVE/CANDLE imports
from improve import framework as frm
from DeepTTC_candle import *

# [Req] Imports from preprocess and train scripts
from deepttc_preprocess_improve import preprocess_params
from deepttc_train_improve import metrics_list

filepath = Path(__file__).resolve().parent  # [Req]

# ---------------------
# [Req] Parameter lists
# ---------------------
# Two parameter lists are required:
# 1. app_infer_params
# 2. model_infer_params
#
# The values for the parameters in both lists should be specified in a
# parameter file that is passed as default_model arg in
# frm.initialize_parameters().

# 1. App-specific params (App: monotherapy drug response prediction)
# Currently, there are no app-specific params in this script.
app_infer_params = []

# 2. Model-specific params (Model: GraphDRP)
# All params in model_infer_params are optional.
# If no params are required by the model, then it should be an empty list.
model_infer_params = []

# [Req] Combine the two lists (the combined parameter list will be passed to
# frm.initialize_parameters() in the main().
infer_params = app_infer_params + model_infer_params
# ---------------------


# [Req]
def run(params):
    """ Run model inference.

    Args:
        params (dict): dict of CANDLE/IMPROVE parameters and parsed values.

    Returns:
        dict: prediction performance scores computed on test data according
            to the metrics_list.
    """
    # import ipdb; ipdb.set_trace()

    # ------------------------------------------------------
    # [Req] Create output dir
    # ------------------------------------------------------
    frm.create_outdir(outdir=params["infer_outdir"])

    # ------------------------------------------------------
    # [Req] Create data names for test set
    # ------------------------------------------------------
    test_data_fname = frm.build_ml_data_name(params, stage="test")

    # ------------------------------------------------------
    # Prepare dataloaders to load model input data (ML data)
    # ------------------------------------------------------
    print("\nTest data:")
    print(f"test_ml_data_dir: {params['test_ml_data_dir']}")
    # print(f"test_batch: {params['test_batch']}")

    print(test_data_fname)
    test_data_path = test_data_fname  # params['test_data_processed']
    test_ml_data_dir = params['test_ml_data_dir']
    test_data = pickle.load(
        open(f'{test_ml_data_dir}/test.pickle', 'rb'))

    # ------------------------------------------------------
    # Load best model and compute predictions
    # ------------------------------------------------------
    # Load the best saved model (as determined based on val data)
    modelpath = frm.build_model_path(
        params, model_dir=params["model_outdir"])  # [Req]

    def determine_device(cuda_name_from_params):
        """Determine device to run PyTorch functions.

        PyTorch functions can run on CPU or on GPU. In the latter case, it
        also takes into account the GPU devices requested for the run.

        :params str cuda_name_from_params: GPUs specified for the run.

        :return: Device available for running PyTorch functionality.
        :rtype: str
        """
        cuda_avail = torch.cuda.is_available()
        print("GPU available: ", cuda_avail)
        if cuda_avail:  # GPU available
            # -----------------------------
            # CUDA device from env var
            cuda_env_visible = os.getenv("CUDA_VISIBLE_DEVICES")
            if cuda_env_visible is not None:
                # Note! When one or multiple device numbers are passed via
                # CUDA_VISIBLE_DEVICES, the values in python script are reindexed
                # and start from 0.
                print("CUDA_VISIBLE_DEVICES: ", cuda_env_visible)
                cuda_name = "cuda:0"
            else:
                cuda_name = cuda_name_from_params
            device = cuda_name
        else:
            device = "cpu"

        return device

    device = determine_device(params["cuda_name"])
    if modelpath.exists() == False:
        raise Exception(f"ERROR ! modelpath not found {modelpath}\n")

    args = candle.ArgumentStruct(**params)
    print(modelpath)
    model = DeepTTC(modeldir=modelpath, args=args)
    model.load_pretrained(modelpath)
    # Compute predictions
    y_label, y_pred, mse, rmse, person, p_val, spearman, s_p_val, CI = model.predict(
        test_data['drug'], test_data['gene_expression'])

    # ------------------------------------------------------
    # [Req] Save raw predictions in dataframe
    # ------------------------------------------------------
    frm.store_predictions_df(
        params,
        y_true=y_label, y_pred=y_pred, stage="test",
        outdir=params["infer_outdir"]
    )

    # ------------------------------------------------------
    # [Req] Compute performance scores
    # ------------------------------------------------------
    test_scores = frm.compute_performace_scores(
        params,
        y_true=y_label, y_pred=y_pred, stage="test",
        outdir=params["infer_outdir"], metrics=metrics_list
    )

    return test_scores


# [Req]
def main(args):
    # [Req]
    additional_definitions = preprocess_params + infer_params
    params = frm.initialize_parameters(
        filepath,
        # default_model="graphdrp_default_model.txt",
        # default_model="graphdrp_params.txt",
        # default_model="params_ws.txt",
        default_model="DeepTTC.default",
        additional_definitions=additional_definitions,
        # required=req_infer_args,
        required=None,
    )
    test_scores = run(params)
    print("\nFinished model inference.")


# [Req]
if __name__ == "__main__":
    main(sys.argv[1:])
