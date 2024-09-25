#!/usr/bin/env python

""" Inference with DeepTTC for drug response prediction.

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
# import pickle
import pandas as pd

# [Req] IMPROVE/CANDLE imports
from improvelib.applications.drug_response_prediction.config import DRPInferConfig
from improvelib.utils import str2bool
import improvelib.utils as frm
from DeepTTC_candle import *

# [Req] Imports from preprocess and train scripts
from deepttc_preprocess_improve import preprocess_params
from deepttc_train_improve import metrics_list, train_params

filepath = Path(__file__).resolve().parent  # [Req]


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
    frm.create_outdir(outdir=params["output_dir"])

    # ------------------------------------------------------
    # [Req] Create data names for test set
    # ------------------------------------------------------
    test_data_fname = frm.build_ml_data_file_name(
        params['data_format'], stage="test")

    # ------------------------------------------------------
    # Prepare dataloaders to load model input data (ML data)
    # ------------------------------------------------------
    print("\nTest data:")
    print(f"test_ml_data_dir: {params['input_data_dir']}")
    # print(f"test_batch: {params['test_batch']}")

    print(test_data_fname)
    test_ml_data_dir = params['input_data_dir']
    # params['test_data_processed']
    test_file_name = frm.build_ml_data_file_name(
        params['data_format'], stage="test")
    test_data_path = f'{test_ml_data_dir}/{test_file_name}'
    test_data = {}
    test_data['drug'] = pd.read_hdf(test_data_path, key='drug')
    test_data['gene_expression'] = pd.read_hdf(
        test_data_path, key='gene_expression')

    # ------------------------------------------------------
    # Load best model and compute predictions
    # ------------------------------------------------------
    # Load the best saved model (as determined based on val data)
    modelpath = modelpath = frm.build_model_path(model_file_name=params["model_file_name"],
                                                 model_file_format=params["model_file_format"],
                                                 model_dir=params["output_dir"])

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

    # device = determine_device(params["cuda_name"])
    if modelpath.exists() == False:
        raise Exception(f"ERROR ! modelpath not found {modelpath}\n")

    args = params  # candle.ArgumentStruct(**params)
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
        y_true=y_label,
        y_pred=y_pred,
        stage="test",
        y_col_name=params["y_col_name"],
        output_dir=params["output_dir"]
    )

    # ------------------------------------------------------
    # [Req] Compute performance scores
    # ------------------------------------------------------
    test_scores = frm.compute_performance_scores(
        y_true=y_label,
        y_pred=y_pred,
        stage="test",
        output_dir=params["output_dir"],
        metric_type=params["metric_type"]
    )

    return test_scores


# [Req]
def main(args):
    # [Req]
    additional_definitions = preprocess_params + train_params
    cfg = DRPInferConfig()
    params = cfg.initialize_parameters(
        pathToModelDir=filepath,
        default_config="deepttc_params.txt",
        additional_definitions=additional_definitions
    )
    test_scores = run(params)
    print("\nFinished model inference.")


# [Req]
if __name__ == "__main__":
    main(sys.argv[1:])
