import os
import json
# import pickle
import pandas as pd
from pathlib import Path
from DeepTTC_candle import get_model
from improvelib.applications.drug_response_prediction.config import DRPTrainConfig
from improvelib.utils import str2bool
import improvelib.utils as frm
from improvelib.metrics import compute_metrics
from model_params_def import preprocess_params, train_params

# 1. App-specific params (App: monotherapy drug response prediction)
# Currently, there are no app-specific params for this script.
app_train_params = []

# 2. Model-specific params (Model: GraphDRP)
# All params in model_train_params are optional.
# If no params are required by the model, then it should be an empty list.

# Combine the two lists (the combined parameter list will be passed to
# frm.initialize_parameters() in the main().
train_params = app_train_params + train_params

metrics_list = ["mse", "rmse", "pcc", "scc", "r2"]


def compute_performace_scores(y_true, y_pred, metrics, outdtd, stage):
    """Evaluate predictions according to specified metrics.

    Metrics are evaluated. Scores are stored in specified path and returned.

    :params array y_true: Array with ground truth values.
    :params array y_pred: Array with model predictions.
    :params listr metrics: List of strings with metrics to evaluate.
    :params Dict outdtd: Dictionary with path to store scores.
    :params str stage: String specified if evaluation is with respect to
            validation or testing set.

    :return: Python dictionary with metrics evaluated and corresponding scores.
    :rtype: dict
    """
    scores = compute_metrics(y_true, y_pred, metrics)
    key = f"{stage}_loss"
    scores[key] = scores["mse"]

    with open(outdtd["scores"], "w", encoding="utf-8") as f:
        json.dump(scores, f, ensure_ascii=False, indent=4)

    # Performance scores for Supervisor HPO
    if stage == "val":
        print("\nIMPROVE_RESULT val_loss:\t{}\n".format(scores["mse"]))
        print("Validation scores:\n\t{}".format(scores))
    elif stage == "test":
        print("Inference scores:\n\t{}".format(scores))
    return scores


def run(params):
    """ Execute specified model training.

    :params: Dict params: A dictionary of CANDLE/IMPROVE keywords and parsed values.

    :return: List of floats evaluating model predictions according to
             specified metrics.
    :rtype: float list
    """
    data_dir = Path(params["input_dir"])
    train_data_path = data_dir / \
        frm.build_ml_data_file_name(params['data_format'], stage="val")
    val_data_path = data_dir / \
        frm.build_ml_data_file_name(params['data_format'], stage="val")
    # test_data_path = model_dir/'test.h5'

    train_data = {}
    train_data['drug'] = pd.read_hdf(train_data_path, key='drug')
    train_data['gene_expression'] = pd.read_hdf(
        train_data_path, key='gene_expression')
    val_data = {}
    val_data['drug'] = pd.read_hdf(val_data_path, key='drug')
    val_data['gene_expression'] = pd.read_hdf(
        val_data_path, key='gene_expression')
    # test_data = pickle.load(open(test_data_path, 'rb'))
    modeldir = params['output_dir']
    modelpath = frm.build_model_path(model_file_name=params["model_file_name"],
                                     model_file_format=params["model_file_format"],
                                     model_dir=params["output_dir"])

    if not os.path.exists(modeldir):
        os.mkdir(modeldir)
    model = get_model(params)
    model = model.train(train_drug=train_data['drug'], train_rna=train_data['gene_expression'],
                        val_drug=val_data['drug'], val_rna=val_data['gene_expression'])
    print(f'Saving model to {modelpath}')

    ############### HACK!!!! ################
    # os.makedirs(str(modelpath).split('.')[0], exist_ok=True)
    #########################################
    model.save_model(modelpath)
    # model.save_model(modelfile)
    print("Model Saved :{}".format(modelpath))

    y_label, y_pred, mse, rmse, person, p_val, spearman, s_p_val, CI = model.predict(
        val_data['drug'], val_data['gene_expression'])

    # Store predictions in data frame
    # Attempt to concat predictions with the cancer and drug ids, and the true values
    # If data frame found, then y_true is read from data frame and returned
    # Otherwise, only a partial data frame is stored (with val_true and val_pred)
    # and y_true is equal to pytorch loaded val_true
    # This includes true and predicted values
    # pred_col_name = params["y_col_name"] + params["pred_col_name_suffix"]
    # true_col_name = params["y_col_name"] + "_true"
    # df = pd.DataFrame({true_col_name: y_label, pred_col_name: y_pred})

    # Save preds df
    opath = Path(params["output_dir"])
    os.makedirs(opath, exist_ok=True)
    # ------------------------------------------------------
    # [Req] Save raw predictions in dataframe
    # ------------------------------------------------------
    frm.store_predictions_df(
        y_true=y_label,
        y_pred=y_pred,
        stage="val",
        y_col_name=params["y_col_name"],
        output_dir=params["output_dir"]
    )

    # ------------------------------------------------------
    # [Req] Compute performance scores
    # ------------------------------------------------------
    val_scores = frm.compute_performance_scores(
        y_true=y_label,
        y_pred=y_pred,
        stage="val",
        output_dir=params["output_dir"],
        metric_type=params["metric_type"]
    )

    return val_scores


def main():
    filepath = Path(__file__).resolve().parent

    additional_definitions = preprocess_params + \
        train_params
    cfg = DRPTrainConfig()
    params = cfg.initialize_parameters(pathToModelDir=filepath,
                                       default_config="deepttc_params.txt",
                                       additional_definitions=additional_definitions
                                       )
    run(params)
    print("\nFinished training DeepTTC model.")


if __name__ == "__main__":
    main()
