#!/usr/bin/env python
import subprocess
import joblib
import pandas as pd
import numpy as np
from typing import Dict
from pathlib import Path
# import pickle

import os
import sys
from Step2_DataEncoding import DataEncoding
from sklearn.preprocessing import StandardScaler, MaxAbsScaler, MinMaxScaler, RobustScaler

# [Req] IMPROVE/CANDLE imports
import candle
from improve import framework as frm
from improve import drug_resp_pred as drp


filepath = Path(__file__).resolve().parent  # [Req]
IMPROVE_DATA_DIR = Path(os.environ["IMPROVE_DATA_DIR"])

# ---------------------
# [Req] Parameter lists
# ---------------------
app_preproc_params = [
    {"name": "y_data_files",  # default
     "type": str,
     "help": "List of files that contain the y (prediction variable) data. \
             Example: [['response.tsv']]",
     },
    {"name": "x_data_canc_files",  # required
     "type": str,
     "help": "List of feature files including gene_system_identifer. Examples: \n\
             1) [['cancer_gene_expression.tsv', ['Gene_Symbol']]] \n\
             2) [['cancer_copy_number.tsv', ['Ensembl', 'Entrez']]].",
     },
    {"name": "x_data_drug_files",  # required
     "type": str,
     "help": "List of feature files. Examples: \n\
             1) [['drug_SMILES.tsv']] \n\
             2) [['drug_SMILES.tsv'], ['drug_ecfp4_nbits512.tsv']]",
     },
    {"name": "canc_col_name",
     "default": "improve_sample_id",  # default
     "type": str,
     "help": "Column name in the y (response) data file that contains the cancer sample ids.",
     },
    {"name": "drug_col_name",  # default
     "default": "improve_chem_id",
     "type": str,
     "help": "Column name in the y (response) data file that contains the drug ids.",
     },
]

# 2. Model-specific params (Model: LightGBM)
# All params in model_preproc_params are optional.
# If no params are required by the model, then it should be an empty list.
model_preproc_params = [
    {"name": "use_lincs",
     "type": frm.str2bool,
     "default": True,
     "help": "Flag to indicate if landmark genes are used for gene selection.",
     },
    {"name": "scaling",
     "type": str,
     "default": "std",
     "choice": ["std", "minmax", "miabs", "robust"],
     "help": "Scaler for gene expression and Mordred descriptors data.",
     },
    {"name": "ge_scaler_fname",
     "type": str,
     "default": "x_data_gene_expression_scaler.gz",
     "help": "File name to save the gene expression scaler object.",
     }
]

# [Req]
preprocess_params = app_preproc_params + model_preproc_params


# TO REMOVE
def load_response_data(inpath_dict: frm.DataPathDict,
                       y_file_name: str,
                       source: str,
                       split_id: int,
                       stage: str,
                       canc_col_name="improve_sample_id",
                       drug_col_name="improve_chem_id",
                       sep: str = "\t",
                       verbose: bool = True) -> pd.DataFrame:
    """
    Returns dataframe with cancer ids, drug ids, and drug response values.
    Samples from the original drug response file are filtered based on
    the specified split ids.

    :params: Dict inpath_dict: Dictionary of paths and info about raw
             data input directories.
    :params: str y_file_name: Name of file for reading the y_data.
    :params: str source: DRP source name.
    :params: int split_id: Split id. If -1, use all data. Note that this
             assumes that split_id has been constructed to take into
             account all the data sources.
    :params: str stage: Type of partition to read. One of the following:
             'train', 'val', 'test'.
    :params: str canc_col_name: Column name that contains the cancer
             sample ids. Default: "improve_sample_id".
    :params: str drug_col_name: Column name that contains the drug ids.
             Default: "improve_chem_id".n
    :params: str sep: Separator used in data file.
    :params: bool verbose: Flag for verbosity. If True, info about
             computations is displayed. Default: True.

    :return: Dataframe that contains single drug response values.
    :rtype: pd.Dataframe
    """
    y_data_file = inpath_dict["y_data"] / y_file_name
    if y_data_file.exists() == False:
        raise Exception(f"ERROR ! {y_file_name} file not available.\n")
    # Read y_data_file
    df = pd.read_csv(y_data_file, sep=sep)

    # Get a subset of samples if split_id is different to -1
    if split_id > -1:
        split_file_name = f"{source}_split_{split_id}_{stage}.txt"
    else:
        split_file_name = f"{source}_all.txt"
    insplit = inpath_dict["splits"] / split_file_name
    if insplit.exists() == False:
        raise Exception(f"ERROR ! {split_file_name} file not available.\n")
    ids = pd.read_csv(insplit, header=None)[0].tolist()
    df = df.loc[ids]

    df = df.reset_index(drop=True)
    if verbose:
        print(f"Data read: {y_file_name}, Filtered by: {split_file_name}")
        print(f"Shape of constructed response data framework: {df.shape}")
        print(f"Unique cells:  {df[canc_col_name].nunique()}")
        print(f"Unique drugs:  {df[drug_col_name].nunique()}")
    return df


# TO REMOVE
def compose_data_arrays(df_response, df_drug, df_cell, drug_col_name, canc_col_name):
    """ Returns drug and cancer feature data, and response values.

    :params: pd.Dataframe df_response: drug response dataframe. This
             already has been filtered to three columns: drug_id,
             cell_id and drug_response.
    :params: pd.Dataframe df_drug: drug features dataframe.
    :params: pd.Dataframe df_cell: cell features dataframe.
    :params: str drug_col_name: Column name that contains the drug ids.
    :params: str canc_col_name: Column name that contains the cancer sample ids.

    :return: Numpy arrays with drug features, cell features and responses
            xd, xc, y
    :rtype: np.array
    """
    xd = []  # To collect drug features
    xc = []  # To collect cell features
    y = []  # To collect responses
    # To collect missing or corrupted data
    # nan_rsp_list = []
    # miss_cell = []
    # miss_drug = []
    count_nan_rsp = 0
    count_miss_cell = 0
    count_miss_drug = 0
    # Convert to indices for rapid lookup
    df_drug = df_drug.set_index([drug_col_name])
    df_cell = df_cell.set_index([canc_col_name])
    # tuples of (drug name, cell id, response)
    for i in range(df_response.shape[0]):
        if i > 0 and (i % 15000 == 0):
            print(i)
        drug, cell, rsp = df_response.iloc[i, :].values.tolist()
        if np.isnan(rsp):
            # nan_rsp_list.append(rsp)
            count_nan_rsp += 1
        # If drug and cell features are available
        try:  # Look for drug
            drug_features = df_drug.loc[drug]
        except KeyError:  # drug not found
            # miss_drug.append(drug)
            count_miss_drug += 1
        else:  # Look for cell
            try:
                cell_features = df_cell.loc[cell]
            except KeyError:  # cell not found
                # miss_cell.append(cell)
                count_miss_cell += 1
            else:  # Both drug and cell were found
                # xd contains list of drug feature vectors
                xd.append(drug_features.values)
                # xc contains list of cell feature vectors
                xc.append(cell_features.values)
                y.append(rsp)

    print("Number of NaN responses: ", count_nan_rsp)
    print("Number of drugs not found: ", count_miss_drug)
    print("Number of cells not found: ", count_miss_cell)
    # Reset index
    df_drug = df_drug.reset_index()
    df_cell = df_cell.reset_index()

    return np.asarray(xd).squeeze(), np.asarray(xc), np.asarray(y)

# TO REMOVE


def preprocess_drug_data(args, drug_data):
    args.vocab_dir = os.path.join(IMPROVE_DATA_DIR, args.vocab_dir)
    obj = DataEncoding(args, args.vocab_dir, args.cancer_id,
                       args.sample_id, args.target_id, args.drug_id)
    drug_smiles = drug_data

    drugid2smile = dict(
        zip(drug_smiles[args.drug_id], drug_smiles['SMILES']))
    smile_encode = pd.Series(drug_smiles['SMILES'].unique()).apply(
        obj._drug2emb_encoder)
    uniq_smile_dict = dict(
        zip(drug_smiles['SMILES'].unique(), smile_encode))

    # drug_data.drop(['SMILES'], inplace=True, axis=1)
    drug_data['smiles'] = [drugid2smile[i] for i in drug_data[args.drug_id]]
    drug_data['drug_encoding'] = [uniq_smile_dict[i]
                                  for i in drug_data['smiles']]
    drug_data = drug_data.reset_index()

    return drug_data


def preprocess(args, rna_data, drug_data, response_data, response_metric='AUC'):
    args.vocab_dir = os.path.join(IMPROVE_DATA_DIR, args.vocab_dir)
    obj = DataEncoding(args, args.vocab_dir, args.cancer_id,
                       args.sample_id, args.target_id, args.drug_id)
    drug_smiles = drug_data

    drugid2smile = dict(
        zip(drug_smiles[args.drug_id], drug_smiles['SMILES']))
    smile_encode = pd.Series(drug_smiles['SMILES'].unique()).apply(
        obj._drug2emb_encoder)
    uniq_smile_dict = dict(
        zip(drug_smiles['SMILES'].unique(), smile_encode))

    # drug_data.drop(['SMILES'], inplace=True, axis=1)
    drug_data['smiles'] = [drugid2smile[i] for i in drug_data[args.drug_id]]
    drug_data['drug_encoding'] = [uniq_smile_dict[i]
                                  for i in drug_data['smiles']]
    drug_data = drug_data.reset_index()

    response_data = response_data[[
        args.canc_col_name, args.drug_id, response_metric]]
    response_data.columns = [args.canc_col_name, args.drug_id, 'Label']
    drug_data = pd.merge(response_data, drug_data,
                         on=args.drug_id, how='inner')
    # drug_data['Label'] = response_data['AUC']

    # response_data = response_data[['CancID', 'DrugID', response_metric]]
    # response_data.columns = ['CancID', 'DrugID', 'Label']
    # response_data = response_data[['CancID', 'DrugID']]

    # rna_data = pd.merge(response_data, rna_data, on='CancID', how='inner')
    # train_rnadata = train_rnadata.T
    drug_data.index = range(drug_data.shape[0])
    rna_data.index = range(rna_data.shape[0])

    print('Preprocessing...!!!')
    print(np.shape(rna_data), np.shape(drug_data))
    # print(list(rna_data.columns))
    return rna_data, drug_data


def build_common_data(params: Dict):
    """Construct common feature data frames.

    :params: Dict params: A Python dictionary of CANDLE/IMPROVE keywords
             and parsed values.
    :params: Dict inputdtd: Path to directories of input data stored in
            dictionary with str key str and Path value.

    :return: drug and cell dataframes and smiles graphs
    :rtype: pd.DataFrame
    """
    # ------------------------------------------------------
    # [Req] Build paths and create output dir
    # ------------------------------------------------------
    # Build paths for raw_data, x_data, y_data, splits
    params = frm.build_paths(params)

    # Create output dir for model input data (to save preprocessed ML data)
    frm.create_outdir(outdir=params["ml_data_outdir"])

    # ------------------------------------------------------
    # [Req] Load X data (feature representations)
    # ------------------------------------------------------
    omics_loader = drp.OmicsLoader(params)
    drugs_loader = drp.DrugsLoader(params)

    gene_expression = omics_loader.dfs['cancer_gene_expression.tsv']
    df_drug = drugs_loader.dfs['drug_SMILES.tsv']
    df_drug = df_drug.reset_index()
    df_drug.columns = [params["drug_col_name"], "smiles"]
    params['drug_id'] = params["drug_col_name"]
    df_drug["SMILES"] = df_drug["smiles"]
    # breakpoint()

    # ------------------------------------------------------
    # Further preprocess X data
    # ------------------------------------------------------
    # Gene selection (based on LINCS landmark genes)
    def gene_selection(df, genes_fpath, canc_col_name):
        """ Takes a dataframe omics data (e.g., gene expression) and retains only
        the genes specified in genes_fpath.
        """
        with open(genes_fpath) as f:
            genes = [str(line.rstrip()) for line in f]
        # genes = ["ge_" + str(g) for g in genes]  # This is for our legacy data
        # print("Genes count: {}".format(len(set(genes).intersection(set(df.columns[1:])))))
        genes = list(set(genes).intersection(set(df.columns[1:])))
        # genes = drp.common_elements(genes, df.columns[1:])
        cols = [canc_col_name] + genes
        return df[cols]

    if params["use_lincs"]:
        genes_fpath = filepath/"landmark_genes"
        gene_expression = gene_selection(
            gene_expression, genes_fpath, canc_col_name=params["canc_col_name"])

    return df_drug, gene_expression


def _download_default_dataset(default_data_url):
    url = default_data_url
    improve_data_dir = os.getenv("IMPROVE_DATA_DIR")
    if improve_data_dir is None:
        improve_data_dir = '.'

    OUT_DIR = improve_data_dir
    print('outdir after: {}'.format(OUT_DIR))

    url_length = len(url.split('/'))-4
    if not os.path.isdir(OUT_DIR):
        os.mkdir(OUT_DIR)
    subprocess.run(['wget', '--recursive', '--no-clobber', '-nH',
                    f'--cut-dirs={url_length}', '--no-parent', f'--directory-prefix={OUT_DIR}', f'{url}'])


def download_model_data(params):
    _download_default_dataset(params["default_data_url"])


def download_dataset(params):
    mainpath = Path(os.environ["IMPROVE_DATA_DIR"])
    command = f'wget --directory-prefix={mainpath} --cut-dirs=7 -nH -np -m ftp://ftp.mcs.anl.gov/pub/candle/public/improve/benchmarks/single_drug_drp/benchmark-data-pilot1/csa_data'
    tokens = command.split(' ')
    subprocess.run(tokens)


def prepare_dataframe(args, gene_expression, smiles, responses):
    gene_expression, drug_data = preprocess(args,
                                            gene_expression, smiles, responses, args.y_col_name)
    drug_data = drug_data.drop(['index'], axis=1)
    drug_columns = [
        x for x in drug_data.columns if x not in [args.canc_col_name, args.drug_col_name]]
    # data = pd.merge(gene_expression, drug_data, on='DrugID', how='inner')
    data = pd.merge(gene_expression, drug_data,
                    on=args.canc_col_name, how='inner')
    gene_expression = gene_expression.drop([args.canc_col_name], axis=1)
    gene_expression_columns = gene_expression.columns

    return data, gene_expression_columns, drug_columns


def get_common_samples(
        df1: pd.DataFrame,
        df2: pd.DataFrame,
        ref_col: str):
    """
    Search for common data in reference column and retain only .

    Args:
        df1, df2 (pd.DataFrame): dataframes
        ref_col (str): the ref column to find the common values

    Returns:
        df1, df2 after filtering for common data.

    Example:
        Before:

        df1:
        col1    ref_col     col3            col4
        CCLE    ACH-000956      Drug_749        0.7153
        CCLE    ACH-000325      Drug_1326       0.9579
        CCLE    ACH-000475      Drug_490        0.213

        df2:
        ref_col     col2                col3                col4
        ACH-000956  3.5619370596224327  0.0976107966264223      4.888499735514123
        ACH-000179  5.202025844609336   3.5046203924035524      3.5058909297299574
        ACH-000325  6.016139702655253   0.6040713236688608      0.0285691521967709

        After:

        df1:
        col1    ref_col     col3            col4
        CCLE    ACH-000956      Drug_749        0.7153
        CCLE    ACH-000325      Drug_1326       0.9579

        df2:
        ref_col     col2                col3                col4
        ACH-000956  3.5619370596224327  0.0976107966264223      4.888499735514123
        ACH-000325  6.016139702655253   0.6040713236688608      0.0285691521967709
    """
    # Retain df1 and df2 samples with common ref_col
    common_ids = list(set(df1[ref_col]).intersection(df2[ref_col]))
    df1 = df1[df1[ref_col].isin(common_ids)].reset_index(drop=True)
    df2 = df2[df2[ref_col].isin(common_ids)].reset_index(drop=True)
    return df1, df2


def scale_df(dataf, scaler_name="std", scaler=None, verbose=False):
    """ Returns a dataframe with scaled data.

    It can create a new scaler or use the scaler passed or return the
    data as it is. If `scaler_name` is None, no scaling is applied. If
    `scaler` is None, a new scaler is constructed. If `scaler` is not
    None, and `scaler_name` is not None, the scaler passed is used for
    scaling the data frame.

    Args:
        dataf: Pandas dataframe to scale.
        scaler_name: Name of scikit learn scaler to apply. Options:
                     ["minabs", "minmax", "std", "none"]. Default: std
                     standard scaling.
        scaler: Scikit object to use, in case it was created already.
                Default: None, create scikit scaling object of
                specified type.
        verbose: Flag specifying if verbose message printing is desired.
                 Default: False, no verbose print.

    Returns:
        pd.Dataframe: dataframe that contains drug response values.
        scaler: Scikit object used for scaling.
    """
    if scaler_name is None or scaler_name == "none":
        if verbose:
            print("Scaler is None (no df scaling).")
        return dataf, None

    # Scale data
    # Select only numerical columns in data frame
    df_num = dataf.select_dtypes(include="number")

    if scaler is None:  # Create scikit scaler object
        if scaler_name == "std":
            scaler = StandardScaler()
        elif scaler_name == "minmax":
            scaler = MinMaxScaler()
        elif scaler_name == "minabs":
            scaler = MaxAbsScaler()
        elif scaler_name == "robust":
            scaler = RobustScaler()
        else:
            print(
                f"The specified scaler {scaler_name} is not implemented (no df scaling).")
            return dataf, None

        # Scale data according to new scaler
        df_norm = scaler.fit_transform(df_num)
    else:  # Apply passed scikit scaler
        # Scale data according to specified scaler
        df_norm = scaler.transform(df_num)

    # Copy back scaled data to data frame
    dataf[df_num.columns] = df_norm
    return dataf, scaler


def build_stage_dependent_data(params: Dict,
                               stage: str,
                               df_drug: pd.DataFrame,
                               df_cell_all: pd.DataFrame,
                               scaler,
                               ):
    """Construct feature and ouput arrays according to training stage.

    :params: Dict params: A Python dictionary of CANDLE/IMPROVE keywords
             and parsed values.
    :params: Dict inputdtd: Path to directories of input data stored in
            dictionary with str key str and Path value.
    :params: Dict outputdtd: Path to directories for output data stored
            in dictionary with str key str and Path value.
    :params: str stage: Type of partition to read. One of the following:
             'train', 'val', 'test'.
    :params: str source: DRP source name.
    :params: int split_id: Split id. If -1, use all data. Note that this
             assumes that split_id has been constructed to take into
             account all the data sources.
    :params: pd.Dataframe df_drug: Pandas dataframe with drug features.
    :params: pd.Dataframe df_cell_all: Pandas dataframe with cell features.
    :params: scikit scaler: Scikit object for scaling data.
    """
    args = candle.ArgumentStruct(**params)
    stages = {"train": params["train_split_file"],
              "val": params["val_split_file"],
              "test": params["test_split_file"]}
    # --------------------------------
    # [Req] Load response data
    # --------------------------------
    print(stages["test"])
    df_response = drp.DrugResponseLoader(params,
                                         split_file=stages[stage],
                                         verbose=False).dfs["response.tsv"]

    # Retain (canc, drug) response samples for which omic data is available
    df_y, df_cell = get_common_samples(df1=df_response,
                                       df2=df_cell_all,
                                       ref_col=params["canc_col_name"])
    print(df_y[[params["canc_col_name"], params["drug_col_name"]]].nunique())

    # Normalize features using training set
    if stage == "train":  # Ignore scaler object even if specified
        # Normalize
        df_cell, scaler = scale_df(df_cell, scaler_name=params["scaling"])
        if params["scaling"] is not None and params["scaling"] != "none":
            # Store normalization object
            scaler_fname = os.path.join(
                params["ml_data_outdir"], "cell_xdata_scaler.gz")
            joblib.dump(scaler, scaler_fname)
            print("Scaling object created is stored in: ", scaler_fname)
    else:
        # Use passed scikit scaler object
        df_cell, _ = scale_df(df_cell, scaler=scaler)

    # Sub-select desired response column (y_col_name)
    # And reduce response dataframe to 3 columns: drug_id, cell_id and selected drug_response
    df_y = df_y[[params["drug_col_name"],
                 params["canc_col_name"], params["y_col_name"]]]
    # Combine data
    data, gene_expression_columns, drug_columns = prepare_dataframe(args,
                                                                    df_cell, df_drug, df_y)
    # xd, xc, y = compose_data_arrays(
    #    df_y, df_drug, df_cell, params["drug_col_name"], params["canc_col_name"])
    # Save the processed (all) data as PyTorch dataset
    # xd['Label'] = y

    # Save the subset of y data
    # fname = f"{stage}_{params['y_data_suffix']}.csv"
    df_gene_expression = data[gene_expression_columns]
    df_drug = data[drug_columns]
    out_path = os.path.join(params["ml_data_outdir"], f'{stage}.h5')
    print(out_path)
    df_output = {'drug': df_drug, 'gene_expression': df_gene_expression}
    for key in df_output:
        df_output[key].to_hdf(out_path, key)
    # pickle.dump(df_output, open(out_path, 'wb'), protocol=4)

    y_df = pd.DataFrame(
        data[['Label', params['canc_col_name'], params['drug_col_name']]])
    frm.save_stage_ydf(y_df, params, stage)

    return scaler


def run(params):

    df_drug, df_cell_all = build_common_data(params)
    stages = ["train", "val", "test"]
    scaler = None
    for st in stages:
        print(f"Building stage: {st}")
        scaler = build_stage_dependent_data(params,
                                            st,
                                            df_drug,
                                            df_cell_all,
                                            scaler,
                                            )

    return


def main(args):
    # [Req]
    additional_definitions = preprocess_params
    params = frm.initialize_parameters(
        filepath,
        default_model="DeepTTC.default",
        additional_definitions=additional_definitions,
        required=None,
    )

    print("\nFinished data preprocessing.")
    download_model_data(params)
    # download_dataset(params)

    ml_data_outdir = run(params)


# [Req]
if __name__ == "__main__":
    main(sys.argv[1:])
