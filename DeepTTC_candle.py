import os
import wget
import torch
import subprocess
from Step3_model import *
from Step2_DataEncoding import DataEncoding
from cross_study_validation import run_cross_study_analysis
from model_params_def import preprocess_params, train_params

file_path = os.path.dirname(os.path.realpath(__file__))


preprocess_params = [
    {"name": "use_lincs",
     "type": bool,
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
     },
    {"name": "default_data_url",
     "type": str,
     "default": "'https://ftp.mcs.anl.gov/pub/candle/public/improve/reproducability/DeepTTC/'",
     "help": "Link to model-specific data",
     },
    {"name": "sample_col_name",
     "type": str,
     "default": "COSMIC_ID",
     "help": "ID format of the samples",
     },
]

additional_definitions = [
    {
        "name": "save_data",
        "type": bool,
        "default": False,
        "help": "Whether to save loaded data in pickle files",
    },
    {
        "name": "use_lincs",
        "type": bool,
        "default": False,
        "help": "Whether to use a LINCS subset of genes ONLY",
    },
    {
        "name": "benchmark_dir",
        "type": str,
        "default": None,
        "help": "Directory with the input data for benchmarking",
    },
    {
        "name": "benchmark_result_dir",
        "type": str,
        "default": None,
        "help": "Directory for benchmark output",
    },
    {
        "name": "generate_input_data",
        "type": bool,
        "default": None,
        "help": "'True' for generating input data anew, 'False' for using stored data",
    },
    {
        "name": "mode",
        "type": str,
        "default": None,
        "help": "Execution mode. Available modes are: 'run', 'benchmark'",
    },
    {
        "name": "cancer_id",
        "type": str,
        "default": None,
        "help": "Column name for cancer",
    },
    {
        "name": "drug_id",
        "type": str,
        "default": None,
        "help": "Column name for drug",
    },
    {
        "name": "sample_id",
        "type": str,
        "default": None,
        "help": "Column name for samples/cell lines",
    },
    {
        "name": "target_id",
        "type": str,
        "default": None,
        "help": "Column name for target",
    },
    {
        "name": "train_data_drug",
        "type": str,
        "default": None,
        "help": "Drug data for training",
    },
    {
        "name": "test_data_drug",
        "type": str,
        "default": None,
        "help": "Drug data for testing",
    },
    {
        "name": "train_data_rna",
        "type": str,
        "default": None,
        "help": "RNA data for training",
    },
    {
        "name": "test_data_rna",
        "type": str,
        "default": None,
        "help": "RNA data for testing",
    },
    {
        "name": "vocab_dir",
        "type": str,
        "default": None,
        "help": "Directory with ESPF vocabulary",
    },
    {
        "name": "transformer_num_attention_heads_drug",
        "type": int,
        "default": None,
        "help": "number of attention heads for drug transformer",
    },
    {
        "name": "input_dim_drug",
        "type": int,
        "default": None,
        "help": "Input size of the drug transformer",
    },
    {
        "name": "transformer_emb_size_drug",
        "type": int,
        "default": None,
        "help": "Size of the drug embeddings",
    },
    {
        "name": "transformer_n_layer_drug",
        "type": int,
        "default": None,
        "help": "Number of layers for drug transformer",
    },
    {
        "name": "transformer_intermediate_size_drug",
        "type": int,
        "default": None,
        "help": "Intermediate size of the drug layers",
    },
    {
        "name": "transformer_attention_probs_dropout",
        "type": float,
        "default": None,
        "help": "number of layers for drug transformer",
    },
    {
        "name": "transformer_hidden_dropout_rate",
        "type": float,
        "default": None,
        "help": "dropout rate for transformer hidden layers",
    },
    {
        "name": "dropout",
        "type": float,
        "default": None,
        "help": "dropout rate for common part",
    },
    {
        "name": "input_dim_drug_classifier",
        "type": int,
        "default": None,
        "help": "input dimensions for drug classifier",
    },
    {
        "name": "input_dim_gene_classifier",
        "type": int,
        "default": None,
        "help": "input dimensions for gene classifier",
    },
    {
        "name": "gene_dim",
        "type": int,
        "default": None,
        "help": "Dimensions of the input gene expression data",
    },
]

required = None


# class DeepTTCCandle(candle.Benchmark):
#    def set_locals(self):
#        if required is not None:
#            self.required = set(required)
#        if additional_definitions is not None:
#            self.additional_definitions = additional_definitions


class DataLoader:
    args = None

    def __init__(self, args):
        self.args = args

    def load_data(self):
        train_drug, test_drug, train_rna, test_rna = self._process_data(
            self.args)
        return train_drug, test_drug, train_rna, test_rna

    def save_data(self, train_drug, test_drug, train_rna, test_rna):
        args = self.args
        pickle.dump(train_drug, open(args.train_data_drug, 'wb'), protocol=4)
        pickle.dump(test_drug, open(args.test_data_drug, 'wb'), protocol=4)
        pickle.dump(train_rna, open(args.train_data_rna, 'wb'), protocol=4)
        pickle.dump(test_rna, open(args.test_data_rna, 'wb'), protocol=4)

    def _download_default_dataset(self, default_data_url):
        url = default_data_url
        candle_data_dir = os.getenv("CANDLE_DATA_DIR")
        if candle_data_dir is None:
            candle_data_dir = '.'

        # OUT_DIR = os.path.join(candle_data_dir, 'GDSC_data')
        # this evaluates to /candle_data_dir/GDSC_data
        # print ('outdir before: {}'.format(OUT_DIR))
        OUT_DIR = self.args.data_dir
        print('outdir after: {}'.format(OUT_DIR))
        # print("IN _download_default_dataset")

        url_length = len(url.split('/'))-3
        if not os.path.isdir(OUT_DIR):
            os.mkdir(OUT_DIR)
        subprocess.run(['wget', '--recursive', '--no-clobber', '-nH',
                        f'--cut-dirs={url_length}', '--no-parent', f'--directory-prefix={OUT_DIR}', f'{url}'])
        # wget.download(url, out=OUT_DIR)

    def _process_data(self, args):
        train_drug = test_drug = train_rna = test_rna = None

        if not os.path.exists(args.train_data_rna) or \
                not os.path.exists(args.test_data_rna) or \
                args.generate_input_data:

            self._download_default_dataset(args.default_data_url)

            # obj = DataEncoding(args.vocab_dir, args.cancer_id,
            #                   args.sample_id, args.target_id, args.drug_id)
            obj = DataEncoding(args.data_dir, args.cancer_id,
                               args.sample_id, args.target_id, args.drug_id)
            train_drug, test_drug = obj.Getdata.ByCancer(
                random_seed=args.rng_seed)

            train_drug, train_rna, test_drug, test_rna = obj.encode(
                traindata=train_drug,
                testdata=test_drug)
            print('Train Drug:')
            print(train_drug)
            print('Train RNA:')
            print(train_rna)

            if args.save_data:
                self.save_data(train_drug, test_drug, train_rna, test_rna)
        else:
            train_drug = pickle.load(open(args.train_data_drug, 'rb'))
            test_drug = pickle.load(open(args.test_data_drug, 'rb'))
            train_rna = pickle.load(open(args.train_data_rna, 'rb'))
            test_rna = pickle.load(open(args.test_data_rna, 'rb'))
        return train_drug, test_drug, train_rna, test_rna


a = """
def initialize_parameters(default_model='DeepTTC.default'):
    # Build benchmark object
    common = DeepTTCCandle(file_path,
                           default_model,
                           'torch',
                           prog='deep_ttc',
                           desc='DeepTTC drug response prediction model')

    # Initialize parameters
    candle_data_dir = os.getenv("CANDLE_DATA_DIR")
    gParameters = candle.finalize_parameters(common)
    relative_paths = ['train_data_rna', 'test_data_rna',
                      'vocab_dir', 'train_data_drug', 'test_data_drug',
                      'output_dir']

    for path in relative_paths:
        gParameters[path] = os.path.join(candle_data_dir, gParameters[path])

    dirs_to_check = ['input', 'results']
    for directory in dirs_to_check:
        path = os.path.join(candle_data_dir, directory)
        if not os.path.exists(path):
            os.makedirs(path)

    return gParameters
"""


def get_model(args):
    net = DeepTTC(modeldir=args['output_dir'], args=args)
    return net


def run(args):
    loader = DataLoader(args)
    train_drug, test_drug, train_rna, test_rna = loader.load_data()
    modeldir = args.output_dir
    modelfile = os.path.join(modeldir, args.model_name)
    if not os.path.exists(modeldir):
        os.mkdir(modeldir)
    model = get_model(args)
    model.train(train_drug=train_drug, train_rna=train_rna,
                val_drug=test_drug, val_rna=test_rna)
    model.save_model()
    print("Model Saved :{}".format(modelfile))


def benchmark(args):
    model = get_model(args)
    run_cross_study_analysis(model, args.benchmark_dir,
                             args.benchmark_result_dir, use_lincs=args.use_lincs)


def main():
    # gParameters = initialize_parameters()
    # args = candle.ArgumentStruct(**gParameters)
    pass


if __name__ == "__main__":
    main()
    try:
        torch.cuda.empty_cache()
    except AttributeError:
        pass
