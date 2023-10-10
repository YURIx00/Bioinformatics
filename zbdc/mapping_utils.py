import numpy as np
import pandas as pd
import scanpy as sc
import torch
import logging

from scipy.sparse.csc import csc_matrix
from scipy.sparse.csr import csr_matrix

from . import mapping_optimizer as mo
from . import utils as ut

# from torch.nn.functional import cosine_similarity

logging.getLogger().setLevel(logging.INFO)


def pp_sc_datas(adata_sc, adata_sp, marker_genes=None):
    # 去除零表达的基因
    sc.pp.filter_genes(adata_sc, min_cells=1)
    sc.pp.filter_genes(adata_sp, min_cells=1)

    if marker_genes is None:
        marker_genes = adata_sc.var.index

    # 去除重复的基因
    adata_sc.var_names_make_unique()
    adata_sp.var_names_make_unique()

    # adata_sc和adata_sp重叠的基因
    overlap_genes = list(set(adata_sc.var.index) & set(adata_sp.var.index))
    adata_sc.uns["overlap_genes"] = overlap_genes
    adata_sp.uns["overlap_genes"] = overlap_genes
    logging.info(
        "{} overlap genes genes are saved in `uns``overlap_genes` of both single cell and spatial Anndatas.".format(
            len(overlap_genes)
        )
    )

    # 待训练的基因：adata_sc、adata_sp、maker_genes重叠的基因
    train_genes = list(set(adata_sc.var.index) & set(adata_sp.var.index) & set(marker_genes))
    adata_sc.uns["training_genes"] = train_genes
    adata_sp.uns["training_genes"] = train_genes
    logging.info(
        "{} training genes genes are saved in `uns``training_genes` of both single cell and spatial Anndatas.".format(
            len(train_genes)
        )
    )

    # 计算空间数据的均匀密度：1/spot点数
    adata_sp.obs["uniform_density"] = np.ones(adata_sp.X.shape[0]) / adata_sp.X.shape[0]
    logging.info(
        "uniform based density prior {} is calculated and saved in `obs``uniform_density` of the spatial Anndata.".format(
            adata_sp.obs["uniform_density"]
        )
    )

    # 计算空间数据的RNA计数百分比密度：
    rna_count_per_spot = np.array(adata_sp.X.sum(axis=1)).squeeze()
    adata_sp.obs["rna_count_based_density"] = rna_count_per_spot / np.sum(rna_count_per_spot)
    logging.info(
        "rna count based density prior {} is calculated and saved in `obs``rna_count_based_density` of the spatial "
        "Anndata.".format(
            adata_sp.obs["rna_count_based_density"]
        )
    )


def map_cells_to_space(
        adata_sc,
        adata_sp,
        cv_train_genes=None,
        mode="cells",
        device="cpu",
        learning_rate=0.1,
        num_epochs=1000,
        scale=True,
        lambda_a=1.0,
        lambda_d=1.0,
        lambda_r=0,
        random_state=None,
        verbose=True,
        density_prior=None,
):
    """
    Map single cell data (`adata_sc`) on spatial data (`adata_sp`).

    Args:
        adata_sc (AnnData): single cell data
        adata_sp (AnnData): gene spatial data
        cv_train_genes (list): Optional. Training gene list. Default is None.
        cluster_label (str): Optional. Field in `adata_sc.obs` used for aggregating single cell data. Only valid for `mode=clusters`.
        mode (str): Optional. Tangram mapping mode. Currently supported: 'cell', 'clusters', 'constrained'. Default is 'cell'.
        device (string or torch.device): Optional. Default is 'cpu'.
        learning_rate (float): Optional. Learning rate for the optimizer. Default is 0.1.
        num_epochs (int): Optional. Number of epochs. Default is 1000.
        lambda_a (float): Optional. Hyperparameter for the G term of the optimizer. Default is 1.
        lambda_b (float): Optional. Hyperparameter for the bias term of the optimizer. Default is 1.
        random_state (int): Optional. pass an int to reproduce training. Default is None.
        verbose (bool): Optional. If print training details. Default is True.
        density_prior (str, ndarray or None): Spatial density of spots, when is a string, value can be 'rna_count_based' or 'uniform', when is a ndarray, shape = (number_spots,). This array should satisfy the constraints sum() == 1. If None, the density term is ignored. Default value is 'rna_count_based'.

    Returns:
        a cell-by-spot AnnData containing the probability of mapping cell i on spot j.
        The `uns` field of the returned AnnData contains the training genes.
        :param lambda_a:
    """

    # check invalid values for arguments
    if lambda_a == 0:
        raise ValueError("lambda_a cannot be 0.")

    # Check if training_genes key exist/is valid in adatas.uns
    if not set(["training_genes", "overlap_genes"]).issubset(set(adata_sc.uns.keys())):
        raise ValueError("Missing tangram parameters. Run `pp_adatas()`.")

    assert list(adata_sp.uns["training_genes"]) == list(adata_sc.uns["training_genes"])

    # get training_genes
    if cv_train_genes is None:
        training_genes = adata_sc.uns["training_genes"]
    elif cv_train_genes is not None:
        if set(cv_train_genes).issubset(set(adata_sc.uns["training_genes"])):
            training_genes = cv_train_genes
        else:
            raise ValueError(
                "Given training genes list should be subset of two AnnDatas."
            )

    logging.info("Allocate tensors for mapping.")
    # Allocate tensors (AnnData matrix can be sparse or not)

    if isinstance(adata_sc.X, csc_matrix) or isinstance(adata_sc.X, csr_matrix):
        S = np.array(adata_sc[:, training_genes].X.toarray(), dtype="float32", )
    elif isinstance(adata_sc.X, np.ndarray):
        S = np.array(adata_sc[:, training_genes].X.toarray(), dtype="float32", )
    else:
        X_type = type(adata_sc.X)
        logging.error("AnnData X has unrecognized type: {}".format(X_type))
        raise NotImplementedError

    if isinstance(adata_sp.X, csc_matrix) or isinstance(adata_sp.X, csr_matrix):
        G = np.array(adata_sp[:, training_genes].X.toarray(), dtype="float32")
    elif isinstance(adata_sp.X, np.ndarray):
        G = np.array(adata_sp[:, training_genes].X, dtype="float32")
    else:
        X_type = type(adata_sp.X)
        logging.error("AnnData X has unrecognized type: {}".format(X_type))
        raise NotImplementedError

    if not S.any(axis=0).all() or not G.any(axis=0).all():
        raise ValueError("Genes with all zero values detected. Run `pp_adatas()`.")

    d_str = density_prior
    d = None
    d_source = None

    if mode == "cells":
        d = np.array(adata_sp.obs["uniform_density"])
    elif mode == "clusters":
        d_source = np.array(adata_sc.obs["cluster_label"])

    # Choose device
    device = torch.device(device)  # for gpu

    if verbose:
        print_each = 100
    else:
        print_each = None

    if mode in ["cells", "clusters"]:
        hyperparameters = {
            "lambda_a": lambda_a,  # G term
            "lambda_d": lambda_d,  # density term
            "lambda_r": lambda_r,  # res term
            "d": d, # density prior
            "d_source": d_source, # density prior for source
        }

        logging.info(
            "Begin training with {} genes...".format(
                len(training_genes)
            )
        )
        mapper = mo.Mapper(
            S=S, G=G, device=device, random_state=random_state, **hyperparameters,
        )

        # TODO `train` should return the loss function

        mapping_matrix, training_history = mapper.train(
            learning_rate=learning_rate, num_epochs=num_epochs, print_each=print_each,
        )

    logging.info("Saving results..")
    adata_map = sc.AnnData(
        X=mapping_matrix,
        obs=adata_sc[:, training_genes].obs.copy(),
        var=adata_sp[:, training_genes].obs.copy(),
    )

    # Annotate cosine similarity of each training gene
    G_predicted = adata_map.X.T @ S


    adata_map.uns["training_history"] = training_history

    return adata_map
