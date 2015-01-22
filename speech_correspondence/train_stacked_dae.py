#!/usr/bin/env python

"""
Train a stacked denoising AE.

Author: Micha Elsner, Herman Kamper
Contact: h.kamper@sms.ed.ac.uk
Date: 2014-2015
"""

from os import path
from pylearn2.config import yaml_parse
import cPickle as pickle
import datetime
import numpy as np
import os
import re
import sys
import theano

theano.gof.compilelock.set_lock_status(False)

TRANSFORMER = "!obj:pylearn2.datasets.transformer_dataset.TransformerDataset"


#-----------------------------------------------------------------------------#
#                                   SETTINGS                                  #
#-----------------------------------------------------------------------------#

# Network parameter settings
parameter_dict = {
    "dataset_npy_fn": "../../data/mfcc_pretrain_htk_update.npy",
    "dim_input": 39,
    "layer_spec_str": "[100] * 9",
    "corruption": 0,
    "max_epochs": 30,
    "batch_size": 256,
    "learning_rate": 0.00025,
    }


#-----------------------------------------------------------------------------#
#              SETUP DIRECTORIES, FILENAMES AND YAML DICTIONARIES             #
#-----------------------------------------------------------------------------#

# Parse some settings
layer_spec = eval(parameter_dict["layer_spec_str"])

# Output directory
model_dir = re.sub("[\ \[\]]", "", parameter_dict["layer_spec_str"]).replace("*", "x").replace(",", "-")
for var in sorted(["corruption", "batch_size"]):
    model_dir += "." + var + str(parameter_dict[var])

# Trainset parameter dict
trainset_yaml_dict = {
    "dataset_npy_fn": parameter_dict["dataset_npy_fn"],
    "strip_dims": parameter_dict["dim_input"],
    "stack_n_frames": 1
    }

# Network parameter dict
dae_yaml_dict = {
    "batch_size": parameter_dict["batch_size"],
    "max_epochs": parameter_dict["max_epochs"],
    "learning_rate": parameter_dict["learning_rate"],
    "save_path": "models/" + model_dir,
    "run_id": "dae",
    "tied_weights": False
    }

# Filenames
trainset_yaml_fn = "yaml/data_trainset.yaml"
dae_yaml_fn = "yaml/dae.yaml"

# Write parameters to file
if not path.isdir(dae_yaml_dict["save_path"]):
    os.mkdir(dae_yaml_dict["save_path"])
open(path.join(dae_yaml_dict["save_path"], "train_stacked_dae.parameters.dict"), "w").write(
    str(parameter_dict) + "\n"
    )


#-----------------------------------------------------------------------------#
#                            LAYER-WISE PRETRAINING                           #
#-----------------------------------------------------------------------------#

for i_layer, nhid in enumerate(layer_spec):

    dae_yaml = open(dae_yaml_fn).read()
    trainset_yaml = open(trainset_yaml_fn).read()

    # Create training string
    trainset_yaml = trainset_yaml % trainset_yaml_dict
    for j_prev_layer in range(i_layer):
        prev_model = path.join(
            dae_yaml_dict["save_path"],
            dae_yaml_dict["run_id"] + ".layer%d.pkl" % j_prev_layer
            )
        trainset_yaml = (
            TRANSFORMER + " {\nraw: %s,\ntransformer: !pkl: \"%s\" \n}" % (trainset_yaml, prev_model)
            )

    # Setup parameters for this layer
    if i_layer == 0:
        nvis = parameter_dict["dim_input"]
    else:
        nvis = layer_spec[i_layer - 1]

    if parameter_dict["corruption"] == 0:
        dae_yaml_dict["corruptor"] = (
            "!obj:pylearn2.corruption.BinomialCorruptor {corruption_level: 0}"
            )
    else:
        dae_yaml_dict["corruptor"] = (
            "!obj:pylearn2.corruption.GaussianCorruptor {stdev: %(corruption)f}"
            % {"corruption": parameter_dict["corruption"]}
            )

    dae_yaml_dict["trainset"] = trainset_yaml
    dae_yaml_dict["nvis"] = nvis
    dae_yaml_dict["nhid"] = nhid
    dae_yaml_dict["layer"] = i_layer
    dae_yaml = dae_yaml % dae_yaml_dict

    print datetime.datetime.now()
    print "Training layer", i_layer
    print "YAML file:"
    print dae_yaml

    train = yaml_parse.load(dae_yaml)
    train.main_loop()

    print datetime.datetime.now()
    print "Model written to:", train.save_path
    dae_yaml_output_fn = path.splitext(train.save_path)[0] + ".yaml"
    open(dae_yaml_output_fn, "w").writelines(dae_yaml)
    print "YAML written to:", dae_yaml_output_fn
