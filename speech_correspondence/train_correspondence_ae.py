#!/usr/bin/env python

"""
Train a correspondence AE neural network.

Authors: Micha Elsner, Herman Kamper
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


#-----------------------------------------------------------------------------#
#                                   SETTINGS                                  #
#-----------------------------------------------------------------------------#

parameter_dict = {
    "dataset_npy_fn_x": "../../data/input/mfcc_train_htk_word1_frames.std80k.npy",
    "dataset_npy_fn_y": "../../data/input/mfcc_train_htk_word2_frames.std80k.npy",
    "models_basedir": "models/",
    "dim_input": 39,
    "layer_spec_str": "[100] * 9",
    "corruption": 0,
    "max_epochs": 120,
    "batch_size": 256,
    "learning_rate": 0.004,
    "start_from_scratch": False,  # do not initialize from other model, but start from scratch
    "reverse": False,  # do pairs both ways
    }


#-----------------------------------------------------------------------------#
#              SETUP DIRECTORIES, FILENAMES AND YAML DICTIONARIES             #
#-----------------------------------------------------------------------------#

# Parse some settings
layer_spec = eval(parameter_dict["layer_spec_str"])

# Output directory
model_dir = re.sub("[\ \[\]]", "", parameter_dict["layer_spec_str"]).replace("*", "x").replace(",", "-")
if not parameter_dict["start_from_scratch"]:
    for var in sorted(["corruption", "batch_size"]):
        model_dir += "." + var + str(parameter_dict[var])

# Output filename
run_id = "correspondence_ae"
run_id += "." + path.splitext(path.split(parameter_dict["dataset_npy_fn_x"])[-1])[0].replace("_word1", "")
for var in sorted(["max_epochs"]):
    run_id += "." + var + str(parameter_dict[var])

# Correspondence parameter dict
correspondence_ae_parameter_dict = {
    "dataset_npy_fn_x": parameter_dict["dataset_npy_fn_x"],
    "dataset_npy_fn_y": parameter_dict["dataset_npy_fn_y"],
    "reverse": parameter_dict["reverse"],
    "nvis": parameter_dict["dim_input"],
    "batch_size": parameter_dict["batch_size"],
    "max_epochs": parameter_dict["max_epochs"],
    "learning_rate": parameter_dict["learning_rate"],
    "save_path": parameter_dict["models_basedir"] + model_dir,
    "run_id": run_id,
    }
load_dae_basename = path.join(correspondence_ae_parameter_dict["save_path"], "dae")

# Filenames
dae_yaml_fn = "yaml/dae_layer.yaml"
mlp_pretrained_yaml_fn = "yaml/mlp_pretrained.yaml"
mlp_linear_yaml_fn = "yaml/mlp_linear.yaml"
correspondence_ae_yaml_fn = "yaml/correspondence_ae.yaml"

# Write parameters to file
if not path.isdir(correspondence_ae_parameter_dict["save_path"]):
    os.mkdir(correspondence_ae_parameter_dict["save_path"])
open(
    path.join(correspondence_ae_parameter_dict["save_path"],
    "train_correspondence_ae.parameters.dict"), "w").write(str(parameter_dict) + "\n"
    )

#-----------------------------------------------------------------------------#
#                           TRAIN CORRESPONDENCE AE                           #
#-----------------------------------------------------------------------------#

# Create the encoding layers string
layer_strs = []
for i_layer, nhid in enumerate(layer_spec):
    if not parameter_dict["start_from_scratch"]:
        mlp_pretrained_yaml = open(mlp_pretrained_yaml_fn).read()
        layer_content = "!pkl: \"" + load_dae_basename + ".layer" + str(i_layer) + ".pkl\""
        mlp_pretrained_yaml = mlp_pretrained_yaml % {
            "layer_name":  "'layer" + str(i_layer) + "'",
            "layer_content": layer_content
            }
        layer_strs.append(mlp_pretrained_yaml)
    else:
        if i_layer == 0:
            nvis = parameter_dict["dim_input"]
        else:
            nvis = layer_spec[i_layer - 1]
        mlp_pretrained_yaml = open(mlp_pretrained_yaml_fn).read()
        dae_yaml = open(dae_yaml_fn).read()
        print dae_yaml 
        layer_content = dae_yaml % {
            "nvis": nvis,
            "nhid": nhid,
            "corruption_level": corruption_level,  # this isn't actually used
            "tied_weights": tied_weights
            }
        mlp_pretrained_yaml = mlp_pretrained_yaml % {
            "layer_name":  "'layer" + str(i_layer) + "'",
            "layer_content": layer_content
            }
        layer_strs.append(mlp_pretrained_yaml)

# Create output layer string
mlp_linear_yaml = open(mlp_linear_yaml_fn).read()
mlp_linear_yaml = mlp_linear_yaml % {
    "layer_name": "layer" + str(len(layer_spec)), "dims": parameter_dict["dim_input"]
    }
layer_strs.append(mlp_linear_yaml)

# Combine everything to create correspondence AE YAML
correspondence_ae_yaml = open(correspondence_ae_yaml_fn).read()
correspondence_ae_parameter_dict["layer_str"] = ",\n".join(layer_strs)
correspondence_ae_yaml = correspondence_ae_yaml % correspondence_ae_parameter_dict

print datetime.datetime.now()
print "Training correspondence AE"
print "YAML file:"
print correspondence_ae_yaml

train = yaml_parse.load(correspondence_ae_yaml)
train.main_loop()

print datetime.datetime.now()
print "Model written to:", train.save_path
yaml_output_fn = path.splitext(train.save_path)[0] + ".yaml"
open(yaml_output_fn, "w").writelines(correspondence_ae_yaml)
print "YAML written to:", yaml_output_fn
