#!/usr/bin/env python

"""
Encode a given test set using a specified model.

Authors: Micha Elsner, Herman Kamper
Contact: h.kamper@sms.ed.ac.uk
Date: 2014-2015
"""

from os import path
from pylearn2.utils import serial
from theano import function
import argparse
import pylearn2.models.autoencoder
import pylearn2.models.mlp
import sys
import theano
import theano.tensor as T

theano.gof.compilelock.set_lock_status(False)

output_dir = "encoded/"


#-----------------------------------------------------------------------------#
#                              UTILITY FUNCTIONS                              #
#-----------------------------------------------------------------------------#

def check_argv():
    """Check the command line arguments."""
    parser = argparse.ArgumentParser(description=__doc__.strip().split("\n")[0], add_help=False)
    parser.add_argument("input_fn", help="test data to encode in .npz format")
    parser.add_argument("model_fn", help="model to use for encoding in .pkl format")
    parser.add_argument(
        "--strip_dims", default=None, type=int,
        help="only keep this many dimensions of each row (useful for stripping off deltas) "
        "(default: %(default)s)"
        )
    parser.add_argument(
        "--use_layer", default=None, type=int,
        help="layer of an MLP model to use as the encoding (default is last)"
        )
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)
    return parser.parse_args()


#-----------------------------------------------------------------------------#
#                                MAIN FUNCTION                                #
#-----------------------------------------------------------------------------#

def main():
    args = check_argv()

    model = serial.load(args.model_fn)

    print "Constructing model"
    if model.__class__ == pylearn2.models.mlp.MLP:
        # This is a correspondence model
        print "Loaded:", args.model_fn
        if args.use_layer is not None:
            use_layer = args.use_layer
            print "Using encoding from layer", use_layer, "out of", len(model.layers)
        else:
            print "Using last layer out of", len(model.layers)
            use_layer = -1
        dAEs = [l.layer_content for l in model.layers[:use_layer]]
    else :
        # This is a normal stacked dAE: get the other layers from filename

        assert args.use_layer is None, "layer already specified in filename"

        model_dir, basename = path.split(args.model_fn)
        use_layer = int(basename.split(".")[-2].replace("layer", ""))

        # if use_layer != 0:
            # This is not single-layer model
        dAEs = []
        for layer in range(use_layer + 1):
            model_fn = path.join(
                model_dir, ".".join(basename.split(".")[:-2]) + ".layer" + str(layer) + ".pkl"
                )
            print "Loading:", model_fn
            dAEs.append(serial.load(model_fn))
    model = pylearn2.models.autoencoder.DeepComposedAutoencoder(dAEs)

    input_dataset = dict(serial.load(args.input_fn))

    # Symbolic matrix of items to encode
    x = T.dmatrix('x')
    encoder = model.encode(x)
    encode_func = function([x], encoder)

    # Perform encoding
    print "Performing encoding"
    result = {}
    for (label, features) in input_dataset.items():
        result[label] = encode_func(features)

    # Write encoded output
    input_basename = path.splitext(path.split(args.input_fn)[-1])[0]
    model_dir, model_basename = path.split(args.model_fn)
    model_basename = path.splitext(model_basename)[0]
    model_basename = path.split(model_dir)[-1] + "." + model_basename
    encoded_fn = path.join(
        output_dir, 
        "encoded." + input_basename + "." + model_basename + 
        (".layer" + str(args.use_layer) if args.use_layer is not None else "") + ".npz"
        )
    print "Writing encoding:", encoded_fn
    np.savez(encoded_fn, **result)


if __name__ == "__main__":
    main()
