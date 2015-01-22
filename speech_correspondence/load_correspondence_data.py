"""
Functions for loading data.

Author: Herman Kamper
Contact: h.kamper@sms.ed.ac.uk
Date: 2014
"""

from numpy.lib.stride_tricks import as_strided
from pylearn2.datasets.dense_design_matrix import DefaultViewConverter, DenseDesignMatrix
import numpy as np


def stack_overlapping_vectors(mat, n_frames, n_rate=1):
    """
    Sweep a window of `n_frames` across the row vectors of `mat` and stack the
    flattened result, taking a window every `n_rate`.

    Return
    ------
    stacked_mat : array
        The stacked flattened vectors; the matrix still use the original `mat`
        as buffer with updated strides and shape.
    """
    # Loosly based on http://wiki.scipy.org/Cookbook/SegmentAxis?action=AttachF
    # ile&do=get&target=segmentaxis.py
    strides = (n_rate*mat.strides[0], mat.strides[1])
    n = 1 + (mat.shape[0] - n_frames) // n_rate
    newshape = (n, mat.shape[1]*n_frames)
    return as_strided(mat, newshape, strides)


def load_data(npy_fn, start=0, stop=None, strip_dims=None, stack_n_frames=1):
    """
    Load the data from `npy_fn` and keep the rows from `start` (inclusive) to
    `stop` (exclusive).

    Parameters
    ----------
    npy_fn : str
    start : int
    stop : int
        Useful for only using a part of the dataset. For data with a frame
        every 10 ms, 360000 frames would give 1 hour of data.
    strip_dims : int
        Only keep this many dimensions of each row (useful for stripping off
        deltas).
    stack_n_frames : None
        If given, treat this many frames as a window and sweep the window
        across the data (1-frame shift).

    Return
    ------
    ddm : DenseDesignMatrix
    """

    X = np.load(npy_fn)
    X = X[start:stop, :strip_dims]

    d_frame = X.shape[1]  # single frame dimension

    # Stack frames
    if stack_n_frames != 1:
        X = stack_overlapping_vectors(X, stack_n_frames, n_rate=1)

    view_converter = DefaultViewConverter((d_frame, X.shape[1]/d_frame, 1))

    return DenseDesignMatrix(X=X, view_converter=view_converter)


def load_xy_data(npy_fn_x, npy_fn_y, start=0, stop=None, strip_dims=None,
        reverse=False):
    """
    Load the data from `npy_fn_x` and `npy_fn_y`, pair them, and keep
    the rows from `start` (inclusive) to `stop` (exclusive).

    Parameters
    ----------
    npy_fn_x : str
    npy_fn_y : str
    start : int
    stop : int
        Useful for only using a part of the dataset. For data with a frame
        every 10 ms, 360000 frames would give 1 hour of data.
    strip_dims : int
        Only keep this many dimensions of each row (useful for stripping off
        deltas).
    reverse : bool
        If set, load the data by first treating `npy_fn_x` as input and
        `npy_fn_y` as output, and then the reverse.

    Return
    ------
    ddm : DenseDesignMatrix
    """

    X = np.load(npy_fn_x)
    X = X[start:stop, :strip_dims]

    Y = np.load(npy_fn_y)
    Y = Y[start:stop, :strip_dims]

    d_frame = X.shape[1]  # single frame dimension

    view_converter = DefaultViewConverter((d_frame, X.shape[1]/d_frame, 1))

    if not reverse:
        return DenseDesignMatrix(X=X, y=Y, view_converter=view_converter)
    else:
        return DenseDesignMatrix(X=np.vstack([X, Y]), y=np.vstack([Y, X]))
