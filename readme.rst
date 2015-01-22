==================================================
Correspondence and Autoencoder Networks for Speech
==================================================


Collaborators
=============
- Herman Kamper
- Micha Elsner
- Sharon Goldwater
- Aren Jansen
- Daniel Renshaw



Typical steps
=============

1.  Put all the speech data for pretraining into a Numpy array and save in .npy
    format. In the code below, this matrix is specified using the
    dataset_npy_fn parameter.

2.  Pretrain stacked AE using the raw speech data::

        ./speech_correspondence/train_stacked_dae.py

    The parameter_dict dictionary is used to set the model parameters. This
    pretrains a model and saves the output.

3.  The next step is to put matching frames from word instances into two Numpy
    arrays and save these separately in .npy format. Every row in the first
    matrix should match with the corresponding row in the second matrix. In the
    code below these are specified with the dataset_npy_fn_x and
    dataset_npy_fn_y parameters.

4.  Train the correspondence AE::

        ./speech_correspondence/train_correspondence_ae.py

    The values in parameter_dict determines which pretrained model is used to
    initialize the network weights.

5.  Finally, test data can be encoded using ./speech_correspondence/encode.py.
    Run this program without any command line parameters to see its options.
