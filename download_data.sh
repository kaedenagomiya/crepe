#!/bin/bash

DATA_DIR='./data'

# for audio dataset =====================
#[list of MIR datasets](https://gist.github.com/alexanderlerch/e3516bffc08ea77b429c419051ab793a)
# VocalSet:https://zenodo.org/records/1442513#.W7OaFBNKjx4
# VocalNotes:https://zenodo.org/records/10065955

# for CREPE ==============================
# [x][MIR-1k](http://mirlab.org/dataset/public/)
wget -P ${DATA_DIR} http://mirlab.org/dataset/public/MIR-1K.zip
unzip ${DATA_DIR}/MIR-1K.zip -d ${DATA_DIR}

# [ ][Nsynth](https://magenta.tensorflow.org/datasets/nsynth)
# for Train: A training set with 289,205 examples. Instruments do not overlap with valid or test.
# wget -P ./data http://download.magenta.tensorflow.org/datasets/nsynth/nsynth-train.jsonwav.tar.gz
# for Valid: A validation set with 12,678 examples. Instruments do not overlap with train.
# wget -P ./data http://download.magenta.tensorflow.org/datasets/nsynth/nsynth-valid.jsonwav.tar.gz
# for Test: A test set with 4,096 examples. Instruments do not overlap with train.
# wget -P ./data http://download.magenta.tensorflow.org/datasets/nsynth/nsynth-test.jsonwav.tar.gz

