#!/bin/bash

# Shell script to execute customized model training.
# YAML configurations files are used to control the training pipeline such as datasets, models, hyperparameters, etc.

cfgfiles=`ls ./cfgs/tune/*.yaml`
for cfgfile in $cfgfiles
do
   echo "$(basename "$cfgfile")"
   FILENAME="$(basename "$cfgfile")"
   python ./src/train_model.py --dir ./cfgs/tune/ --name $FILENAME
done