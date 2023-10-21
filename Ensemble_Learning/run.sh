#!/bin/sh

echo "Results for Adaboost"
python3 Adaboostbank.py

echo "Results for bagged tree"
python3 baggedtree.py

echo "Results for biasvariance"
python3 biasvariance.py

echo "Results for randomforest"
python3 randomforest.py
