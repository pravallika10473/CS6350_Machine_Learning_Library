#!/bin/sh

echo "Income>50K probability predictions using decissionTree "
python3 incomePredictionDecissionTree.py

echo "Income>50K probability predictions using Random Forest "
python3 incomePredictionRandomForest.py