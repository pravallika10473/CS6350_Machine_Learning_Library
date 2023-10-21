#!/bin/sh

echo "Result of batch_gradienet_descent"
python3 batch_gradient_descent.py

echo "Results for stochastic_gradient_descent"
python3 stochastic_gradient_descent.py

echo "Results for optimalweight"
python3 optimalweight.py
