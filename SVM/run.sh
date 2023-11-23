#!/bin/sh

echo "Result of 2a"
python svm_primal.py

echo "Result of 2b"
python svm_primal2.py

echo "Result of 2c"
python svm_primal_difference.py

echo "Result of 3a"
python svm_dual.py

echo "Result of 3b"
python svm_dual_kernel.py

echo "Result of 3c"
python svm_support_vectors.py

echo "Result of 3d"
python svm_kernel_perceptron.py