# AND (default)
./perceptron_cuda

# Explicit gates
./perceptron_cuda --and
./perceptron_cuda --or
./perceptron_cuda --nand

# Tweak epochs / learning rate
./perceptron_cuda --or --epochs 50 --lr 0.2

