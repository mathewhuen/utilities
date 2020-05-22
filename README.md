# utilities
General functions that I use across projects. May overlap with my other repositories (pytorch_light, rbf, etc)

### utils.py
sig_fig - function for formatting floats to a certain number of significant digits

status_bar - class for progress bars (for loops)

path_dist - function for converting nx all_pairs_shortest_path_length generator to a distance matrix (will update later for other path length generators

### utils_pytorch.py
row_tile - simple function for tiling first dimension of a tensor (will update later for multidimensional tiling)

row_dist - function for calculating a distance matrix for all pairs of rows of a tensor (somewhat inefficient)

SDC - Depthwise Separable Convolution Module (MobileNetV2)

IR - Inverted Residual Module (MobileNetV2)

RBF - An RBF Module

gaussian - A gaussian kernel for the above RBF module

inverse_quadratic - An inverse quadratic kernel for the above RBF module

linear - A linear kernel for the above RBF module
