import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import itertools

def row_tile(tens, times):
    inds = list([i]*times for i in range(tens.shape[0]))
    inds = list(itertools.chain.from_iterable(inds))
    return tens[inds]
    
def row_dist(tens, p=2):
    n = tens.shape[0]
    tens_1 = tens.repeat(n,1)
    tens_2 = row_tile(tens, n)
    out = (tens_1 - tens_2).abs().pow(p).sum(1).pow(1/p).reshape(n,n)
    return out




#separable depth-wise convolution (MobileNetV2)
class SDC(nn.Module):
    def __init__(self, D_in, D_out, kernel_size=3, stride=1):
        """
        """
        super(SDC, self).__init__()
        self.separable_layer = nn.Sequential()
        self.separable_layer.add_module('sepdepconv_1', nn.Conv2d(D_in, D_in, kernel_size=kernel_size, padding=1, stride=stride, groups=D_in))
        self.separable_layer.add_module('sepdepconv_2', nn.Conv2d(D_in, D_out, kernel_size=1))

    def forward(self, x):
        """
        """
        y_pred = self.separable_layer(x)
        return y_pred

#inverted residual block
class IR(nn.Module):
    def __init__(self, input_size, output_size, kernel_size=3, stride=1, expansion=6, residual='sum'):#,  layers):
        """
        residual types: add, concat, convolv.  if concat, then output_size refers to the output of the 'embedding' layer + input_size
        """
        super(IR, self).__init__()
        expanded_size = int(expansion*input_size)
        self.inverted_residual = nn.Sequential()
        self.inverted_residual.add_module('expansion', nn.Conv2d(input_size, expanded_size, kernel_size=1))
        self.inverted_residual.add_module('relu6_expansion', nn.ReLU6())
        self.inverted_residual.add_module('spatial_convolution', SDC(expanded_size, expanded_size, kernel_size=kernel_size, stride=stride))
        self.inverted_residual.add_module('relu6_spatial', nn.ReLU6())
        out_size = output_size
        if(residual=='sum'):
            assert output_size>=input_size, 'output_size must be greater than or equal to input_size for residual="sum"'#'Support for summing accross mismatched input and outputs is not yet implemented. Please make sure that input_size = output_size.'
        elif(residual=='concat'):
            assert output_size>(input_size+1), 'output_size must be larger than input_size to allow concatenation.'
            out_size -= input_size
        self.inverted_residual.add_module('embedding', nn.Conv2d(expanded_size, out_size, kernel_size=1))
        
        self.conv = nn.Sequential()
        self.conv.add_module('conv', nn.Conv2d(input_size+output_size, output_size, kernel_size=1))
        
        self.residual = residual
    def forward(self, x):
        y_pred = self.inverted_residual(x)
        if(self.residual=='sum'):
            y_pred[:, :x.shape[1]] += x
        elif(self.residual=='concat'):
            y_pred = torch.cat([x, y_pred], dim=1)
        elif(self.residual=='conv'):
            y_pred = torch.cat([x, y_pred], dim=1)
            y_pred = self.conv(y_pred)
            
        return y_pred
    
    
class RBF(nn.Module):
    """
    A pytorch RBF module (a more general version of https://github.com/JeremyLinux/PyTorch-Radial-Basis-Function-Layer/blob/master/Torch%20RBF/torch_rbf.py with rejection according to output of rbf layer)
    Accepts any input shape.
    
    """
    def __init__(self, d_in, k, basis_fnc='gaussian', init_args=None):#d_in and d_out must be tuples with first dimension corresponding to the squeezed shape of a single input to batch index
        """
        
        Arguments
        ---------
        d_in: int,tuple
            The dimensions of the input data (excluding the batch dimension).
        
        d_out: int
            the number of outputs (classes).
        
        k: int
            The number of RBF kernels to use.
            
        basis_fnc: str, optional
            The type of kernel to use (phi). Default is gaussian.
            
        init_args:  dict, optional
            A custom dictionary defining initialization for paramters. Default in None.
            
            This dictionary must contain keys for kernel 'centers' and any other parameters used in the kernel function. Each key should map to another dictionary with keys 'fnc' and 'args' corresponding to an initialization function (eg, nn.init.normal_) and a kwargs dict for said initialization function, respectively.
        
        classifiy: bool, optional (currently not implemented and will probably remove since it only makes sense with rbf kernels that scale positively with distance)
            Indicates whether or not a dense layer with softmax activation should be appended to the results of the rbf kernel. Default is True
            
            If classify is set to False, d_out is ignored, and the output is of the shape (batch size, k).
            
        
        Usage
        -----
        There are two recommended uses for this layer:
        1) Shallow RBF Network - accepting either input directly from the raw feature space or a simple model
            A) input -> rbf layer -> dense layer -> softmax
            B) input -> shallow model -> rbf layer -> dense layer -> softmax
            
        2) Deep RBF Network - accepting input from a deep model.
            A) input -> deep model -> rbf layer (k = number of classes) -> softmax
        
        See the associated readme file for more explanation and examples: https://github.com/mathewhuen/RBF
        """
        super(RBF, self).__init__()
        self.d_in = d_in if type(d_in)==tuple else (d_in,)
#         self.d_out = d_out
        self.k = k
#         self.classify = classify
        
        basis_fncs = {
            'gaussian':gaussian,
            'inverse_quadratic':inverse_quadratic,
            'linear':linear
        }
        
        basis_fnc_params = {
            'gaussian' : {
                'gamma':{
                    'fnc':nn.init.constant_,
                    'dims':(self.k,),
                    'args':{'val':1}
                }
            },
            'inverse_quadratic' : {
            },
            'linear' : {
                
            }
        }
        
        
        
        self.basis_fnc = basis_fncs[basis_fnc]
        
        params = {}
        params.update([('centers', nn.Parameter(torch.Tensor(1, *self.d_in, k))), 
                       *[(key, nn.Parameter(torch.Tensor(*(basis_fnc_params[basis_fnc][key]['dims'])))) for key in basis_fnc_params[basis_fnc].keys()]])
        self.params = nn.ParameterDict(params)
        self.threshold = nn.Parameter()
        
        if(init_args==None):
            init_args = {}
            init_args.update([('centers', {'fnc': nn.init.normal_, 'args':{'mean':0, 'std':1}}), 
                       *[(key, {'fnc':basis_fnc_params[basis_fnc][key]['fnc'], 'args':basis_fnc_params[basis_fnc][key]['args']}) for key in basis_fnc_params[basis_fnc].keys()]])
        self._initialize_params(init_args)
        
        self.input_size_error = 'Actual input dimension {ain} does not match specified input dimension {spin} for all dim>=1. Please note that RBF does not yet support variable shaped inputs. If you are working with variable shaped convolutional inputs, consider using global max pooling to force the dimension to the number of feature maps or another dimension forcing technique like Spatial Pyramid Pooling.'

    def _initialize_params(self, kwargs):
        for param in self.params.keys():
            kwargs[param]['fnc'](self.params[param], **(kwargs[param]['args']))
            
    def forward(self, x, p=2):
        if(len(x.shape)==1):
            x = x.reshape(1,x.shape[0])
        if(self.d_in!=x[0].shape):
            error_str = self.input_size_error.format(ain=x.shape, spin=(1, *self.d_in))
            raise TypeError(error_str)
        
        dim_mod = [1]*(len(self.d_in)+1)
        dist = self.params['centers']-x.reshape(*x.shape, 1).repeat(*dim_mod,self.k)#use squeeze and unsqueeze and expand
        dist = dist.pow(p).sum(tuple(range(1,len(dist.shape)-1)))#.pow(1/p)
        r = self.basis_fnc(dist, {key:self.params[key] for key in self.params.keys() if key!='centers'})
        
        return(r)
    
    
    
def gaussian(dist, params, p=2):
    """
    """
    n_obs = dist.shape[0]
    gamma = params['gamma']
    dist = dist * gamma.unsqueeze(0).expand(n_obs, -1)
    phi = torch.exp(-dist)
    return(phi)
    
def inverse_quadratic(dist, params, p=2):
    phi = torch.ones_like(dist) / (torch.ones_like(dist) + dist)
    return phi

def linear(dist, params, p=2):
    phi = dist.pow(1/p)
    return phi
