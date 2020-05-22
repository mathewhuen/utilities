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
