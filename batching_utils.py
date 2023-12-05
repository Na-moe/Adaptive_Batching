import numpy as np

from server import Server


def batch_cost(A:np.ndarray, server:Server, Batch:np.ndarray):
    """
    The cost of applying Batch on A at server.
    """
    f_eta = server.f_eta
    C = server.C

    idx = 0
    BN = Batch.shape[0]
    Cost = [0]
    last_start = 0
    for n in range(1, BN):
        bs = Batch[n]
        last_bs = Batch[n-1]
        last_comp_cost = last_bs * f_eta(last_bs) * C
        start = max(A[idx+bs], last_start + last_comp_cost)
        last_start = start
        idx += bs
        cost = start + bs * f_eta(bs) * C
        Cost.append(cost)
    return np.array(Cost)