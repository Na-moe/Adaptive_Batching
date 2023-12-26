import numpy as np

from math import floor, ceil
from server import Server
from batching_utils import batch_cost
from arrival import Arrival


def adaptive_batching(A:np.ndarray, server:Server):
    """
    Partition arrivals into batches offline to minimize the final completion time.

    Parameters
    ----------
    A: array_like
        Arrivals of tasks. A[0] = 0, A[1] is the first arrival.
    server: Server
        The server object.

    Returns
    -------
    Start: array_like
        Start index of arrivals of each batch.
    Batch: array_like
        Batch size of each batch.
    Cost: array_like
        Cumsum cost of batches.
    """
    # initation
    if type(A[0]) is Arrival:
        A = np.array([a.time for a in A])

    f_eta = server.f_eta
    max_bs = server.max_bs
    C = server.C

    CostArr = np.full_like(A, fill_value=np.inf, dtype=np.float64)
    CostArr[0] = 0
    StartArr = np.zeros((A.shape[0], A.shape[0]), dtype=np.float64)
    BatchArr = np.zeros_like(A, dtype=np.int32)
    PathArr = np.zeros_like(A, dtype=np.int32)

    # recursion
    N = A.shape[0] - 1
    for n in range(1, N+1):
        # select min cost of <1, ..., n-bs, | ... n> for bs in (1, n)
        bs_max = min(n, max_bs)
        bs = np.arange(1, bs_max+1)
        bbs = BatchArr[n-bs]
        An = np.full_like(bs, fill_value=A[n], dtype=np.float64)
        StartArr[n, bs] = np.maximum(An, StartArr[n-bs, bbs] + bbs*f_eta(bbs)*C)
        waiting_cost = StartArr[n, bs] - (StartArr[n-bs, bbs] + bbs*f_eta(bbs)*C)
        bs_cost = CostArr[n-bs] + waiting_cost + bs*f_eta(bs)*C

        bs_index = np.argmin(bs_cost)
        min_cost = bs_cost[bs_index]
        min_bs = bs[bs_index]

        CostArr[n] = min_cost
        BatchArr[n] = min_bs
        PathArr[n] = n - min_bs

    Start = [N]
    k = PathArr[N]
    while k != 0:
        Start.append(k)
        k = PathArr[k]
    Start.append(0)
    Start = np.flip(np.array(Start))
    Batch = BatchArr[Start]
    Cost = CostArr[Start]
    Start = Start + 1
    return Start, Batch, Cost

def adaptive_batching_with_batch(A, server, Batching_Res):
    if type(A[0]) is Arrival:
        A = np.array([a.time for a in A])

    f_eta = server.f_eta
    C = server.C
    
    Start, Batch, Cost = Batching_Res
    num_consumed = np.sum(Batch)
    A_new = A[num_consumed+1:].copy()
    last_end, last_batch = Start[-1]-1, Batch[-1]
    
    available_time = A[last_end] + last_batch * f_eta(last_batch) * C
    not_available = A_new < available_time
    A_new[not_available] = available_time
    A_new = np.concatenate([[0], A_new])

    Start_new, Batch_new, Cost_new = adaptive_batching(A_new, server)
    Start_new += num_consumed
    Start = np.concatenate([Start, Start_new[1:]])
    Batch = np.concatenate([Batch, Batch_new[1:]])
    Cost = np.concatenate([Cost, Cost_new[1:]])
    return Start, Batch, Cost

def window_adaptive_batching(A, server, ws):
    if type(A[0]) is Arrival:
        A = np.array([a.time for a in A])

    NA = A.shape[0]
    if NA-1 <= ws+1:
        return adaptive_batching(A, server)
    
    I = np.mean(np.diff(A[:ws+1]))
    A_pred = np.concatenate([A[:ws+1], A[ws] + np.arange(1, NA-ws) * I])
    Start, Batch, Cost = adaptive_batching(A_pred, server)
    Cost = batch_cost(A, server, Batch)
    End = np.cumsum(Batch)
    batch_in_window = End < ws
    last_end = End[batch_in_window][-1]

    while last_end < NA-1:
        res = (Start[batch_in_window], Batch[batch_in_window], Cost[batch_in_window])
        
        if last_end+ws+1 >= NA:
            Start, Batch, Cost = adaptive_batching_with_batch(A, server, res)
            return Start, Batch, Cost
        
        I = np.mean(np.diff(A[last_end+1:last_end+ws+1]))
        A_pred = np.concatenate([A[:last_end+ws+1], A[last_end+ws] + np.arange(1, NA-last_end-ws) * I])
        Start, Batch, Cost = adaptive_batching_with_batch(A_pred, server, res)
        Cost = batch_cost(A, server, Batch)
        End = np.cumsum(Batch)
        batch_in_window = End < last_end + ws
        prev_last_end = last_end
        last_end = End[batch_in_window][-1]

        if prev_last_end == last_end:
            batch_in_window[np.sum(batch_in_window)] = True
            last_end = End[batch_in_window][-1]

    return Start, Batch, Cost

def static_batching(A:np.ndarray, server:Server, bs:int or np.ndarray):
    def static_batching_int(A, server, bs):
        N = A.shape[0] - 1
        nb = N // bs if N % bs == 0 else N // bs + 1 # num_batch
        Batch = np.ones(nb+1, dtype=np.int32) * bs
        Batch[0] = 0
        Batch[-1] = bs if N % bs == 0 else N % bs
        return Batch, batch_cost(A, server, Batch)
    
    def static_batching_array(A, server, bs):
        N = A.shape[0]-1
        assert isinstance(bs, np.ndarray)
        nb = np.ones_like(bs, dtype=np.int32) * (N // bs + 1) # num_batch
        nb[N % bs == 0] -= 1
        Cost = []
        for n, b in zip(nb, bs):
            Batch = np.ones(n+1, dtype=np.int32) * b
            Batch[0] = 0
            Batch[-1] = b if N % b == 0 else N % b
            cost = batch_cost(A, server, Batch)[-1]
            Cost.append(cost)
        return np.array(Cost)
    
    if type(A[0]) is Arrival:
        A = np.array([a.time for a in A])
    
    if type(bs) is int:
        return static_batching_int(A, server, bs)
    elif type(bs) is np.ndarray:
        return static_batching_array(A, server, bs)
    else:
        raise NotImplementedError('bs should be int or np.ndarray')

def opportunistic_batching(A, server, threshold):
    """
    This will batch arrivals within threshold from the first arrvial.
    """
    if type(A[0]) is Arrival:
        A = np.array([a.time for a in A])

    f_eta = server.f_eta
    C = server.C
    max_bs = server.max_bs

    AN = A.shape[0]
    if AN == 1: # No Arrivals
        return np.array([threshold])
    
    batch, Batch = 1, [0]
    cost, Cost = 0, [0]
    idx = 0
    while idx < AN-1:
        batch_in_threshold = (A[idx] <= A) & (A <= A[idx] + threshold)
        batch_in_threshold = np.sum(batch_in_threshold)
        batch = min(batch_in_threshold, max_bs, AN-1-idx)
        if batch == batch_in_threshold:
            start = max(cost, A[idx]+threshold)
        else:
            start = max(A[idx+batch], cost)
        idx += batch
        Batch.append(batch)
        cost = start + batch * f_eta(batch) * C
        Cost.append(cost)
        
    return np.array(Batch), np.array(Cost)


def aimd_batching(A, server, add_bs=4, ratio=0.9, latency=100):
    """
    Additive-Increase, Multiplicative-Decrease (AIMD) Batching.
    Proposed in
        Clipper: A Low-Latency Online Prediction Serving System
        https://arxiv.org/abs/1612.03079
    """
    if type(A[0]) is Arrival:
        A = np.array([a.time for a in A])

    f_eta = server.f_eta
    C = server.C
    max_bs = server.max_bs

    AN = A.shape[0]
    if AN == 1:
        return np.array([0])
    if AN == 2:
        return np.array([A[1] + C])

    bs, Batch = add_bs, [0]
    opt_bs_flag = False
    cost, Cost = 0, [0]
    idx = 0
    while idx < AN-1:
        if idx + bs > AN-1:
            bs = AN-1 - idx
        start = max(A[idx+bs], cost)
        idx += bs
        Batch.append(bs)
        cost = start + bs * f_eta(bs) * C
        Cost.append(cost)
        if opt_bs_flag:
            continue
        if bs * f_eta(bs) * C < latency:
            bs += add_bs
            bs = min(bs, max_bs)
        else:
            opt_bs_flag = True
            bs = int(bs * ratio)

    return np.array(Batch), np.array(Cost)
        
def sot_batching(A, server):
    """
    Stochastic Optimal Task (SOT) Batching.
    Proposed in 
        EdgeBatch: Towards AI-empowered Optimal Task Batching in Intelligent Edge Systems
        https://ieeexplore.ieee.org/document/9052125
    """
    if type(A[0]) is Arrival:
        A = np.array([a.time for a in A])
    f_eta = server.f_eta
    C = server.C
    max_bs = server.max_bs

    Weights = np.full(max_bs, fill_value=1/max_bs, dtype=np.float64)
    Cdf = np.cumsum(Weights)
    Etas = np.full(max_bs, fill_value=0.5, dtype=np.float64)

    AN = A.shape[0]
    if AN == 1:
        return np.array([0])
    if AN == 2:
        return np.array([A[1] + C])

    rand_seed = 0
    np.random.seed(rand_seed)
    bs = np.searchsorted(Cdf, np.random.rand()) + 1
    Batch = [0]
    cost, Cost = 0, [0]
    idx = 0
    Regrets_t = []
    while idx < AN-1:
        if idx + bs > AN-1:
            bs = AN-1 - idx
        start = max(A[idx+bs], cost)
        idx += bs
        Batch.append(bs)
        cost = start + bs * f_eta(bs) * C
        Cost.append(cost)

        Props = Etas * Weights / np.sum(Etas * Weights)
        Regret = cost - adaptive_batching(A[:idx+1], server)[-1][-1]
        Regrets = np.zeros_like(Props)
        Regrets[bs-1] = Regret
        Regrets_t.append(Regrets ** 2)
        Regrets_t_arr = np.array(Regrets_t)
        last_Etas = Etas
        Etas = np.minimum(0.5, np.sqrt(np.log(max_bs) / 1+np.sum(Regrets_t_arr)))
        Weights = np.power(Weights * (1+Etas*Regrets), Etas/last_Etas)
        Cdf = np.cumsum(Weights)

        rand_seed += 1
        np.random.seed(rand_seed)
        bs = np.searchsorted(Cdf, np.random.rand()) + 1

    return np.array(Batch), np.array(Cost)#, Weights

def split_shift_batching(A, server, delta):
    def linear_fitting(server):
        f_eta = server.f_eta
        C = server.C
        max_bs = server.max_bs

        bs = np.arange(1, max_bs+1)
        latency = f_eta(bs) * bs * C
        alpha, beta = np.polyfit(bs, latency, 1)        
        return alpha, beta
    
    if type(A[0]) is Arrival:
        A = np.array([a.time for a in A])
    
    alpha, beta = linear_fitting(server)
    num_tasks = A.shape[0] - 1
    gamma_ = A[-1] / num_tasks
    gamma = delta * gamma_ / (delta - gamma_)

    num_local_s = floor((delta * num_tasks - alpha) / (beta + delta))

    num_batches = 1
    Batch = []
    if num_local_s * gamma < alpha:
        num_local = floor((delta * num_tasks - alpha) / (beta + delta + gamma))
        Batch.append(num_local)
    else:
        num_local_1 = ceil((num_local_s * gamma - alpha) / (beta + gamma))
        while num_local_1 * gamma < alpha:
            num_batches += 1
            num_local_now = ceil((num_local_1 * gamma - alpha) / (beta + gamma))
            num_local_b = num_local_1 - num_local_now
            Batch.append(num_local_b)
            num_local_1 = num_local_now
        else:
            Batch.append(num_local_1)
    Batch = reversed(Batch)
    num_local_1 = Batch[1]
    num_local = floor((delta * num_tasks - alpha * num_batches - num_local_1 * gamma) / (beta + delta))
    while sum(Batch[:num_batches]) - num_local >= Batch[num_batches-1]:
        num_batches -= 1
        num_local = floor((delta * num_tasks - alpha * num_batches - num_local_1 * gamma) / (beta + delta))
    else:
        num_local_b = Batch[num_batches-1] + num_local - sum(Batch[:num_batches])
    Batch[num_batches-1] = num_local_b
    Batch = np.concatenate([[0], Batch[:num_batches]])
    Cost = batch_cost(A, server, Batch)
    return Batch, Cost