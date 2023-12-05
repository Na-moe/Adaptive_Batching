import numpy as np

from server import Server
from batching_utils import batch_cost

          
def multiserver_adaptive_batching(A:np.ndarray, Servers:np.ndarray):
    """
    Partition arrivials into batches offline and appoint to appropriate server.

    Parameters
    ----------
    A: array_like
        Arrivals of tasks. A[0] = 0, A[1] is the first arrival.
    Servers: array_like
        Available server list.

    Returns
    -------
    Serv: array_like
        Appointed server of each batch.
    Batch: array_like
        Batch size of each batch.
    Cost: array_like
        Cumsum cost of batches.
    """
    req_num = A.shape[0]
    serv_num = Servers.shape[0]
    # CostArr[n][s1][bs][s2]:
    # the end time of s2, when task n-bs+1:n+1 are completed in server s1
    CostArr = np.zeros([req_num, serv_num, req_num, serv_num], dtype=np.float64)
    CostArr[0,:,:,:] = 0
    # CostMaxArr[n][s][bs] = max(CostArr[n][s][bs][:])
    # the max end time of all servers, when task n with previous bs-1 tasks are completed in server s
    CostMaxArr = np.zeros([req_num, serv_num, req_num], dtype=np.float64)
    CostMaxArr[0,:,:] = 0
    # saving dp path
    # BestPrevBS[n][s][bs] = prev_bs, BestPrevS[n][s][bs] = prev_s
    # means min cost of [n][s][bs] is transferred from [n-bs][prev_s][prev_bs]
    BestPrevBS = np.zeros([req_num, serv_num, req_num], dtype=np.int32)
    BestPrevS = np.zeros([req_num, serv_num, req_num], dtype=np.int32)

    # saving each tasks' best plan to debug and backtrack
    BestS = np.zeros_like(A, dtype=np.int32)
    BestBS = np.zeros_like(A, dtype=np.int32)
    MinCost = np.full_like(A, fill_value=np.inf, dtype=np.float64)

    # DP
    for n in range(1, req_num):
        # appoint task n to server s
        for s in range(0, serv_num):
            # get info of server s
            f_eta = Servers[s].f_eta
            max_bs = min(n, Servers[s].max_bs)
            C = Servers[s].C
            tau = Servers[s].tau
            # solve task n in a batch of bs tasks
            for bs in range(1, max_bs+1):
                task_time = bs*f_eta(bs)*C # computation time of task n with its batch
                best_prev_s = 0
                best_prev_bs = 1
                min_cost_for_best_prev = np.inf
                new_cost_in_server_s = np.inf
                if n-bs == 0: # batch all previous tasks
                    new_cost = A[n] + tau + task_time
                    if new_cost < min_cost_for_best_prev:
                        min_cost_for_best_prev = new_cost
                        best_prev_bs = 0
                        best_prev_s = 0 # does not matter
                        new_cost_in_server_s = A[n] + task_time
                else:
                    # rewrite the following code in vector form for prev_bs
                    for prev_s in range(0, serv_num):
                        prev_max_bs = min(n-bs, Servers[prev_s].max_bs)
                        prev_bs = np.arange(1, prev_max_bs+1)
                        start_time_now = np.maximum(A[n]+tau, CostArr[n-bs][prev_s][prev_bs][:, s]) # server s
                        new_cost = np.maximum(CostMaxArr[n-bs][prev_s][prev_bs], start_time_now + task_time)
                        min_cost_now = np.min(new_cost)
                        if min_cost_now < min_cost_for_best_prev:
                            idx = np.argmin(new_cost)
                            min_cost_for_best_prev = min_cost_now
                            best_prev_bs = prev_bs[idx]
                            best_prev_s = prev_s
                            new_cost_in_server_s = start_time_now[idx] + task_time

                CostArr[n][s][bs] = CostArr[n-bs][best_prev_s][best_prev_bs]
                CostArr[n][s][bs][s] = new_cost_in_server_s
                CostMaxArr[n][s][bs] = min_cost_for_best_prev
                BestPrevBS[n][s][bs] = best_prev_bs
                BestPrevS[n][s][bs] = best_prev_s
                if CostMaxArr[n][s][bs] < MinCost[n]:
                    BestS[n] = s
                    BestBS[n] = bs
                    MinCost[n] = CostMaxArr[n][s][bs]

    # track the best plan
    # each batch give to which server
    ServArr = []
    # size of each batch
    BatchArr = []
    # end time of each batch
    CostArr = []
    n_now = req_num - 1
    bs_now = BestBS[n]
    s_now = BestS[n]
    cost_now = MinCost[n]
    while n_now > 0:
        ServArr.append(s_now)
        BatchArr.append(bs_now)
        CostArr.append(cost_now)
        n_new = n_now - bs_now
        bs_new = BestPrevBS[n_now][s_now][bs_now]
        s_new = BestPrevS[n_now][s_now][bs_now]
        n_now, bs_now, s_now = n_new, bs_new, s_new
        cost_now = CostMaxArr[n_now][s_now][bs_now]
    Serv = np.flip(np.array(ServArr))
    Batch = np.flip(np.array(BatchArr))
    Cost = np.flip(np.array(CostArr))
    # print(CostMaxArr)
    return Serv, Batch, Cost

def multiserver_adaptive_batching_with_costs(A:np.ndarray, Servers:np.ndarray, Costs:np.ndarray, ws:int):
    """
    Partition arrivials into batches offline and appoint to appropriate server.

    Parameters
    ----------
    A: array_like
        Arrivals of tasks. A[0] = 0, A[1] is the first arrival.
    Servers: array_like
        Available server list.
    Costs: array_like
        Optional
        Use when already have tasks before A[0], usually in window adaptive batching
    ws: integer
        Window size.
        Used in window adaptive batching.
        
    Returns
    -------
    Serv: array_like
        Appointed server of each batch.
    Batch: array_like
        Batch size of each batch.
    Cost: array_like
        Cumsum cost of batches.
    WindowFinalBatchCostArray: 
        After executing last batch which have some part inside the window, each servers' costs.
    WindowFinalBatchStartCost:
        Before giving last batch to the executing server, the servers' costs.
    """
    req_num = A.shape[0]
    serv_num = Servers.shape[0]
    # CostArr[n][s1][bs][s2]:
    # the end time of s2, when task n with previous bs-1 tasks are completed in server s1
    CostArr = np.zeros([req_num, serv_num, req_num, serv_num], dtype=np.float64)
    CostArr[0,:,:,:] = 0
    # CostMaxArr[n][s][bs] = max(CostArr[n][s][bs][:])
    # the max end time of all servers, when task n with previous bs-1 tasks are completed in server s
    CostMaxArr = np.zeros([req_num, serv_num, req_num], dtype = np.float64)
    CostMaxArr[0,:,:] = 0
    # saving dp path
    # BestPrevBS[n][s][bs] = prev_bs, BestPrevS[n][s][bs] = prev_s
    # means min cost of [n][s][bs] is transferred from [n-bs][prev_s][prev_bs]
    BestPrevBS = np.zeros([req_num, serv_num, req_num], dtype = np.int32)
    BestPrevS = np.zeros([req_num, serv_num, req_num], dtype = np.int32)

    # saving each tasks' best plan to debug and backtrack
    BestS = np.zeros_like(A, dtype=np.int32)
    BestBS = np.zeros_like(A, dtype=np.int32)
    MinCost = np.full_like(A, fill_value=np.inf, dtype=np.float64)

    # DP
    for n in range(1, req_num):
        # appoint task n to server s
        for s in range(0, serv_num):
            # get info of server s
            f_eta = Servers[s].f_eta
            max_bs = min(n, Servers[s].max_bs)
            C = Servers[s].C
            tau = Servers[s].tau
            # solve task n in a batch of bs tasks
            for bs in range(1, max_bs+1):
                task_time = bs*f_eta(bs)*C # computation time of task n with its batch
                best_prev_s = 0
                best_prev_bs = 1
                min_cost_for_best_prev = np.inf
                new_cost_in_server_s = np.inf
                if n-bs == 0: # batch all previous tasks
                    new_cost = 0
                    if Costs.shape[0] == 0: # no origin costs
                        new_cost = A[n] + tau + task_time
                    else:
                        new_cost = max(Costs[s], A[n] + tau) + task_time
                    if new_cost < min_cost_for_best_prev:
                        min_cost_for_best_prev = new_cost
                        best_prev_bs = 0
                        best_prev_s = 0 # does not matter
                        new_cost_in_server_s = A[n] + task_time
                else:
                    for prev_s in range(0, serv_num):
                        prev_max_bs = min(n-bs, Servers[prev_s].max_bs)
                        for prev_bs in range(1, prev_max_bs+1):
                            start_time_now = max(A[n]+tau, CostArr[n-bs][prev_s][prev_bs][s]) # server s
                            new_cost = max(CostMaxArr[n-bs][prev_s][prev_bs], start_time_now + task_time)
                            if new_cost < min_cost_for_best_prev:
                                min_cost_for_best_prev = new_cost
                                best_prev_bs = prev_bs
                                best_prev_s = prev_s
                                new_cost_in_server_s = start_time_now + task_time
                CostArr[n][s][bs] = CostArr[n-bs][best_prev_s][best_prev_bs]
                CostArr[n][s][bs][s] = new_cost_in_server_s
                CostMaxArr[n][s][bs] = min_cost_for_best_prev
                BestPrevBS[n][s][bs] = best_prev_bs
                BestPrevS[n][s][bs] = best_prev_s
                if CostMaxArr[n][s][bs] < MinCost[n]:
                    BestS[n] = s
                    BestBS[n] = bs
                    MinCost[n] = CostMaxArr[n][s][bs]

    # track the best plan
    # each batch give to which server
    ServArr = []
    # size of each batch
    BatchArr = []
    # end time of each batch
    CostResultArr = []
    n_now = req_num - 1
    bs_now = BestBS[n]
    s_now = BestS[n]
    cost_now = MinCost[n]
    while n_now > 0:
        ServArr.append(s_now)
        BatchArr.append(bs_now)
        CostResultArr.append(cost_now)
        n_new = n_now - bs_now
        bs_new = BestPrevBS[n_now][s_now][bs_now]
        s_new = BestPrevS[n_now][s_now][bs_now]
        n_now, bs_now, s_now = n_new, bs_new, s_new
    Serv = np.flip(np.array(ServArr))
    Batch = np.flip(np.array(BatchArr))
    Cost = np.flip(np.array(CostResultArr))
    
    if ws == 0:
        return Serv, Batch, Cost

    # all request is real, don't need to return more details about last batch in/accross window
    if req_num <= ws + 1:
        return Serv, Batch, Cost, 0, 0
    bs_i = 0
    bs_accum = 0
    while bs_i < Batch.shape[0]:
        bs_accum += Batch[bs_i]
        if bs_accum == ws:
            return Serv, Batch, Cost, CostArr[bs_accum][Serv[bs_i]][Batch[bs_i]], 0 # last batch in window just fill the window, don't need to return more details
        elif bs_accum > ws:
            # last batch is across the window. Because part of the tasks' arrival time is predicted,
            # we need to return more details to calculate the real cost
            WindowFinalBatchCostArray = CostArr[bs_accum][Serv[bs_i]][Batch[bs_i]]
            prev_bs = BestPrevBS[bs_accum][Serv[bs_i]][Batch[bs_i]]
            prev_s = BestPrevS[bs_accum][Serv[bs_i]][Batch[bs_i]]
            WindowFinalBatchStartCost = CostArr[bs_accum-Batch[bs_i]][prev_s][prev_bs][Serv[bs_i]]
            return Serv, Batch, Cost, WindowFinalBatchCostArray, WindowFinalBatchStartCost
    raise NotImplementedError("Wrong in MultiServer_Adaptive_Batching!")

def multiserver_window_adaptive_batching(A:np.ndarray, Servers:np.ndarray, ws:int):
    req_num = A.shape[0]
    # according to the window, get the whole predicted array A_pred
    I = np.mean(np.diff(A[:ws+1]))
    A_pred = np.array([])
    if req_num > ws+1:
        A_pred = np.concatenate([A[1:ws+1], A[ws] + np.arange(1, req_num-ws) * I])
    else:
        A_pred = A[1:]

    Costs = np.zeros_like(Servers, dtype=np.float64)
    batch_pointer = 1

    # results saving
    ServArr = []
    BatchArr = []
    CostArr = []

    while batch_pointer < req_num:
        A_now = np.concatenate([np.array([0]), A_pred])
        serv_now, batch_now, cost_now, cost_array, start_cost = multiserver_adaptive_batching_with_costs(A_now, Servers, Costs, ws)
       
        # only execute batch inside window
        bs_i = 0
        bs_accum = 0
        while bs_i < batch_now.shape[0]:
            bs_accum += batch_now[bs_i]
            if bs_accum <= ws:
                ServArr.append(serv_now[bs_i])
                BatchArr.append(batch_now[bs_i])
                CostArr.append(cost_now[bs_i])
                if bs_accum == ws:
                    break
            else:
                ServArr.append(serv_now[bs_i])
                BatchArr.append(batch_now[bs_i])
                # because part of this batch is predicted, replace predicted tasks with real arrival time
                real_start_time = max(start_cost, A[batch_pointer+bs_accum-1]+Servers[serv_now[bs_i]].tau)
                f_eta = Servers[serv_now[bs_i]].f_eta
                C = Servers[serv_now[bs_i]].C
                cost_array[serv_now[bs_i]] = real_start_time + batch_now[bs_i]*f_eta(batch_now[bs_i])*C
                CostArr.append(cost_array[np.argmax(cost_array)])
                break
            bs_i += 1
        batch_pointer += bs_accum
        if batch_pointer >= req_num:
            break
        
        # update A_pred
        if req_num - batch_pointer <= ws:
            A_pred = A[batch_pointer:]
        else:
            A_pred = np.concatenate([A[batch_pointer:batch_pointer+ws], A[batch_pointer+ws-1] + np.arange(1, req_num-ws-batch_pointer+1) * I])
        # update Costs
        Costs = cost_array
    return np.array(ServArr), np.array(BatchArr), np.array(CostArr)