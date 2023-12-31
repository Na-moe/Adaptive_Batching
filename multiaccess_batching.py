import numpy as np
    
from arrival import Arrival, Arrivals


def multiaccess_adaptive_batching(Arr:Arrivals, Servers:np.ndarray, Transmission):
    assert len(Servers) == len(Transmission)
    assert len(Servers) == len(Transmission[0])
    
    A = Arr.view_1d()
    req_num = A.shape[0]
    serv_num = Servers.shape[0]
    # CostArr[n][s1][bs][s2]:
    # the end time of s2, when task n-bs+1:n+1 are completed in server s1
    CostArr = np.zeros([req_num, serv_num, req_num, serv_num], dtype=np.float64)
    # CostMaxArr[n][s][bs] = max(CostArr[n][s][bs][:])
    # the max end time of all servers, when task n with previous bs-1 tasks are completed in server s
    CostMaxArr = np.zeros([req_num, serv_num, req_num], dtype=np.float64)
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
            tau = Transmission[:, s]
            # solve task n in a batch of bs tasks
            for bs in range(1, max_bs+1):
                task_time = bs*f_eta(bs)*C
                best_prev_s = 0
                best_prev_bs = 1
                min_cost_for_best_prev = np.inf
                new_cost_in_server_s = np.inf
                if n-bs == 0:
                    new_cost = A[n].time + tau[A[n].server_id] + task_time
                    if new_cost < min_cost_for_best_prev:
                        min_cost_for_best_prev = new_cost
                        best_prev_bs = 0
                        best_prev_s = 0
                        new_cost_in_server_s = A[n].time + task_time
                else:
                    for prev_s in range(0, serv_num):
                        prev_max_bs = min(n-bs, Servers[prev_s].max_bs)
                        prev_bs = np.arange(1, prev_max_bs+1)
                        start_time_now = np.maximum(A[n].time+tau[A[n].server_id], CostArr[n-bs][prev_s][prev_bs][:, s])
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