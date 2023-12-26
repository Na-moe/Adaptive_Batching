import numpy as np
    
from arrival import Arrival, Arrivals
from batching import adaptive_batching, sot_batching
from batching_utils import batch_cost_arrival


def multiaccess_task_balance(Arr_list, Servers, Transmission):
    Assigned_server = [[i for _ in range(len(arr))] for i, arr in enumerate(Arr_list)]

    Arr_list_now = Arr_list.copy()
    Solutions = [adaptive_batching(Arr_list_now[i], Servers[i]) for i in range(len(Servers))]
    Starts = [sol[0] for sol in Solutions]
    Batches = [sol[1] for sol in Solutions]
    Costs = [sol[2] for sol in Solutions]
    End_Costs = np.array([cost[-1] for cost in Costs])

    iter_steps = 0 # for debug
    should_rebatch = False
    while True:
        # print(f"in-loop iter_steps:{iter_steps}")
        # find the server with the largest cost
        m = max_cost_server = np.argmax(End_Costs)
        max_cost = End_Costs[m]
        # print(f"m:{m}", Starts[m])
        # iterate over latest task of each batch and all tasks of last batch from right to left
        iter_idxs = Starts[m][1:] - 1 # next start - 1 = prev end
        iter_idxs = np.concatenate((iter_idxs, np.arange(Starts[m][-2], Starts[m][-1] - 1)))
        iter_idxs = np.flip(iter_idxs)
        # iterate
        accepted_i = False
        for i in iter_idxs:
            if accepted_i:
                break
            arrivals_m = Arr_list_now[m]
            arrival_i = arrivals_m[i]
            start_m, batch_m = Starts[m], Batches[m]
            batch_m_idx = np.argwhere((start_m[:-1] <= i) & (i < start_m[1:])).flatten()
            assert len(batch_m_idx) == 1
            batch_m_copy = batch_m.copy()
            batch_m_copy[batch_m_idx+1] -= 1
            if batch_m_copy[batch_m_idx+1] == 0:
                batch_m_copy = np.delete(batch_m_copy, batch_m_idx+1)
            
            arrivals_m_copy = np.delete(arrivals_m, i)
            # print(len(arrivals_m), np.sum(batch_m_copy), batch_m_copy[0])
            cost_m = batch_cost_arrival(arrivals_m_copy, Servers[m], batch_m_copy)

            # if max_cost is already the largest, no offloading is better
            if cost_m[-1] >= max_cost:
                continue

            # try offload it to other servers
            for j, server_j in enumerate(Servers):
                if j == m:
                    continue
                else:
                    iter_steps += 1
                    tau = Transmission[arrival_i.server_id][j]
                    arrival_i_tau = arrival_i + tau
                    arrivals_j = Arr_list_now[j]
                    arrivals_j = np.sort(arrivals_j)
                    insert_idx = np.searchsorted(arrivals_j, arrival_i_tau)
                    arrivals_j_new = np.insert(arrivals_j, insert_idx, arrival_i_tau)
                    start_j, batch_j = Starts[j], Batches[j]
                    # print(arrival_i_tau, arrivals_j, start_j, batch_j)

                    # case 0: no batches in server_j
                    if np.sum(batch_j) == 0:
                        # print(f'case 0')
                        batch_j1 = np.array([0, 1])
                        cost_j1 = batch_cost_arrival(arrivals_j_new, server_j, batch_j1)
                        batch_js = np.array([batch_j1])
                        cost_js = np.array([cost_j1])
                        end_cost_js = np.array([cost_j1[-1]])

                    # case 1: arrival_i_tau is in a batch
                    elif np.any((arrivals_j[start_j[1:-1]] <= arrival_i_tau) & (arrival_i_tau <= arrivals_j[start_j[2:]-1])):
                        # print(f'case 1')
                        # op1: insert arrival_i_tau into batch[idx]
                        batch_j_mask = (arrivals_j[start_j[:-1]] <= arrival_i_tau) & (arrival_i_tau <= arrivals_j[start_j[1:]-1])
                        batch_j_idx = np.argwhere(batch_j_mask).flatten()
                        batch_j1 = batch_j.copy()
                        batch_j1[batch_j_idx+1] += 1
                        # calculate batch_cost_arrival of server j
                        cost_j = batch_cost_arrival(arrivals_j_new, server_j, batch_j1)
                        # vertorizingly select the better one
                        batch_js = np.array([batch_j1])
                        cost_js = np.array([cost_j])
                        end_cost_js = np.array([cost_j[-1]])
                            
                    # case 2.1: arrival_i_tau is before first batch
                    elif arrival_i_tau < arrivals_j[1]:
                        # print(f'case 2.1')
                        # op1: arrival_i_tau as a seperate batch
                        batch_j1 = np.insert(batch_j, 1, 1)
                        cost_j1 = batch_cost_arrival(arrivals_j_new, server_j, batch_j1)
                        # op2: arrival_i_tau merged into batch[0]
                        batch_j2 = batch_j.copy()
                        batch_j2[1] += 1
                        cost_j2 = batch_cost_arrival(arrivals_j_new, server_j, batch_j2)
                        # vertorizingly select the better one
                        batch_js = np.array([batch_j1, batch_j2], dtype=object)
                        cost_js = np.array([cost_j1, cost_j2], dtype=object)
                        end_cost_js = np.array([cost_j1[-1], cost_j2[-1]])
                        
                    # case 2.2: arrival_i_tau is after last batch
                    elif arrival_i_tau > arrivals_j[-1]:
                        # print(f'case 2.2')
                        # op1: arrival_i_tau as a seperate batch
                        batch_j1 = np.insert(batch_j, len(batch_j), 1)
                        cost_j1 = batch_cost_arrival(arrivals_j_new, server_j, batch_j1)
                        # op2: arrival_i_tau merged into batch[-1]
                        batch_j2 = batch_j.copy()
                        batch_j2[-1] += 1
                        cost_j2 = batch_cost_arrival(arrivals_j_new, server_j, batch_j2)
                        # vertorizingly select the better one
                        batch_js = np.array([batch_j1, batch_j2], dtype=object)
                        cost_js = np.array([cost_j1, cost_j2], dtype=object)
                        end_cost_js = np.array([cost_j1[-1], cost_j2[-1]])

                    # case 2.3: arrival_i_tau is between 2 batches
                    else:
                        # print(f'case 2.3')
                        # find arrival_i_tau is between which 2 batches
                        batch_j_mask = (arrivals_j[start_j[1:-1]-1] < arrival_i_tau) & (arrival_i_tau < arrivals_j[start_j[1:-1]])
                        batch_j_idx = np.argwhere(batch_j_mask).flatten() + 1
                        # op1: arrival_i_tau as a seperate batch
                        batch_j1 = np.insert(batch_j, batch_j_idx+1, 1)
                        cost_j1 = batch_cost_arrival(arrivals_j_new, server_j, batch_j1)
                        # op2: arrival_i_tau merged into left batch
                        batch_j2 = batch_j.copy()
                        batch_j2[batch_j_idx] += 1
                        cost_j2 = batch_cost_arrival(arrivals_j_new, server_j, batch_j2)
                        # op3: arrival_i_tau merged into right batch
                        batch_j3 = batch_j.copy()
                        batch_j3[batch_j_idx+1] += 1
                        cost_j3 = batch_cost_arrival(arrivals_j_new, server_j, batch_j3)

                        # vertorizingly select the better one
                        batch_js = np.array([batch_j1, batch_j2, batch_j3], dtype=object)
                        cost_js = np.array([cost_j1, cost_j2, cost_j3], dtype=object)
                        end_cost_js = np.array([cost_j1[-1], cost_j2[-1], cost_j3[-1]])

                    if np.any(end_cost_js < max_cost):

                        idx = np.argmin(end_cost_js)
                        # print(f'idx:{idx}, all_idxs:{len(end_cost_js)}')
                        Arr_list_now[m], Arr_list_now[j] = arrivals_m_copy, arrivals_j_new
                        Batches[m], Batches[j] = batch_m_copy, batch_js[idx]
                        Starts[m], Starts[j] = np.cumsum(Batches[m])+1, np.cumsum(Batches[j])+1
                        Costs[m], Costs[j] = cost_m, cost_js[idx]
                        End_Costs[m], End_Costs[j] = cost_m[-1], end_cost_js[idx]

                        Assigned_server[arrival_i.server_id][arrival_i.idx] = j
                        accepted_i = True
                        break

        # An offloading is accepeted
        if accepted_i:
            should_rebatch = True

        # no offloading is better, 
        else:
            if not should_rebatch:
                break

            # print(f"rebatching: {[len(arr) for arr in Arr_list_now]}")
            Solutions = [adaptive_batching(Arr_list_now[i], Servers[i]) for i in range(len(Servers))]
            Starts = [sol[0] for sol in Solutions]
            Batches = [sol[1] for sol in Solutions]
            Costs = [sol[2] for sol in Solutions]
            End_Costs = np.array([cost[-1] for cost in Costs])

            should_rebatch = False
    # print(f"iter_steps:{iter_steps}")
    return Solutions, Assigned_server, Arr_list_now

def multiaccess_task_batching(Arr_list, Servers, Transmission, batching:function):
    Assigned_server = [[i for _ in range(len(arr))] for i, arr in enumerate(Arr_list)]

    Arr_list_now = Arr_list.copy()
    Solutions = [batching(Arr_list_now[i], Servers[i]) for i in range(len(Servers))]
    Batches = [sol[-2] for sol in Solutions]
    Costs = [sol[-1] for sol in Solutions]
    Starts = [np.cumsum(Batches[i])+1 for i in range(len(Servers))]
    End_Costs = np.array([cost[-1] for cost in Costs])

    iter_steps = 0 # for debug
    should_rebatch = False
    while True:
        # print(f"in-loop iter_steps:{iter_steps}")
        # find the server with the largest cost
        m = max_cost_server = np.argmax(End_Costs)
        max_cost = End_Costs[m]
        # print(f"m:{m}", Starts[m])
        # iterate over latest task of each batch and all tasks of last batch from right to left
        iter_idxs = Starts[m][1:] - 1 # next start - 1 = prev end
        iter_idxs = np.concatenate((iter_idxs, np.arange(Starts[m][-2], Starts[m][-1] - 1)))
        iter_idxs = np.flip(iter_idxs)
        # iterate
        accepted_i = False
        for i in iter_idxs:
            if accepted_i:
                break
            arrivals_m = Arr_list_now[m]
            arrival_i = arrivals_m[i]
            start_m, batch_m = Starts[m], Batches[m]
            batch_m_idx = np.argwhere((start_m[:-1] <= i) & (i < start_m[1:])).flatten()
            assert len(batch_m_idx) == 1
            batch_m_copy = batch_m.copy()
            batch_m_copy[batch_m_idx+1] -= 1
            if batch_m_copy[batch_m_idx+1] == 0:
                batch_m_copy = np.delete(batch_m_copy, batch_m_idx+1)
            
            arrivals_m_copy = np.delete(arrivals_m, i)
            # print(len(arrivals_m), np.sum(batch_m_copy), batch_m_copy[0])
            cost_m = batch_cost_arrival(arrivals_m_copy, Servers[m], batch_m_copy)

            # if max_cost is already the largest, no offloading is better
            if cost_m[-1] >= max_cost:
                continue

            # try offload it to other servers
            for j, server_j in enumerate(Servers):
                if j == m:
                    continue
                else:
                    iter_steps += 1
                    tau = Transmission[arrival_i.server_id][j]
                    arrival_i_tau = arrival_i + tau
                    arrivals_j = Arr_list_now[j]
                    arrivals_j = np.sort(arrivals_j)
                    insert_idx = np.searchsorted(arrivals_j, arrival_i_tau)
                    arrivals_j_new = np.insert(arrivals_j, insert_idx, arrival_i_tau)
                    start_j, batch_j = Starts[j], Batches[j]
                    # print(arrival_i_tau, arrivals_j, start_j, batch_j)

                    # case 0: no batches in server_j
                    if np.sum(batch_j) == 0:
                        # print(f'case 0')
                        batch_j1 = np.array([0, 1])
                        cost_j1 = batch_cost_arrival(arrivals_j_new, server_j, batch_j1)
                        batch_js = np.array([batch_j1])
                        cost_js = np.array([cost_j1])
                        end_cost_js = np.array([cost_j1[-1]])

                    # case 1: arrival_i_tau is in a batch
                    elif np.any((arrivals_j[start_j[1:-1]] <= arrival_i_tau) & (arrival_i_tau <= arrivals_j[start_j[2:]-1])):
                        # print(f'case 1')
                        # op1: insert arrival_i_tau into batch[idx]
                        batch_j_mask = (arrivals_j[start_j[:-1]] <= arrival_i_tau) & (arrival_i_tau <= arrivals_j[start_j[1:]-1])
                        batch_j_idx = np.argwhere(batch_j_mask).flatten()
                        batch_j1 = batch_j.copy()
                        batch_j1[batch_j_idx+1] += 1
                        # calculate batch_cost_arrival of server j
                        cost_j = batch_cost_arrival(arrivals_j_new, server_j, batch_j1)
                        # vertorizingly select the better one
                        batch_js = np.array([batch_j1])
                        cost_js = np.array([cost_j])
                        end_cost_js = np.array([cost_j[-1]])
                            
                    # case 2.1: arrival_i_tau is before first batch
                    elif arrival_i_tau < arrivals_j[1]:
                        # print(f'case 2.1')
                        # op1: arrival_i_tau as a seperate batch
                        batch_j1 = np.insert(batch_j, 1, 1)
                        cost_j1 = batch_cost_arrival(arrivals_j_new, server_j, batch_j1)
                        # op2: arrival_i_tau merged into batch[0]
                        batch_j2 = batch_j.copy()
                        batch_j2[1] += 1
                        cost_j2 = batch_cost_arrival(arrivals_j_new, server_j, batch_j2)
                        # vertorizingly select the better one
                        batch_js = np.array([batch_j1, batch_j2], dtype=object)
                        cost_js = np.array([cost_j1, cost_j2], dtype=object)
                        end_cost_js = np.array([cost_j1[-1], cost_j2[-1]])
                        
                    # case 2.2: arrival_i_tau is after last batch
                    elif arrival_i_tau > arrivals_j[-1]:
                        # print(f'case 2.2')
                        # op1: arrival_i_tau as a seperate batch
                        batch_j1 = np.insert(batch_j, len(batch_j), 1)
                        cost_j1 = batch_cost_arrival(arrivals_j_new, server_j, batch_j1)
                        # op2: arrival_i_tau merged into batch[-1]
                        batch_j2 = batch_j.copy()
                        batch_j2[-1] += 1
                        cost_j2 = batch_cost_arrival(arrivals_j_new, server_j, batch_j2)
                        # vertorizingly select the better one
                        batch_js = np.array([batch_j1, batch_j2], dtype=object)
                        cost_js = np.array([cost_j1, cost_j2], dtype=object)
                        end_cost_js = np.array([cost_j1[-1], cost_j2[-1]])

                    # case 2.3: arrival_i_tau is between 2 batches
                    else:
                        # print(f'case 2.3')
                        # find arrival_i_tau is between which 2 batches
                        batch_j_mask = (arrivals_j[start_j[1:-1]-1] < arrival_i_tau) & (arrival_i_tau < arrivals_j[start_j[1:-1]])
                        batch_j_idx = np.argwhere(batch_j_mask).flatten() + 1
                        # op1: arrival_i_tau as a seperate batch
                        batch_j1 = np.insert(batch_j, batch_j_idx+1, 1)
                        cost_j1 = batch_cost_arrival(arrivals_j_new, server_j, batch_j1)
                        # op2: arrival_i_tau merged into left batch
                        batch_j2 = batch_j.copy()
                        batch_j2[batch_j_idx] += 1
                        cost_j2 = batch_cost_arrival(arrivals_j_new, server_j, batch_j2)
                        # op3: arrival_i_tau merged into right batch
                        batch_j3 = batch_j.copy()
                        batch_j3[batch_j_idx+1] += 1
                        cost_j3 = batch_cost_arrival(arrivals_j_new, server_j, batch_j3)

                        # vertorizingly select the better one
                        batch_js = np.array([batch_j1, batch_j2, batch_j3], dtype=object)
                        cost_js = np.array([cost_j1, cost_j2, cost_j3], dtype=object)
                        end_cost_js = np.array([cost_j1[-1], cost_j2[-1], cost_j3[-1]])

                    if np.any(end_cost_js < max_cost):

                        idx = np.argmin(end_cost_js)
                        # print(f'idx:{idx}, all_idxs:{len(end_cost_js)}')
                        Arr_list_now[m], Arr_list_now[j] = arrivals_m_copy, arrivals_j_new
                        Batches[m], Batches[j] = batch_m_copy, batch_js[idx]
                        Starts[m], Starts[j] = np.cumsum(Batches[m])+1, np.cumsum(Batches[j])+1
                        Costs[m], Costs[j] = cost_m, cost_js[idx]
                        End_Costs[m], End_Costs[j] = cost_m[-1], end_cost_js[idx]

                        Assigned_server[arrival_i.server_id][arrival_i.idx] = j
                        accepted_i = True
                        break

        # An offloading is accepeted
        if accepted_i:
            should_rebatch = True

        # no offloading is better, 
        else:
            if not should_rebatch:
                break

            # print(f"rebatching: {[len(arr) for arr in Arr_list_now]}")
            Solutions = [batching(Arr_list_now[i], Servers[i]) for i in range(len(Servers))]
            Batches = [sol[-2] for sol in Solutions]
            Costs = [sol[-1] for sol in Solutions]
            Starts = [np.cumsum(Batches[i])+1 for i in range(len(Servers))]
            End_Costs = np.array([cost[-1] for cost in Costs])

            should_rebatch = False
    # print(f"iter_steps:{iter_steps}")
    return Solutions, Assigned_server, Arr_list_now

def ocai(Arr_list, Servers, Transmission):
    time_slot = 100

    def resouce_listing(Servers):
        holding_cost = time_slot * .4
        for server in Servers:
            bs = np.arange(1, server.max_bs+1)
            comp_cost = server.f_eta(bs) * bs *server.C
            D = np.sum(comp_cost < time_slot) # Demand
            server.D = D
            idle_cost = time_slot - np.mean(comp_cost[:D] / bs[:D])
            last_Q = Q = 10
            last_R = R = D // 2
            Q = int(np.sqrt(D * idle_cost * (D-last_R)**2 / holding_cost))
            R = int(D - holding_cost * last_Q / (idle_cost * D))
            while Q != last_Q or R != last_R:
                last_Q, last_R = Q, R
                Q = int(np.sqrt(D * idle_cost * (D-last_R)**2 / holding_cost))
                R = int(D - holding_cost * last_Q / (idle_cost * D))
            server.Q, server.R = Q, R

    resouce_listing(Servers)
    NS = num_servers = len(Servers)
    comp_cost = np.zeros(NS)
    for j, server in enumerate(Servers):
        bs = np.arange(1, server.max_bs+1)
        comp_cost[j] = np.mean(server.f_eta(bs) * server.C)
    utility = np.zeros((NS, NS))
    for i in range(NS):
        for j, server in enumerate(Servers):
            utility[i,j] = -server.Q * (comp_cost[j] + Transmission[i][j])
    max_utility = np.argsort(utility, axis=1)[:, ::-1]

    def bidding(Arr_list, Assigned_server, idx, Servers):
        start, end = time_slot * idx, time_slot * (idx+1)
        As_in_slot = [Assigned_server[i][(start <= Arr) & (Arr < end)] for i, Arr in enumerate(Arr_list)]
        for i in range(len(Servers)):
            js = max_utility[i]
            jdx = 0
            for j in js:
                off = min(len(As_in_slot[i])-jdx, Servers[j].Q)
                As_in_slot[i][jdx:jdx+off] = j
                jdx += off
            Arr = Arr_list[i]
            Assigned_server[i][(start <= Arr) & (Arr < end)] = As_in_slot[i]
        return As_in_slot
    
    Assigned_server = [np.array([i for _ in range(len(arr))]) for i, arr in enumerate(Arr_list)]
    idx = 0
    last_arr = max(*[Arr[-1] for Arr in Arr_list])
    while time_slot * idx <= last_arr:
        bid = bidding(Arr_list, Assigned_server, idx, Servers)
        idx += 1
    return Assigned_server

def edge_batch(Arr_list, Servers, Transmission):
    Assigned_server = ocai(Arr_list, Servers, Transmission)
    Batch = []
    for i in range(len(Servers)):
        Arr = np.concatenate([Arr_list[j][As==i] + Transmission[j][i] for j, As in enumerate(Assigned_server)])
        Arr = np.sort(Arr)
        Batch.append(sot_batching(Arr, Servers[i]))
    return Batch