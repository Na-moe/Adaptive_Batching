import numpy as np

from typing import List


class Arrival():
    def __init__(self, arrival_time:float, server_id:int):
        self.time = arrival_time
        self.server_id = server_id

    def __repr__(self) -> str:
        return f'Arrival: time={self.time}, server_id={self.server_id}.'
    
    def __str__(self) -> str:
        return f'Arrival: time={self.time}, server_id={self.server_id}.'
    
    def __lt__(self, other):
        lt = self.time < other.time if self.time != other.time else self.server_id < other.server_id
        return lt

class Arrivals():
    def __init__(self, arrivals_list:List[Arrival], num_servers:int):
        assert len(arrivals_list) == num_servers

        self.arrivals_list = arrivals_list
        # unpack list and concat into one array
        arrivals = np.concatenate(arrivals_list)
        sorted_arrivals = np.sort(arrivals, order='time')
        if sorted_arrivals[0].time != 0:
            A = np.concatenate([np.array([Arrival(0, -1)]), sorted_arrivals])
        else:
            A = sorted_arrivals
        self.A = A

    def __repr__(self) -> str:
        return f'Arrivals: {self.A}.'
    
    def view_2d(self):
        return self.arrivals_list
    
    def view_1d(self):
        return self.A
    