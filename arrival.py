import numpy as np

from typing import List


class Arrival():
    def __init__(self, arrival_time:float, server_id:int, idx:int):
        self.ori_time = arrival_time
        self.time = arrival_time
        self.server_id = int(server_id)
        self.idx = idx

    def __repr__(self) -> str:
        return f'Arrival: time={self.time}, server_id={self.server_id}.'
    
    def __str__(self) -> str:
        return f'Arrival: time={self.time}, server_id={self.server_id}.'
    
    def __lt__(self, other):
        if type(other) is Arrival:
            lt = self.time < other.time if self.time != other.time else self.server_id < other.server_id
        else:
            lt = self.time < other
        return lt
    
    def __le__(self, other):
        if type(other) is Arrival:
            le = self.time <= other.time if self.time != other.time else self.server_id <= other.server_id
        else:
            le = self.time <= other
        return le
    
    def __eq__(self, other):
        if type(other) is Arrival:
            eq = self.time == other.time
        else:
            eq = self.time == other
        return eq
    
    def __gt__(self, other):
        if type(other) is Arrival:
            gt = self.time > other.time if self.time != other.time else self.server_id > other.server_id
        else:
            gt = self.time > other
        return gt
    
    def __ge__(self, other):
        if type(other) is Arrival:
            ge = self.time >= other.time if self.time != other.time else self.server_id >= other.server_id
        else:
            ge = self.time >= other
        return ge
    
    def __add__(self, other):
        if type(other) is not Arrival:
            self.time = self.ori_time + other
        else:
            raise NotImplementedError(f"Unsupported Type{type(other)} in __add__ of Class Arrival")
        return self

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
    