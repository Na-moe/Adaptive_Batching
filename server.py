from server_utils import get_f_eta, get_max_bs, get_computing_time


class Server():
    def __init__(self, hardware, net, tau=0.0):
        self.hardware = hardware
        self.net = net
        self.tau = tau
        self.fast_init()

    def fast_init(self):
        self.f_eta = get_f_eta(self.hardware, self.net)
        self.max_bs = get_max_bs(self.hardware, self.net)
        self.C = get_computing_time(self.hardware, self.net)
    
    def __repr__(self):
        return f'Server(hardware={self.hardware}, net={self.net}, max_bs={self.max_bs}, C={self.C}, tau={self.tau})'