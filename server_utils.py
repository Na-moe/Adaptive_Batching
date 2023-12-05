import numpy as np

from functools import partial


def get_f_eta(hardware='V100', net='resnet50'):
    """
    Profile the corresponding f_eta according to specific hardware and neural network.

    Parameters
    ----------
    hardware: str
    net: str

    Returns
    -------
    f_eta: func(batch_size) -> eta
        Mapping batch_size to eta(parallelism efficiency).
    """
    if hardware == 'V100':
        ggnet = (1.74344597, -0.713036799, 0.7273)
        resnet50 = (2.117312253, -0.600869565, 0.5476)
        vgg19 = (0.517647059, 1.242701525, 0.6250)
    elif hardware == 'AGX_Xavier_15W':
        ggnet = (0.17912012, 1.265605618, 0.8056)
        resnet50 = (0.188389155, 1.410441255, 0.6964)
        vgg19 = (0.0558153, 1.845600676, 0.6387)
    elif hardware == 'AGX_Xavier_MAXN':
        ggnet = (0.203300575, 1.307995367, 0.7333)
        resnet50 = (0.203300575, 1.307995367, 0.7333)
        vgg19 = (0.434379505, 1.12778763, 0.6641)
    else:
        raise NotImplementedError(f'eta of hardware {hardware} not implemented')
    
    f_eta_dict = {
            'ggnet': ggnet,
            'resnet50': resnet50,
            'vgg19': vgg19,
        }

    def f_eta(batch_size, net):
        k, b, eta2 = f_eta_dict[net]
        b = 2 ** (b / k)

        if isinstance(batch_size, np.ndarray):
            ans = 1./ (k * np.log2(b * batch_size))
            ans[batch_size==1] = 1
            ans[batch_size==2] = eta2
            return ans
        
        if batch_size == 0:
            ans = 0
        if batch_size == 1:
            ans = 1
        elif batch_size == 2:
            ans = eta2
        else:
            ans = 1./ (k * np.log2(b * batch_size))
        return ans
    
    return partial(f_eta, net=net)

def get_max_bs(hardware='V100', net='resnet50'):
    """
    Get the maximum batch size of specific hardware and neural network.

    Parameters
    ----------
    hardware: str
    net: str

    Returns
    -------
    max_bs: int
        The maximum batch size.
    """
    if hardware == 'V100':
        ggnet = 128
        resnet50 = 128
        vgg19 = 128
    elif hardware == 'AGX_Xavier_15W':
        ggnet = 32
        resnet50 = 32
        vgg19 = 16
    elif hardware == 'AGX_Xavier_MAXN':
        ggnet = 32
        resnet50 = 32
        vgg19 = 16
    else:
        raise NotImplementedError(f'eta of hardware {hardware} not implemented')
    
    max_bs_dict = {
            'ggnet': ggnet,
            'resnet50': resnet50,
            'vgg19': vgg19,
        }

    return max_bs_dict[net]

def get_computing_time(hardware='V100', net='resnet50'):
    """
    Get the computing time of specific hardware and neural network.

    Parameters
    ----------
    hardware: str
    net: str

    Returns
    -------
    computing_time: float
        The computing time.
    """
    if hardware == 'V100':
        resnet50 = 30
    elif hardware == 'AGX_Xavier_15W':
        resnet50 = 40
    elif hardware == 'AGX_Xavier_MAXN':
        resnet50 = 21.5
    else:
        raise NotImplementedError(f'eta of hardware {hardware} not implemented')
    
    computing_time_dict = {
            'resnet50': resnet50,
        }
    
    return computing_time_dict[net]

