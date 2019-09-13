import torch.optim as optim
from layers.scheduler_base import *

class Adam3(SchedulerBase):
    def __init__(self,params_list=None):
        super(Adam3, self).__init__()
        self._lr = 1e-4
        self._cur_optimizer = None
        self.params_list=params_list
    def schedule(self,net, epoch, epochs, **kwargs):
        lr = 1e-4
        if epoch > 25:
            lr = 5e-5
        if epoch > 35:
            lr = 1e-5
        self._lr = lr
        if self._cur_optimizer is None:
            self._cur_optimizer = optim.Adam(net.parameters(), lr=lr)  # , weight_decay=0.0005
        return self._cur_optimizer, self._lr
