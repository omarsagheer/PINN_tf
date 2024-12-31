from abc import ABC

from forward_problem.forward_pinn_structure.training_f_pinn import TrainingFPINN
from forward_problem.forward_pinn_structure.util_f_pinn import UtilFPINN


class ForwardPINN(TrainingFPINN, UtilFPINN, ABC):
    """This is a 'funnel' class that combines the training and utility classes for the forward PINN structure"""
    def __init__(self, n_int, n_sb, n_tb, x_domain, t_domain, **kwargs):
        super().__init__(n_int, n_sb, n_tb, x_domain, t_domain, **kwargs)