import numpy as np

class Cell:
    '''
    template of a cell with default parameters
    '''
    def __init__(self):
        self._set_timescales()
        self._set_connections()
    def _set_timescales(self):
        self.alpha_pyr=50
        self.alpha_bic=50
        self.alpha_cck=80 
        self.alpha_pv=100
    def _set_connections(self):
        self.wpyrpyr = 0.03
        self.wpyrbic = 0.04
        self.wpyrpv = 0.02

        self.wbicpyr = -0.03 #-0.08

        self.wcckcck = -0.15
        self.wcckpv = -0.15

        self.wpvpv = -0.055
        self.wpvpyr = -0.04 # used to be 0.0399
        self.wpvcck = -0.075
        
        # inputs
        self.D_pyr = 0.001
        self.D_cck = 0.001
        self.D_pv = 0.001
        self.D_bic = 0.001

        self.i_pyr = 0.07
        self.i_bic = -1.05
        self.i_cck = 0.70
        self.i_pv = 0.45
    def _set_init_state(self, N):
        '''
        sets initial state for a simulation
        N = number of samples
        '''
        self.r_pyr = np.zeros(N)
        self.r_bic = np.zeros(N)
        self.r_cck = np.zeros(N)
        self.r_pv = np.zeros(N)
