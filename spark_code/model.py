import numpy as np
import scipy as sp
import pandas as pd
from scipy.integrate import solve_ivp
import warnings

#The Model class
#Team changed model to use virus data
class Model(object):
    def __call__(self, init_point, para, time_range):
        ''' forward function '''
        warnings.warn('Model call method does not implement')
        raise NotImplementedError

class Learner_SuEIR(Model):
    #Model Initialized with required parameters.
    def __init__(self, N, E_0, I_0, R_0, a, decay,vac_data,vac_date_diff, bias=0.005):
        self.N = N
        self.E_0 = E_0
        self.I_0 = I_0
        self.R_0 = R_0
        self.a = a
        self.decay = decay
        self.FRratio = a * \
            np.minimum(np.exp(-decay * (np.arange(1000) + 0)), 1)+bias
        self.pop_in = 0
        self.pop = N*5
        self.bias=1000000
        self.vac_data = vac_data
        self.vac_date_diff = vac_date_diff
        self.initial_N = N
        self.initial_pop_in = self.pop_in
        self.initial_bias=1000000
        self.prev_t =vac_date_diff
        self.last_segment_of_data =False

    def __call__(self, size, params, init, lag=0):

        beta, gamma, sigma, mu = params
        # the function for solver_ivp method that calculates the initial values for the differential equations.
        # This method is invoked multiple times based on the input size of the array passed to solver_ivp API method.
        def calc_grad(t, y):
            S, E, I, Removed = y
            if self.last_segment_of_data:
                if t >= self.prev_t:
                    diff_in_days = t - self.prev_t
                    self.prev_t = t
                    vaccinations = self.vac_data[round(t)-self.vac_date_diff]*diff_in_days
                else:
                    self.prev_t = self.vac_date_diff
                    vaccinations = 0
            else:
                vaccinations = 0
            new_pop_in = self.pop_in*(self.pop-self.N)*(np.exp(-0.03*np.maximum(0, t-self.bias))+0.05)
            return [new_pop_in-beta*S*(E+I)/self.N-0.5*vaccinations*(S/(S+Removed)), beta*S*(E+I)/self.N-sigma*E, mu*E-gamma*I, gamma*I+0.5*vaccinations*(S/(S+Removed))]
        #Solve an initial value problem for a system of ODEs, from scipy package
        solution = solve_ivp(
            calc_grad, [0, size], init, t_eval=np.arange(0, size, 1))

        # returned solution is [S, E, I, R]
        # Removed perday
        temp_r_perday = np.diff(solution.y[3])
        # Since SuEIR does not provide death dynamic, estimate death
        # grab FR_ratio per day * r perday
        temp_F_perday = temp_r_perday * \
            self.FRratio[lag:len(temp_r_perday)+lag]
        # Since the -1 day info is not accessable, we treat the ddeath of 0 day is exactly the R_0
        # which means no recover before 0-day. Then calculated cumulative death.
        temp_F = np.empty(len(temp_F_perday) + 1)
        np.cumsum(temp_F_perday, out=temp_F[1:])
        temp_F[0] = 0
        temp_F += solution.y[3][0]

        # Note that I is the active cases instead of the cumulative confirmed
        # Confirm = I + R, death is prior estimated
        # return pred_S, pred_E, pred_I, pred_R, pred_confirm, pred_fatality
        return solution.y[0], solution.y[1], solution.y[2], solution.y[3], solution.y[2] + solution.y[3], temp_F

    # reset model for params
    def reset(self):
        self.N = self.initial_N
        self.pop_in = self.initial_pop_in
        self.bias = self.initial_bias

if __name__ == '__main__':
    # m = xxx()
    # m(None, None, 1)

    pass
