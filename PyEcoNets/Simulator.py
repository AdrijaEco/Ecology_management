from jitcode import jitcode, y, t
from scipy.interpolate import interp1d
import symengine
# from symengine import lambdify
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd

class Simulator:
    def __init__(self, sim_config, sim_specs):
        self.sim_config = sim_config
        self.sim_specs = sim_specs
    
    def run(self, IsManaged = None, IsTemperatureDependent = None, SpeciesPopEnd = None, t_i = None, t_f = None, t_steps = None, show_progress = False, show_plot = False, ReturnPopEnd = False, save_results = False, results_filename = None, ReturnData = False):
        t_i = t_i if t_i is not None else self.sim_config.timesteps[0]
        t_f = t_f if t_f is not None else self.sim_config.num_years
        t_steps = t_steps if t_steps is not None else self.sim_config.num_years*365

        SIN = self.sim_config.SIN
        n, m = SIN.shape
        D = np.concatenate((np.sum(np.array(SIN), axis = 1),np.sum(np.array(SIN), axis = 0)), axis = None)

        alpha = self.sim_specs['alpha']
        gamma0 = self.sim_specs['gamma0']
        h = self.sim_specs['h']
        mu = self.sim_specs['mu']
        p = self.sim_specs['p']
        bii = self.sim_specs['bii']
        bij = self.sim_specs['bij']
        kappa = self.sim_specs['kappa']
        T_opt = self.sim_config.temperature_optimal
        sigma_alpha = self.sim_config.sigma_alpha
        sigma_h = self.sim_config.sigma_h
        Ak = self.sim_config.Ak
        
        IsManaged = IsManaged if IsManaged is not None else self.sim_config.is_managed
        IsTemperatureDependent = IsTemperatureDependent if IsTemperatureDependent is not None else self.sim_config.is_temperature_dependent
        
        if IsManaged:
            num_species = m if self.sim_config.managed_axis == 0 else n
            managed_order = np.argsort(np.sum(np.array(SIN), axis = self.sim_config.managed_axis))[::-1]
            managed_indices = managed_order[:int(num_species*self.sim_config.manage_fraction_multiple)] if self.sim_config.manage_type == 'multiple' else managed_order[:1]
        else:
            managed_indices = []

        sum_beta_plant = symengine.Symbol("sum_beta_plant")
        sum_beta_pollinator = symengine.Symbol("sum_beta_pollinator")
        helpers = [
            (sum_beta_plant, sum( y(i) for i in range(n) )),
            (sum_beta_pollinator, sum( y(n + i) for i in range(m) ))
        ]
        # T_interp_func = interp1d(self.sim_config.timesteps, self.sim_config.T['T'], kind='linear', fill_value='extrapolate')
        # T_interp = interp1d(self.sim_config.timesteps, self.sim_config.T['T'], kind='linear', fill_value='extrapolate') if IsTemperatureDependent else None
        # current_time = 0.0
        # T_sym = symengine.Function("T")(symengine.Symbol("t"))
        # def T_callback(y, timeNow):
            # return float(T_interp_func(timeNow))

        my_T = symengine.Function("my_T")
        current_time = 0.0
        T_interp_func = interp1d(self.sim_config.timesteps, self.sim_config.T['T'], kind='linear', fill_value='extrapolate')
        # def my_T_callback(y, timeNow):
        #     return float(T_interp_func(timeNow))

        T_sym = symengine.Symbol("T_sym")

        def f():
            if IsTemperatureDependent:
                # T = my_T(current_time)
                # T = T_interp(t)
                # T = T_interp(current_time)
                # T = self.sim_config.T.loc[current_time, 'T']
                alpha_ = alpha * symengine.exp(-1*((T_sym - T_opt)**2/(2*(sigma_alpha**2))))
                h_ = h * symengine.exp(1*((T_sym - T_opt)**2/(2*(sigma_h**2))))
                kappa_ = kappa * symengine.exp(Ak*((1/T_opt)-(1/T_sym)))
                bij_ = bij * symengine.exp(Ak*((1/T_opt)-(1/T_sym)))
                bii_ = bii * symengine.exp(Ak*((1/T_opt)-(1/T_sym)))
            else:
                alpha_ = alpha
                h_ = h
                kappa_ = kappa
                bij_ = bij
                bii_ = bii
            for i in range(n+m):
                if i < n:
                    if i in managed_indices and self.sim_config.managed_axis == 1:
                        yield 0
                    else:
                        m_sum = sum( y(n + j)*(gamma0 / ((D[i])**p)) for j in range(m) if SIN.iloc[i,j] )
                        yield y(i)*(alpha_ - sum_beta_plant*bij_ + (bij_ - bii_)*y(i) + (m_sum / (1 + h_*m_sum) ) ) + mu
                else:
                    i_c = i - n
                    if i_c in managed_indices and self.sim_config.managed_axis == 0:
                        yield 0
                    else:
                        m_sum = sum( y(j)*(gamma0 / ((D[n+i_c])**p)) for j in range(n) if SIN.iloc[j,i_c] )
                        yield y(i)*(alpha_ - kappa_ - sum_beta_pollinator*bij_ + (bij_ - bii_)*y(i) + (m_sum / (1 + h_*m_sum) ) ) + mu
        
        if SpeciesPopEnd is None:
            initial_state = np.ones(n+m)*1e-3
        else:
            initial_state = np.array(SpeciesPopEnd)
        if IsManaged:
            for managed_index in managed_indices:
                if self.sim_config.managed_axis == 0:
                    initial_state[n + managed_index] = self.sim_config.manage_at
                elif self.sim_config.managed_axis == 1:
                    initial_state[managed_index] = self.sim_config.manage_at

        ODE = jitcode(f, helpers=helpers, n = n+m, verbose = False, control_pars = [T_sym])
        ODE.set_parameters([T_interp_func(current_time)])
        ODE.set_integrator("dopri5")
        # ODE.set_integrator("RK45")

        ODE.set_initial_value(initial_state, 0.0)
        data = []
        times = np.linspace(t_i, t_f, t_steps, endpoint = False)
        if show_progress:
            for time in tqdm(times, total = len(times)):
                # current_time = time
                ODE.set_parameters([T_interp_func(time)])
                data.append(ODE.integrate(time))
        else:
            for time in times:
                # current_time = time
                ODE.set_parameters([T_interp_func(time)])
                data.append(ODE.integrate(time))
        data = np.array(data)
        data = pd.DataFrame(data, columns = list(SIN.index) + list(SIN.columns))
        if show_plot:
            plt.figure()
            # Different color for plant and pollinators by filtering columns that contain A or B
            plant_cols = [col for col in data.columns if 'A' in col]
            pollinator_cols = [col for col in data.columns if 'B' in col]
            plt.plot(data[plant_cols], color = 'g', alpha = 0.5)
            plt.plot(data[pollinator_cols], color = 'r', alpha = 0.5)
            # Add custom legend handles for plant and pollinators (only 2 lines) and manually add them using Line2D
            plt.legend(handles=[Line2D([0], [0], color='g', lw=2), Line2D([0], [0], color='r', lw=2)], labels=['Plant', 'Pollinator'])
            plt.show()

        if save_results:
            results_filename = 'temp.csv' if results_filename is None else results_filename
            # Add year, month, date columns to the data
            data['year'] = self.sim_config.T['year'].values
            data['month'] = self.sim_config.T['month'].values
            data['date'] = self.sim_config.T['date'].values
            data.to_csv(results_filename, index = True)

        if save_results:
            if ReturnPopEnd and (not ReturnData):
                # return data.iloc[-1].tolist()
                return data.iloc[-1].iloc[:-3].tolist(), None
            elif ReturnData and (not ReturnPopEnd):
                return None, data
            elif ReturnPopEnd and ReturnData:
                return data.iloc[-1].iloc[:-3].tolist(), data
        else:
            if ReturnPopEnd and (not ReturnData):
                # return data.iloc[-1].tolist()
                return data.iloc[-1].tolist(), None
            elif ReturnData and (not ReturnPopEnd):
                return None, data
            elif ReturnPopEnd and ReturnData:
                return data.iloc[-1].tolist(), data

# class UnsuccessfulIntegration(Exception):
# 	"""
# 		This exception is raised when the integrator cannot meet the accuracy and step-size requirements. If you want to know the exact state of your system before the integration fails or similar, catch this exception.
# 	"""
# 	pass

    # Modify the run to catch the exception
    def run_catch_exception(self, IsManaged = None, IsTemperatureDependent = None, SpeciesPopEnd = None, t_i = None, t_f = None, t_steps = None, show_progress = False, show_plot = False, ReturnPopEnd = False, save_results = False, results_filename = None, ReturnData = False):
        try:
            return self.run(IsManaged, IsTemperatureDependent, SpeciesPopEnd, t_i, t_f, t_steps, show_progress, show_plot, ReturnPopEnd, save_results, results_filename, ReturnData)
        except Exception as e:
            raise e
            # return None, None