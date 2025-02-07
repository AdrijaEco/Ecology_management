import configparser
import numpy as np
import pandas as pd

class GlobalConfig:
    def __init__(self, config_file):
        config = configparser.ConfigParser()
        config.read(config_file)

        sections = {k:len(config[k]) for k in config.sections()}
        num_sections = len(sections.keys())

        self.ROOT = config.get('PATHS', 'ROOT')
        self.DATA = config.get('PATHS', 'DATA')
        self.RESULTS = config.get('PATHS', 'RESULTS')

        SIN = config.get('PARAMS', 'SpeciesInteractionNetwork')
        SIN = pd.read_csv(SIN, index_col=0)
        SIN.columns = np.char.add('B',np.array(range(len(SIN.columns)), dtype = 'str'))
        SIN.index = np.char.add('A',np.array(range(len(SIN.index)), dtype = 'str'))
        SIN[SIN > 0] = 1
        self.SIN = SIN

        self.year_start = int(config.get('PARAMS', 'YearStart'))
        self.year_end = int(config.get('PARAMS', 'YearEnd'))
        self.num_years = self.year_end - self.year_start + 1

        self.is_managed =  config.getboolean('PARAMS', 'IsManaged')
        managed_target = config.get('PARAMS', 'ManagedTarget').lower()
        if managed_target == 'pollinator':
            self.managed_axis = 0
        elif managed_target == 'plant':
            self.managed_axis = 1
        else:
            raise ValueError(f'Invalid managed target: {managed_target}')
        self.manage_at = float(config.get('PARAMS', 'ManagedAt'))
        self.manage_type = config.get('PARAMS', 'ManagedType').lower()
        self.manage_fraction_multiple = float(config.get('PARAMS', 'FractionManagedWhenMultiple'))

        self.is_temperature_dependent = config.getboolean('PARAMS', 'IsTemperatureDependent')
        self.frequency = config.get('PARAMS', 'Frequency')
        self.temperature_optimal = float(config.get('PARAMS', 'TemperatureOptimal'))
        self.sigma_alpha = float(config.get('PARAMS', 'SigmaAlpha'))
        self.sigma_h = float(config.get('PARAMS', 'Sigmah'))
        self.Ak = float(config.get('PARAMS', 'Ak'))

        temperatures_filepath = config.get('PARAMS', 'TemperatureData')
        self.temperatures = pd.read_csv(temperatures_filepath, index_col=0, parse_dates=True)
        self.temperatures = self.temperatures[~((self.temperatures.index.month == 2) & (self.temperatures.index.day == 29))]
        self.temperatures['year'] = self.temperatures.index.year
        self.temperatures['month'] = self.temperatures.index.month
        self.temperatures['date'] = self.temperatures.index.day

        self.timesteps = np.linspace(0, self.num_years, self.num_years*365, endpoint=False)
        self.dates = pd.date_range(f'{self.year_start}-01-01', f'{self.year_end}-12-31', freq='D')
        self.dates = self.dates[~((self.dates.month == 2) & (self.dates.day == 29))]

        self.T = pd.DataFrame(index = self.timesteps, columns = ['year', 'month', 'date', 'T'])
        self.T['year'] = self.dates.year
        self.T['month'] = self.dates.month
        self.T['date'] = self.dates.day
        if self.frequency == 'Y':
            for _, row in self.temperatures.iterrows():
                self.T.loc[self.T['year'] == row['year'], 'T'] = row['T']
        elif self.frequency == 'M':
            for _, row in self.temperatures.iterrows():
                self.T.loc[(self.T['year'] == row['year']) & (self.T['month'] == row['month']), 'T'] = row['T']
        elif self.frequency == 'D':
            for _, row in self.temperatures.iterrows():
                self.T.loc[(self.T['year'] == row['year']) & (self.T['month'] == row['month']) & (self.T['date'] == row['date']), 'T'] = row['T']

        self.params = {
            'alpha': None,
            'gamma0': None,
            'h': None,
            'mu': None,
            'p': None,
            'bii': None,
            'bij': None,
            'kappa': None,
        }

        section = 'CONSTANTS'
        for key in self.params.keys():
            if key in config[section]:
                self.params[key] = (section, float(config[section][key]))
        section = 'RANGE'
        for key in self.params.keys():
            if key in config[section]:
                self.params[key] = (section, [float(k) for k in config[section][key].split(',')])
        # section = 'RANDOM'
        # for key in self.params.keys():
        #     if key in config[section]:
        #         self.params[key] = (section, [float(k) for k in config[section][key].split(',')])

# config = GlobalConfig('/home/sarth/Desktop/GNN_rsynced/AAM/PyEcoNets/config.ini')
# config.params