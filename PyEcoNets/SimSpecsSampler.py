class SimSpecsSampler:
    def __init__(self, params):
        self.params = params

        num_constants = sum([k[0] == 'CONSTANTS' for k in self.params.values()])
        num_range = sum([k[0] == 'RANGE' for k in self.params.values()])
        num_random = sum([k[0] == 'RANDOM' for k in self.params.values()])
        # print(f'num_constants: {num_constants}, num_range: {num_range}, num_random: {num_random}')

        self.AllSimSpecs = []
        SingleSimSpecs = {
            'alpha': None,
            'gamma0': None,
            'h': None,
            'mu': None,
            'p': None,
            'bii': None,
            'bij': None,
            'kappa': None,
        }
        for param in self.params.keys():
            if self.params[param][0] == 'CONSTANTS':
                SingleSimSpecs[param] = self.params[param][1]
            # elif self.params[param][0] == 'RANDOM':
            #     SingleSimSpecs[param] = self.params[param][1]
            #     SingleSimSpecs[param][-1] = int(SingleSimSpecs[param][-1])
        for param in self.params.keys():
            if (SingleSimSpecs[param] is None) and (self.params[param][0] == 'RANGE'):
                for entry in self.params[param][1]:
                    SingleSimSpecs[param] = entry
                    self.AllSimSpecs.append(SingleSimSpecs.copy())

# Sampler = SimSpecsSampler(config.params)
    