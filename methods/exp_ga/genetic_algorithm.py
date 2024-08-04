import numpy as np
import pandas as pd


class GA:
    def __init__(self, nums, bound, func, DNA_SIZE=None, cross_rate=0.8, mutation=0.003):
        self.nums = np.array(nums)
        self.bound = np.array(bound)
        self.func = func
        self.cross_rate = cross_rate
        self.mutation = mutation

        self.min_nums, self.max_nums = self.bound[:, 0], self.bound[:, 1]
        self.var_len = self.max_nums - self.min_nums

        if DNA_SIZE is None:
            self.DNA_SIZE = int(np.ceil(np.max(np.log2(self.var_len + 1))))
        else:
            self.DNA_SIZE = DNA_SIZE

        self.POP_SIZE = len(nums)
        self.POP = self.nums.copy()
        self.copy_POP = self.nums.copy()

    def get_fitness(self, non_negative=False):
        result = [self.func(individual) for individual in self.POP]
        if non_negative:
            result -= np.min(result)
        return result

    def select(self):
        fitness = self.get_fitness()
        probabilities = fitness / np.sum(fitness)
        if not np.isclose(np.sum(probabilities), 1):
            probabilities = probabilities / np.sum(probabilities)  # Normalize to ensure sum is 1
        self.POP = self.POP[
            np.random.choice(np.arange(self.POP_SIZE), size=self.POP_SIZE, replace=True, p=list(probabilities.squeeze()))]

    def crossover(self):
        for i in range(self.POP_SIZE):
            if np.random.rand() < self.cross_rate:
                partner_idx = np.random.randint(0, self.POP_SIZE)
                cross_points = np.random.randint(0, len(self.bound))
                end_points = np.random.randint(cross_points, len(self.bound))
                self.POP[i][cross_points:end_points] = self.POP[partner_idx][cross_points:end_points]

    def mutate(self):
        for individual in self.POP:
            for gene in range(self.DNA_SIZE):
                if np.random.rand() < self.mutation:
                    individual[0][gene] = np.random.randint(self.bound[gene][0], self.bound[gene][1])

    def evolve(self):
        self.select()
        self.crossover()
        self.mutate()

    def reset(self):
        self.POP = self.copy_POP.copy()

    def log(self):
        fitness = self.get_fitness()
        population_log = ["".join(map(str, individual)) for individual in self.POP]
        return population_log, fitness

# Usage example (assuming a defined fitness function `fitness_func`):
# nums = [[...], [...], ...]  # Define your population here
# bounds = [(min, max), (min, max), ...]  # Define your bounds here
# ga = GeneticAlgorithm(nums, bounds, fitness_func)
# ga.evolve()


# if __name__ == '__main__':
#     nums = [[3,0,10,3,1,6,3,0,1,0,0,40,0],[4,3,20,13,2,5,3,0,0,0,0,50,0],[3,0,14,1,0,4,2,4,1,0,0,80,0],[5,0,5,3,1,0,5,0,0,0,0,40,0]]
#     bound = config.input_bounds
#     # func = lambda x, y: x*np.cos(2*np.pi*y)+y*np.sin(2*np.pi*x)
#     DNA_SIZE = len(bound)
#     cross_rate = 0.7
#     mutation = 0.01
#     ga = GA(nums=nums, bound=bound, func=evaluate_local, DNA_SIZE=DNA_SIZE, cross_rate=cross_rate, mutation=mutation)
#     res = ga.log()
#     print(res)
#     for i in range(10):
#         ga.evolution()
#         res = ga.log()
#         print(res)