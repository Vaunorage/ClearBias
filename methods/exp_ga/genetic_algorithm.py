import numpy as np
import pandas as pd


class GA:
    def __init__(self, nums, bound, func, DNA_SIZE=None, cross_rate=0.8, mutation=0.003):
        # Convert input nums to proper numpy array shape
        self.nums = np.array(nums, dtype=np.int32)
        if len(self.nums.shape) == 2:
            self.nums = self.nums.reshape(len(nums), -1)

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

        # Ensure POP has correct shape
        if len(self.POP.shape) == 1:
            self.POP = self.POP.reshape(self.POP_SIZE, -1)
            self.copy_POP = self.copy_POP.reshape(self.POP_SIZE, -1)

    def get_fitness(self, non_negative=False):
        result = np.array([self.func(individual) for individual in self.POP])
        if non_negative:
            result = result - np.min(result)
        return result

    def select(self):
        fitness = self.get_fitness()

        if len(fitness) == 0:
            raise ValueError("Fitness array is empty.")

        total_fitness = np.sum(fitness)
        if total_fitness == 0:
            probabilities = np.ones(len(fitness)) / len(fitness)
        else:
            probabilities = fitness / total_fitness

        probabilities = probabilities.squeeze()
        if probabilities.ndim == 0:
            probabilities = np.array([probabilities])

        probabilities = probabilities / np.sum(probabilities)

        selected_indices = np.random.choice(np.arange(self.POP_SIZE), size=self.POP_SIZE, replace=True, p=probabilities)
        self.POP = self.POP[selected_indices]

    def crossover(self):
        for i in range(self.POP_SIZE):
            if np.random.rand() < self.cross_rate:
                partner_idx = np.random.randint(0, self.POP_SIZE)

                # Ensure proper array shapes for crossover
                if len(self.POP[i].shape) == 1 and len(self.POP[partner_idx].shape) == 1:
                    cross_points = np.random.randint(0, self.POP[i].size)
                    end_points = np.random.randint(cross_points + 1, self.POP[i].size + 1)

                    # Create copies to avoid modifying original arrays
                    temp = self.POP[i].copy()
                    temp[cross_points:end_points] = self.POP[partner_idx][cross_points:end_points]
                    self.POP[i] = temp

    def mutate(self):
        for i in range(len(self.POP)):
            individual = self.POP[i].flatten()  # Ensure 1D array for mutation

            for gene_idx in range(individual.size):
                if np.random.rand() < self.mutation:
                    if gene_idx < len(self.bound):  # Check if we have bounds for this gene
                        low = int(self.bound[gene_idx][0])
                        high = int(self.bound[gene_idx][1])

                        if high > low:
                            new_value = np.random.randint(low, high + 1)
                            individual[gene_idx] = new_value

            # Reshape back if necessary and update population
            self.POP[i] = individual.reshape(self.POP[i].shape)

    def evolve(self):
        self.select()
        self.crossover()
        self.mutate()

    def reset(self):
        self.POP = self.copy_POP.copy()

    def log(self):
        fitness = self.get_fitness()
        population_log = [ind.flatten().tolist() for ind in self.POP]
        return population_log, fitness.tolist()


# Example usage:
if __name__ == '__main__':
    # Example input data
    nums = [
        [3, 0, 10, 3, 1, 6, 3, 0, 1, 0, 0, 40, 0],
        [4, 3, 20, 13, 2, 5, 3, 0, 0, 0, 0, 50, 0],
        [3, 0, 14, 1, 0, 4, 2, 4, 1, 0, 0, 80, 0],
        [5, 0, 5, 3, 1, 0, 5, 0, 0, 0, 0, 40, 0]
    ]

    # Example bounds
    bound = [[0, 10] for _ in range(13)]


    # Example fitness function
    def evaluate_local(x):
        return sum(x)


    # Initialize GA
    ga = GA(
        nums=nums,
        bound=bound,
        func=evaluate_local,
        DNA_SIZE=len(bound),
        cross_rate=0.7,
        mutation=0.01
    )

    # Run evolution
    try:
        for generation in range(10):
            ga.evolve()
            population, fitness = ga.log()
            print(f"Generation {generation + 1}:")
            print(f"Population: {population}")
            print(f"Fitness: {fitness}\n")
    except Exception as e:
        print(f"Error occurred: {str(e)}")
        print(f"Current population shape: {ga.POP.shape}")