import random

def generate_categorical_distribution(categories, skewness):
    n = len(categories)
    raw_probs = [skewness**(i-1) for i in range(1, n+1)]
    normalization_factor = sum(raw_probs)
    normalized_probs = [p / normalization_factor for p in raw_probs]
    return dict(zip(categories, normalized_probs))

def generate_samples(categories, skewness, num_samples):
    # Get the distribution
    distribution = generate_categorical_distribution(categories, skewness)
    # Extract categories and their probabilities
    categories, probabilities = zip(*distribution.items())
    # Generate samples
    samples = random.choices(categories, weights=probabilities, k=num_samples)
    return samples

# Example usage
categories = ['A', 'B', 'C', 'D']
skewness = 0.5  # Change this value to adjust skewness
num_samples = 100  # Number of samples to generate

samples = generate_samples(categories, skewness, num_samples)
print(samples)
