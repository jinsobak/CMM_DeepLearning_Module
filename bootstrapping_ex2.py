from sklearn.utils import resample 
import numpy as np

# Generate some example data
np.random.seed(2023)
original_data = np.random.normal(loc=10, scale=2, size=100)

# Number of bootstrap samples
num_samples = 1000

# Perform bootstrap resampling
bootstrap_samples = []
for _ in range(num_samples):
    resampled_data = resample(original_data) # replace = True: default
    bootstrap_samples.append(resampled_data)

# Now 'bootstrap_samples' contains 1000 bootstrap samples

# You can then use these samples to compute confidence intervals or other statistics
# For example, let's compute the mean and 95% confidence interval 
means = np.mean(bootstrap_samples, axis=1)
confidence_interval = np.percentile(means, [2.5, 97.5])

print("Original data mean:", np.mean(original_data))
print("Bootstrap resampled means 95% CI:", confidence_interval)

"""결과값
Original data mean: 9.901016443098726
Bootstrap resampled means 95% CI: [ 9.45744398 10.32271447]"""