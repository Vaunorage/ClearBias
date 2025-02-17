from data_generator.main import get_real_data, generate_from_real_data, generate_data
from methods.adf.main1 import adf_fairness_testing
from methods.utils import reformat_discrimination_results

# %%
# Generate synthetic data

data_obj, schema = generate_from_real_data('adult')
# data_obj, schema = get_real_data('adult')

# Run fairness testing
results_df, metrics = adf_fairness_testing(
    data_obj,
    max_global=5000,
    max_local=2000,
    max_iter=5,
    cluster_num=50
)

#%%
res = reformat_discrimination_results(results_df)

#%%
res_dd = generate_data(group_info_df=res)

print("\nTesting Metrics:")
for metric, value in metrics.items():
    print(f"{metric}: {value:.4f}")

# %%
# data_obj, schema = generate_from_real_data('adult')
data_obj, schema = get_real_data('adult')

# Run fairness testing
results_df, metrics = adf_fairness_testing(
    data_obj,
    max_global=5000,
    max_local=2000,
    max_iter=6,
    cluster_num=50
)

print("\nTesting Metrics:")
for metric, value in metrics.items():
    print(f"{metric}: {value:.4f}")
