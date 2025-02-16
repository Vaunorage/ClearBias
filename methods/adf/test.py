from data_generator.main import get_real_data
from methods.adf.main1 import adf_fairness_testing

#%%
ge, ge_schema = get_real_data('adult')

results_df, metrics = adf_fairness_testing(
    ge,
    max_global=100,
    max_local=1000,
    max_iter=10,
    cluster_num=4
)

print("\nTesting Metrics:")
for metric, value in metrics.items():
    print(f"{metric}: {value:.4f}")
