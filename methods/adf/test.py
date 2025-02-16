from data_generator.main import get_real_data, generate_from_real_data
from methods.adf.main import dnn_fair_testing
from methods.adf.main1 import adf_fairness_testing

#%%
ge, ge_schema = get_real_data('adult')

discriminatory_df, metrics = dnn_fair_testing(ge=ge, max_tsn=500, max_global=1000,
                                              max_local=100, max_iter=1000)

print(discriminatory_df)
#%%
# Generate synthetic data
data_obj, schema = generate_from_real_data('adult')
# data_obj, schema = get_real_data('adult')

# Run fairness testing
results_df, metrics = adf_fairness_testing(
    data_obj, max_global=100, max_local=1000, max_iter=10, cluster_num=4
)

print("\nTesting Metrics:")
for metric, value in metrics.items():
    print(f"{metric}: {value:.4f}")

#%%
# data_obj, schema = generate_from_real_data('adult')
data_obj, schema = get_real_data('adult')

# Run fairness testing
results_df, metrics = adf_fairness_testing(
    data_obj, max_global=100, max_local=1000, max_iter=10, cluster_num=4
)

print("\nTesting Metrics:")
for metric, value in metrics.items():
    print(f"{metric}: {value:.4f}")