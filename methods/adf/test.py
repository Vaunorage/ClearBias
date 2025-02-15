from data_generator.main import get_real_data
from methods.adf.main import dnn_fair_testing

#%%
ge, ge_schema = get_real_data('adult')

discriminatory_df, metrics = dnn_fair_testing(ge=ge, max_tsn=500, max_global=1000,
                                              max_local=100, max_iter=1000)

print(discriminatory_df)