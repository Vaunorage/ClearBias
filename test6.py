from sdv.datasets.demo import download_demo

real_data, metadata = download_demo(
    modality='single_table',
    dataset_name='fake_hotel_guests')

#%%
from sdv.single_table import GaussianCopulaSynthesizer

synthesizer = GaussianCopulaSynthesizer(metadata)
synthesizer.fit(data=real_data)

#%%
synthetic_data = synthesizer.sample(num_rows=500)

#%%
from sdv.evaluation.single_table import evaluate_quality

quality_report = evaluate_quality(
    real_data,
    synthetic_data,
    metadata)

#%%
ll = quality_report.get_details(property_name='Column Shapes')
ll2 = quality_report.get_details(property_name='Column Pair Trends')