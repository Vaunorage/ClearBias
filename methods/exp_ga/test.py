from data_generator.main import generate_data
from methods.exp_ga.algo import run_expga

ge = generate_data(min_number_of_classes=2, max_number_of_classes=6, nb_attributes=6,
                   prop_protected_attr=0.3, nb_groups=500, max_group_size=50, hiddenlayers_depth=3,
                   min_similarity=0.0, max_similarity=1.0, min_alea_uncertainty=0.0, max_alea_uncertainty=1.0,
                   min_epis_uncertainty=0.0, max_epis_uncertainty=1.0, min_magnitude=0.0, max_magnitude=1.0,
                   min_frequency=0.0, max_frequency=1.0, categorical_outcome=True, nb_categories_outcome=4)

dd = run_expga(ge, threshold=0.5, threshold_rank=0.5, max_global=50, max_local=50)

print('dd')
