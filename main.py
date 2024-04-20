import random
from dataclasses import dataclass
from typing import Literal, List


@dataclass
class Attribute:
    name: str
    type: Literal['protected', 'unprotected']


@dataclass
class CategoricalAttribute(Attribute):
    p: List[int]


@dataclass
class NumericalAttribute(Attribute):
    mean: float
    std: float


@dataclass
class DatasetRow:
    original_protected_attrs: dict
    original_unprotected_attrs: dict
    other_attrs: dict
    individual_repl_id: int


@dataclass
class DatasetSchema:
    attributes: List[Attribute]

    def _get_attributes(self, type):
        return [attr for attr in self.attributes if attr.type == type]

    @property
    def protected_attributes(self) -> List[Attribute]:
        return self._get_attributes('protected')

    @property
    def unprotected_attributes(self) -> List[Attribute]:
        return self._get_attributes('unprotected')


# %%
def generate_subgroups(schema: DatasetSchema, max_group_size: int = None) -> List[List[Attribute]]:
    protected = schema.protected_attributes
    unprotected = schema.unprotected_attributes
    all_subgroups = []

    random_protected = random.sample(protected, len(protected))

    for i in range(1, len(random_protected) + 1):
        subgroup = random_protected[:i]
        num_unprotected = random.randint(1, len(unprotected))
        random_unprotected = random.sample(unprotected, num_unprotected)
        combined_group = subgroup + random_unprotected
        all_subgroups.append(combined_group)

    return all_subgroups


def generate_individual(attribute):
    if isinstance(attribute, CategoricalAttribute):
        return random.choice(attribute.p)
    elif isinstance(attribute, NumericalAttribute):
        return random.gauss(attribute.mean, attribute.std)


def generate_individuals(group, num_individuals, shema: DatasetSchema):
    individuals = []
    for i in range(1, num_individuals + 1):
        base_individual = {attr.name: generate_individual(attr) for attr in group}
        for replication_id in range(i):
            varied_individual = {
                attr.name: generate_individual(attr) for attr in shema.attributes if attr not in base_individual
            }
            dataset_row = DatasetRow(
                original_protected_attrs={attr.name: base_individual[attr.name] for attr in shema.attributes if
                                          attr.type == 'protected'},
                original_unprotected_attrs={attr.name: base_individual[attr.name] for attr in shema.attributes if
                                            attr.type == 'unprotected'},
                other_attrs=varied_individual,
                individual_repl_id=replication_id + 1
            )
            individuals.append(dataset_row)
    return individuals


attributes = [
    CategoricalAttribute(name='Age', type='protected', p=[20, 30, 40]),
    NumericalAttribute(name='Income', type='unprotected', mean=50000, std=10000),
    CategoricalAttribute(name='Gender', type='protected', p=[0, 1]),
    NumericalAttribute(name='Expenditure', type='unprotected', mean=2000, std=500)
]

schema = DatasetSchema(attributes=attributes)
subgroups = generate_subgroups(schema)

all_individuals = []
for group in subgroups:
    individuals = generate_individuals(group, 100, schema)
    all_individuals.extend(individuals)

# Optionally, print out some of the individual entries for verification
print(f"Total individuals generated: {len(all_individuals)}")
for individual in all_individuals[:10]:  # Print first 10 for brevity
    print(individual)
