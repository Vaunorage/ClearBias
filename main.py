import random
from dataclasses import dataclass
from typing import Literal, List, Optional


@dataclass
class Attribute:
    name: str
    type: Literal['protected', 'unprotected']

    def generate_value(self):
        raise NotImplementedError()


@dataclass
class CategoricalAttribute(Attribute):
    p: List[int]

    def random_value(self):
        return random.choice(self.p)


@dataclass
class NumericalAttribute(Attribute):
    mean: float
    std: float

    def random_value(self):
        return random.gauss(self.mean, self.std)


@dataclass
class DatasetRow:
    original_protected_attrs: dict
    original_unprotected_attrs: dict
    other_attrs: dict
    nb_replications: int

    @property
    def nb_protected_attr(self):
        return len(self.original_protected_attrs)

    @property
    def group_id(self):
        return hash(frozenset(self.original_protected_attrs.items()))

    def props(self):
        return {'group_id': self.group_id, 'replications': self.nb_replications}

    def to_values(self):
        res = self.original_protected_attrs.values()
        res += self.original_unprotected_attrs.values()
        res += self.other_attrs.values()
        return res


@dataclass
class SubGroup:
    protected_attr: dict
    unprotected_attr: dict


@dataclass
class DiscriminatedSubgroups:
    subgroup1: SubGroup
    subgroup2: SubGroup


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

    def __getitem__(self, item_name: str) -> Optional[Attribute]:
        for attr in self.attributes:
            if attr.name == item_name:
                return attr
        return None


# %%
def generate_subgroups(schema: DatasetSchema) -> List[List[Attribute]]:
    protected = schema.protected_attributes
    unprotected = schema.unprotected_attributes
    all_subgroups = []

    random_protected = random.sample(protected, len(protected))

    for i in range(1, len(random_protected) + 1):
        subgroup = random_protected[:i]
        num_unprotected = random.randint(1, len(unprotected))
        random_unprotected = random.sample(unprotected, num_unprotected)
        combined_group = subgroup + random_unprotected
        ll = [e.random_value() for e in combined_group]
        all_subgroups.append(combined_group)

    return all_subgroups


def generate_individuals(group, num_individuals, schema: DatasetSchema):
    individuals = []
    for i in range(1, num_individuals + 1):
        base_individual = {attr.name: generate_value(attr) for attr in group}
        for replication_id in range(i):
            varied_individual = {
                attr.name: generate_value(attr) for attr in schema.attributes if attr.name not in base_individual
            }
            dataset_row = DatasetRow(
                original_protected_attrs={attr: base_individual[attr] for attr in base_individual if
                                          schema[attr].type == 'protected'},
                original_unprotected_attrs={attr: base_individual[attr] for attr in base_individual if
                                            schema[attr].type == 'unprotected'},
                other_attrs=varied_individual,
                nb_replications=i
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

print(f"Total individuals generated: {len(all_individuals)}")
for individual in all_individuals[:10]:  # Print first 10 for brevity
    print(individual)
