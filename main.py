import random
import pandas as pd
from dataclasses import dataclass
from typing import Literal, List, Optional
import plotly.express as px
from dash import Dash, dcc, html
import dash_bootstrap_components as dbc


@dataclass
class Attribute:
    name: str
    type: Literal['protected', 'unprotected']

    def random_value(self):
        raise NotImplementedError()


@dataclass
class CategoricalAttribute(Attribute):
    p: List[int]

    def random_value(self, exclude: Optional[int] = None) -> int:
        available_choices = [x for x in self.p if x != exclude]
        if not available_choices:
            raise ValueError("No available choices remaining to select from.")
        return random.choice(available_choices)


@dataclass
class NumericalAttribute(Attribute):
    mean: float
    std: float

    def random_value(self, exclude: Optional[float] = None) -> float:
        while True:
            value = random.gauss(self.mean, self.std)
            if value != exclude:
                return value


@dataclass
class Instance:
    original_protected_attrs: dict
    original_unprotected_attrs: dict
    other_attrs: dict
    nb_repeat: int
    magnitude: float
    epis_uncertainty: float
    alea_uncertainty: float
    outcome: float
    subgroup_id: str
    individual_id: str

    @property
    def nb_protected_attr(self):
        return len(self.original_protected_attrs)

    @property
    def group_id(self):
        return hash(frozenset(self.original_protected_attrs.items()))

    def props(self):
        return {'group_id': self.group_id, 'replications': self.nb_repeat}

    def to_values(self):
        res = {'subgroup_id': self.subgroup_id, 'individual_id': self.individual_id,
               'alea_uncertainty': self.alea_uncertainty, 'magnitude': self.magnitude,
               'epis_uncertainty': self.epis_uncertainty,
               'granularity': len(self.original_protected_attrs),
               'outcome': self.outcome}
        res.update(self.original_protected_attrs)
        res.update(self.original_unprotected_attrs)
        res.update(self.other_attrs)
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

    @property
    def outcomes(self) -> List[Attribute]:
        return self._get_attributes('outcome')

    def __getitem__(self, item_name: str) -> Optional[Attribute]:
        for attr in self.attributes:
            if attr.name == item_name:
                return attr
        return None

    def generate_subgroups(self) -> List[DiscriminatedSubgroups]:
        protected = self.protected_attributes
        unprotected = self.unprotected_attributes
        all_subgroups = []

        random_protected = random.sample(protected, len(protected))

        for i in range(1, len(random_protected) + 1):
            subgroup = random_protected[:i]
            num_unprotected = random.randint(1, len(unprotected))
            random_unprotected = random.sample(unprotected, num_unprotected)
            subgroup1 = SubGroup(protected_attr={e.name: e.random_value() for e in subgroup},
                                 unprotected_attr={e.name: e.random_value() for e in random_unprotected})
            subgroup2 = SubGroup(
                protected_attr={e.name: e.random_value(exclude=subgroup1.protected_attr[e.name]) for e in subgroup},
                unprotected_attr={e.name: e.random_value(exclude=subgroup1.unprotected_attr[e.name]) for e in
                                  random_unprotected}
            )

            discr_group = DiscriminatedSubgroups(subgroup1=subgroup1, subgroup2=subgroup2)
            all_subgroups.append(discr_group)

        return all_subgroups

    def generate_instances_from_subgroups(self, discriminated_sub_group: List[DiscriminatedSubgroups]):
        num_individuals = 100
        individuals = []
        for dgroup_i, dgroup in enumerate(discriminated_sub_group):
            for i in range(1, num_individuals + 1):
                nb_repeat = i
                magnitude = 1 - i / num_individuals
                epis_uncertainty = 1 - i / num_individuals

                instance1_outcome = random.uniform(0, 1)
                instance2_outcome = abs(random.gauss(abs(instance1_outcome - magnitude), epis_uncertainty))

                for el in range(nb_repeat):
                    instance1 = Instance(
                        original_protected_attrs=dgroup.subgroup1.protected_attr,
                        original_unprotected_attrs=dgroup.subgroup1.unprotected_attr,
                        other_attrs={
                            attr.name: attr.random_value() for attr in self.attributes if
                            attr.name not in dgroup.subgroup1.protected_attr and attr.name not in dgroup.subgroup1.unprotected_attr
                        },
                        nb_repeat=i,
                        magnitude=magnitude,
                        epis_uncertainty=epis_uncertainty,
                        alea_uncertainty=i / num_individuals,
                        outcome=instance1_outcome,
                        subgroup_id=str(dgroup_i),
                        individual_id=f"{dgroup_i}_{i}_{el}"

                    )

                    instance2 = Instance(
                        original_protected_attrs=dgroup.subgroup2.protected_attr,
                        original_unprotected_attrs=dgroup.subgroup2.unprotected_attr,
                        other_attrs={
                            attr.name: attr.random_value() for attr in self.attributes if
                            attr.name not in dgroup.subgroup2.protected_attr and attr.name not in dgroup.subgroup2.unprotected_attr
                        },
                        nb_repeat=i,
                        magnitude=magnitude,
                        epis_uncertainty=epis_uncertainty,
                        alea_uncertainty=i / num_individuals,
                        outcome=instance2_outcome,
                        subgroup_id=str(dgroup_i),
                        individual_id=f"{dgroup_i}_{i}_{el}"
                    )

                    individuals.extend([instance1.to_values(), instance2.to_values()])

        return individuals


# %%
attributes = [
    CategoricalAttribute(name='Age', type='protected', p=[20, 30, 40]),
    NumericalAttribute(name='Income', type='unprotected', mean=50000, std=10000),
    CategoricalAttribute(name='Gender', type='protected', p=[0, 1]),
    NumericalAttribute(name='Expenditure', type='unprotected', mean=2000, std=500)
]

schema = DatasetSchema(attributes=attributes)
subgroups = schema.generate_subgroups()
all_individuals = schema.generate_instances_from_subgroups(subgroups)
df = pd.DataFrame(all_individuals)
print(df.head(5).to_string())

# Create the 3D scatter plot
fig = px.scatter_3d(df, x='magnitude', y='epis_uncertainty', z='alea_uncertainty', size='granularity',
                    size_max=18, hover_data=['individual_id'])

# Create a Dash app
app = Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

# Define the layout with a single graph
app.layout = html.Div([
    dcc.Graph(figure=fig)
])

# Save the figure as an HTML file
fig.write_html("3Dscatter.html")

print("ddd")
