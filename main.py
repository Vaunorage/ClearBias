import random
import sqlite3

import numpy as np
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
    min: float
    max: float

    def random_value(self, exclude: Optional[float] = None) -> float:
        while True:
            value = random.uniform(self.min, self.max)
            if value != exclude:
                return value


@dataclass
class Instance:
    original_protected_attrs: dict
    original_unprotected_attrs: dict
    other_attrs: dict
    nb_repeat: int = None
    magnitude: float = None
    epis_uncertainty: float = None
    alea_uncertainty: float = None
    outcome: float = None
    subgroup_id: str = None
    individual_id: str = None
    case: int = None
    diff_outcome: float = None

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
               'case': self.case,
               'alea_uncertainty': self.alea_uncertainty, 'magnitude': self.magnitude,
               'epis_uncertainty': self.epis_uncertainty,
               'granularity': len(self.original_protected_attrs),
               'outcome': self.outcome,
               'diff_outcome': self.diff_outcome}
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

    def __post_init__(self):
        self.Wt = np.random.rand(1, len(self.protected_attributes))
        self.Wx = np.random.rand(1, len(self.unprotected_attributes))

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

    def generate_subgroups(self, nb_protected) -> DiscriminatedSubgroups:
        assert nb_protected <= len(self.protected_attributes)

        random_protected = random.sample(self.protected_attributes, nb_protected)
        num_unprotected = random.randint(1, len(self.unprotected_attributes))
        random_unprotected = random.sample(self.unprotected_attributes, num_unprotected)
        subgroup1 = SubGroup(protected_attr={e.name: e.random_value() for e in random_protected},
                             unprotected_attr={e.name: e.random_value() for e in random_unprotected})
        subgroup2 = SubGroup(
            protected_attr={e.name: e.random_value(exclude=subgroup1.protected_attr[e.name]) for e in random_protected},
            unprotected_attr={e.name: e.random_value(exclude=subgroup1.unprotected_attr[e.name]) for e in
                              random_unprotected}
        )

        discr_group = DiscriminatedSubgroups(subgroup1=subgroup1, subgroup2=subgroup2)

        return discr_group

    def _generate_other_vars_from_subgroup(self, subgroup: SubGroup):
        return {
            attr.name: attr.random_value() for attr in self.attributes if
            attr.name not in subgroup.protected_attr and attr.name not in subgroup.unprotected_attr
        }

    def get_instance_from_discriminated_subgroup(self, subgroup: SubGroup,
                                                 magnitude, epis_uncertainty, alea_uncertainty, case):

        instances = []
        for i in range(alea_uncertainty * 100):
            res = subgroup.protected_attr
            res.update(subgroup.unprotected_attr)
            dgroup_i = hash(frozenset(res.items()))
            res.update(self._generate_other_vars_from_subgroup(subgroup))

            protected_attrs = np.array([res[e.name] for e in self.protected_attributes])
            unprotected_attrs = np.array([res[e.name] for e in self.unprotected_attributes])

            outcome = np.concatenate([(magnitude * self.Wt * protected_attrs), (self.Wx * unprotected_attrs)],
                                     axis=1).sum()
            outcome = random.gauss(outcome, epis_uncertainty)
            instance = Instance(
                original_protected_attrs=subgroup.protected_attr,
                original_unprotected_attrs=subgroup.unprotected_attr,
                other_attrs=self._generate_other_vars_from_subgroup(subgroup),
                outcome=outcome,
                magnitude=magnitude,
                subgroup_id=str(dgroup_i),
                individual_id=f"{dgroup_i}_{i}",
                case=case,
            )
            instances.append(instance)
        return instances

    def generate_instances_from_subgroups(self, num_individuals=100):
        num_individuals = 100
        individuals = []
        for i in range(1, num_individuals + 1):
            alea_uncertainty = random.uniform(0, 1)
            epis_uncertainty = random.uniform(0, 1)
            magnitude = random.uniform(0, 1)

            granularity = random.choice(list(range(1, 1 + len(self._get_attributes('protected')))))
            discriminated_sub_group = self.generate_subgroups(granularity)

            instances1 = self.get_instance_from_discriminated_subgroup(discriminated_sub_group.subgroup1,
                                                                      magnitude, epis_uncertainty, alea_uncertainty, case=1)
            instances2 = self.get_instance_from_discriminated_subgroup(discriminated_sub_group.subgroup2,
                                                                      magnitude, epis_uncertainty, alea_uncertainty, case=2)

        for dgroup_i, dgroup in enumerate(discriminated_sub_group):
            for i in range(1, num_individuals + 1):
                nb_repeat = i
                magnitude = 1 - i / num_individuals
                epis_uncertainty = 1 - i / num_individuals

                instance1_outcome = random.uniform(0, 1)
                instance2_outcome = abs(random.gauss(abs(instance1_outcome - magnitude), epis_uncertainty))

                diff_outcome = abs(instance1_outcome - instance2_outcome)
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
                        individual_id=f"{dgroup_i}_{i}_{el}",
                        case=1,
                        diff_outcome=diff_outcome

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
                        individual_id=f"{dgroup_i}_{i}_{el}",
                        case=2,
                        diff_outcome=diff_outcome
                    )

                    individuals.extend([instance1.to_values(), instance2.to_values()])

        return individuals


attributes = [
    NumericalAttribute(name='Age', type='protected', min=0, max=1),
    CategoricalAttribute(name='Preference', type='protected', p=[0, 0.3, 0.6]),
    NumericalAttribute(name='Income', type='unprotected', min=0, max=1),
    CategoricalAttribute(name='Gender', type='protected', p=[0, 0.3, 0.6]),
    NumericalAttribute(name='Expenditure', type='unprotected', min=0, max=1)
]

schema = DatasetSchema(attributes=attributes)
all_individuals = schema.generate_instances_from_subgroups()
df = pd.DataFrame(all_individuals)

connection = sqlite3.connect('elements.db')
df.to_sql('discriminations', con=connection, if_exists='replace', index=False)
connection.close()

# %%
fig = px.parallel_coordinates(
    df[['granularity', 'alea_uncertainty', 'epis_uncertainty', 'magnitude', 'diff_outcome']].drop_duplicates(),
    color='diff_outcome', labels={"granularity": "granularity",
                                  "alea_uncertainty": "alea_uncertainty",
                                  "epis_uncertainty": "epis_uncertainty",
                                  "magnitude": "magnitude"},
    color_continuous_scale=px.colors.diverging.Tealrose,
    color_continuous_midpoint=3)
fig.write_image("parallel_coordinates_plot1.png")
print('done')
# %%
# print(df.head(5).to_string())
#
# # Create the 3D scatter plot
# fig = px.scatter_3d(df, x='magnitude', y='epis_uncertainty', z='alea_uncertainty', size='granularity',
#                     size_max=18, hover_data=['individual_id'])
#
# # Create a Dash app
# app = Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
#
# # Define the layout with a single graph
# app.layout = html.Div([
#     dcc.Graph(figure=fig)
# ])
#
# # Save the figure as an HTML file
# fig.write_html("3Dscatter.html")

print("ddd")
