import copy
import random
import sqlite3
from uuid import uuid4

import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Literal, List, Optional
import plotly.express as px
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier


@dataclass
class Attribute:
    name: str
    type: Literal['protected', 'unprotected']

    def random_value(self):
        raise NotImplementedError()


@dataclass
class CategoricalAttribute(Attribute):
    p: List[float]

    def random_value(self, exclude: Optional[int] = None) -> int:
        available_choices = [x for x in self.p if x != exclude]
        if not available_choices:
            raise ValueError("No available choices remaining to select from.")
        return random.choice(available_choices)


@dataclass
class NumericalAttribute(Attribute):
    min: float = 0.0
    max: float = 1.0

    def random_value(self, exclude: Optional[float] = None) -> float:
        while True:
            value = random.uniform(self.min, self.max)
            if value != exclude:
                return value


@dataclass
class ContinuousOutcome:
    name: str

    def from_cont_val(self, val: np.array):
        return (val - val.min()) / (val.max() - val.min())


@dataclass
class CategoricalOutcome:
    name: str
    nb_categ: int

    def from_cont_val(self, val: np.array):
        val = (val - val.min()) / (val.max() - val.min())
        bins = np.linspace(0, 1, self.nb_categ + 1)
        categories = pd.cut(val, bins, include_lowest=True, labels=False)
        return categories


@dataclass
class Instance:
    original_protected_attrs: dict
    original_unprotected_attrs: dict
    other_attrs: dict
    granularity: int = None
    magnitude: float = None
    epis_uncertainty: float = None
    alea_uncertainty: float = None
    outcome: float = None
    subgroup_id: str = None
    individual_id: str = None
    case: int = None
    processed_outcomes: dict = None

    @property
    def nb_protected_attr(self):
        return len(self.original_protected_attrs)

    @property
    def group_id(self):
        return hash(frozenset(self.original_protected_attrs.items()))

    def to_values(self):
        res = {'subgroup_id': self.subgroup_id, 'individual_id': self.individual_id,
               'case': self.case,
               'alea_uncertainty': self.alea_uncertainty, 'magnitude': self.magnitude,
               'epis_uncertainty': self.epis_uncertainty,
               'granularity': self.granularity,
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
    features: List[Attribute]

    def __post_init__(self):
        self.Wt = np.random.rand(1, len(self.protected_attributes))
        self.Wx = np.random.rand(1, len(self.unprotected_attributes))

    @property
    def attributes(self):
        return [a for a in self.features if isinstance(a, Attribute)]

    def _get_attributes(self, type):
        return [attr for attr in self.attributes if attr.type == type]

    @property
    def protected_attributes(self) -> List[Attribute]:
        return self._get_attributes('protected')

    @property
    def unprotected_attributes(self) -> List[Attribute]:
        return self._get_attributes('unprotected')

    @property
    def outcomes(self) -> List[ContinuousOutcome | CategoricalOutcome]:
        return [e for e in self.features if isinstance(e, ContinuousOutcome) or isinstance(e, CategoricalOutcome)]

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
                                                 subgroup_id, magnitude, granularity,
                                                 epis_uncertainty, alea_uncertainty, nb_individuals, case):

        instances = []
        nb_repeat = int(alea_uncertainty * nb_individuals * 0.1)
        nb_repeat = nb_repeat if nb_repeat > 0 else 1
        for i in range(nb_repeat):
            res = copy.deepcopy(subgroup.protected_attr)
            res.update(subgroup.unprotected_attr)
            res.update(self._generate_other_vars_from_subgroup(subgroup))

            protected_attrs = np.array([res[e.name] for e in self.protected_attributes])
            unprotected_attrs = np.array([res[e.name] for e in self.unprotected_attributes])

            sigmoid = lambda z: 1 / (1 + np.exp(-z))
            outcome = np.concatenate([((1 + magnitude) * self.Wt * protected_attrs), (self.Wx * unprotected_attrs)],
                                     axis=1).sum()
            outcome = random.gauss(outcome, epis_uncertainty)
            outcome = sigmoid(outcome)

            instance = Instance(
                original_protected_attrs=subgroup.protected_attr,
                original_unprotected_attrs=subgroup.unprotected_attr,
                other_attrs=self._generate_other_vars_from_subgroup(subgroup),
                outcome=outcome,
                magnitude=magnitude,
                granularity=granularity,
                alea_uncertainty=alea_uncertainty,
                epis_uncertainty=epis_uncertainty,
                subgroup_id=subgroup_id,
                individual_id=str(i),
                case=case
            )
            instances.append(instance)
        return instances

    def generate_instances_from_subgroups(self, num_individuals=100):
        individuals = []
        for i in range(1, num_individuals + 1):
            alea_uncertainty = random.uniform(0.01, 1)
            epis_uncertainty = random.uniform(0, 1)
            magnitude = random.uniform(0, 1)

            granularity = random.choice(list(range(1, 1 + len(self._get_attributes('protected')))))
            discriminated_sub_group = self.generate_subgroups(granularity)
            subgroup_id = str(uuid4())
            instances1 = self.get_instance_from_discriminated_subgroup(discriminated_sub_group.subgroup1,
                                                                       subgroup_id,
                                                                       magnitude, granularity, epis_uncertainty,
                                                                       alea_uncertainty,
                                                                       num_individuals,
                                                                       case=1)
            instances2 = self.get_instance_from_discriminated_subgroup(discriminated_sub_group.subgroup2,
                                                                       subgroup_id,
                                                                       magnitude, granularity, epis_uncertainty,
                                                                       alea_uncertainty,
                                                                       num_individuals,
                                                                       case=2)

            df_temp = pd.DataFrame([e.to_values() for e in instances1 + instances2])
            df_temp.sort_values(by=['subgroup_id', 'individual_id', 'case'], inplace=True)
            for e in self.outcomes:
                df_temp[f'outcome_{e.name}'] = e.from_cont_val(df_temp[f'outcome'].to_numpy())
                df_temp[f'diff_outcome_{e.name}_ind'] = df_temp.groupby(['subgroup_id', 'individual_id'])[
                    f'outcome_{e.name}'].diff().abs().bfill()
                df_temp[f'diff_outcome_{e.name}'] = df_temp[f'diff_outcome_{e.name}_ind'].mean()

            df_temp['diff_outcome_ind'] = df_temp.groupby(['subgroup_id', 'individual_id'])[
                'outcome'].diff().abs().bfill()
            df_temp['diff_outcome'] = df_temp['diff_outcome_ind'].mean()
            individuals.append(df_temp)

        individuals = pd.concat(individuals)

        for e in self.attributes:
            if isinstance(e, CategoricalAttribute):
                bins = np.linspace(0, 1, len(e.p) + 1)
                individuals[e.name] = pd.cut(individuals[e.name].to_numpy(), bins, include_lowest=True, labels=False)

        return individuals


attributes = [
    NumericalAttribute(name='Age', type='protected'),
    CategoricalAttribute(name='Preference', type='unprotected', p=[0, 0.3, 0.6]),
    NumericalAttribute(name='Income', type='unprotected'),
    CategoricalAttribute(name='Gender', type='protected', p=[0, 0.3, 0.6]),
    NumericalAttribute(name='Expenditure', type='unprotected'),
    CategoricalAttribute(name='HairColor', type='unprotected', p=[0.25, 0.50, 0.75, 1.0]),
    ContinuousOutcome(name='out1'),
    CategoricalOutcome(name='out2', nb_categ=2)
]

schema = DatasetSchema(features=attributes)
df = schema.generate_instances_from_subgroups(100)
df = df.reset_index()
connection = sqlite3.connect('elements.db')
df.to_sql('discriminations', con=connection, if_exists='replace', index=False)
connection.close()

df[['HairColor', 'Preference', 'Gender', 'outcome_out2']].to_csv(
    "/home/vaunorage/PycharmProjects/Aequitas1/Phemus/Examples/Synthetic/discriminations.csv", index=False)


model = RandomForestClassifier(n_estimators = 10)

# %%
gg = df[['granularity', 'alea_uncertainty', 'epis_uncertainty', 'magnitude', 'diff_outcome']]. \
    drop_duplicates().sort_values(['magnitude']).reset_index().astype(float).drop(columns=['index'])

fig = px.parallel_coordinates(
    gg,
    color="diff_outcome",
    labels={
        "granularity": "granularity",
        "alea_uncertainty": "alea_uncertainty",
        "epis_uncertainty": "epis_uncertainty",
        "magnitude": "magnitude",
        "diff_outcome": "diff_outcome"
    },
    color_continuous_scale=px.colors.diverging.Tealrose,
    color_continuous_midpoint=gg['diff_outcome'].max() / 2)

fig.update_layout(coloraxis_showscale=True)

fig.write_image("figure3.png")

print("ddd")


# %%
def scale_dataframe(df, reverse=False, min_values=None, max_values=None):
    if not reverse:
        min_values = df.min()
        max_values = df.max()
        scaled_df = (df - min_values) / (max_values - min_values)
        return scaled_df, min_values, max_values
    else:
        if min_values is None or max_values is None:
            raise ValueError("min_values and max_values must be provided to reverse scaling.")
        original_df = df * (max_values - min_values) + min_values
        return original_df


df['subgroup_id'] = df['subgroup_id'].replace(
    {e: k for k, e in enumerate(df['subgroup_id'].drop_duplicates().tolist())})

dff = df[
    ['granularity', 'alea_uncertainty', 'epis_uncertainty', 'magnitude', 'Preference', 'Age', 'Expenditure', 'Income',
     'Gender', 'diff_outcome', 'subgroup_id']].astype(float).drop_duplicates()
scaled_df, min_values, max_values = scale_dataframe(dff)

original_df = scale_dataframe(scaled_df, reverse=True, min_values=min_values, max_values=max_values)

scaled_df_attr = scaled_df[['Preference', 'Age', 'Expenditure', 'Income', 'Gender']]
scaled_df_meta = scaled_df[['granularity', 'alea_uncertainty', 'epis_uncertainty', 'magnitude']]


def embd_to_1_dim(df):
    vec = np.arange(1, df.shape[1] + 1)
    ll = df.to_numpy().dot(vec)
    minll, maxll = np.full_like(vec, 0).dot(vec), np.full_like(vec, 1).dot(vec)
    res = (ll - minll) / (maxll - minll)
    res = np.concatenate([res, np.array([0, 1])])
    return res


embd_attr = embd_to_1_dim(scaled_df_attr)
embd_meta = embd_to_1_dim(scaled_df_meta)
outcome = np.concatenate([dff['diff_outcome'].to_numpy(), [dff['diff_outcome'].min(), dff['diff_outcome'].max()]])

plt.clf()

plt.scatter(embd_attr, embd_meta, c=outcome, cmap='viridis', s=1)

plt.colorbar(label='Values')

plt.xlabel('Attributes')
plt.ylabel('Metadata')

plt.savefig("figure5.png")
