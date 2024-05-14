import sqlite3
import numpy as np
import pandas as pd
import plotly.express as px
import matplotlib.pyplot as plt

connection = sqlite3.connect('elements.db')

# SQL query to select all columns from the table
query = "SELECT * FROM discriminations"

df = pd.read_sql_query(query, connection)

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
