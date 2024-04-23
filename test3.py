import sqlite3

import pandas as pd

connection = sqlite3.connect('elements.db')

# SQL query to select all columns from the table
query = "SELECT * FROM discriminations"

df = pd.read_sql_query(query, connection)

# %%
ll = df[['magnitude', 'diff_outcome']].drop_duplicates().sort_values(['magnitude'])
# %%
import matplotlib.pyplot as plt

df['diff_outcome'].plot.hist()
# df['magnitude'].plot.hist()
plt.savefig('figure1.png')

# %%
pd.plotting.parallel_coordinates(
    df[['granularity', 'alea_uncertainty', 'epis_uncertainty', 'magnitude']].drop_duplicates(),
    'magnitude')
plt.savefig('figure4.png')

# %%
gg = df[
    ['granularity', 'alea_uncertainty', 'epis_uncertainty', 'magnitude', 'diff_outcome']].drop_duplicates().sort_values(
    ['magnitude']).reset_index().astype(float).drop(columns=['index'])

import plotly.express as px

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
    color_continuous_midpoint=gg['diff_outcome'].max()/2)

fig.update_layout(coloraxis_showscale=True)

fig.write_image("figure3.png")

# %%
