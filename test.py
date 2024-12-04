import os
import certifi

from ucimlrepo import fetch_ucirepo, list_available_datasets

# check which datasets can be imported
certifi.where()
list_available_datasets()
#%%