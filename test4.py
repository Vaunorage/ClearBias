import sqlite3
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

connection = sqlite3.connect('elements.db')

# SQL query to select all columns from the table
query = "SELECT * FROM discriminations"

df = pd.read_sql_query(query, connection)



