import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
DATA_PATH = os.path.join(ROOT, 'data')

df = pd.read_csv(os.path.join(DATA_PATH, 'raw', 'dataset.csv'))

# Drop Duplicates
df = df.drop_duplicates()

# Fix right-skew in 'Amount'
df['Amount'] = np.log1p(df['Amount'])

df.to_csv(os.path.join(DATA_PATH, 'processed', 'processed.csv'), index=False)