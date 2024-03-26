from config import cfg
import pandas as pd

data = pd.read_csv(**cfg.goodreads_limres)

print(data)
