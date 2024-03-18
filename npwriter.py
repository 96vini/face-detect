import pandas as pd
import numpy as np
import os

f_name = "face_data.csv"
def write(name: str, data: list):

    if os.path.isfile(f_name) and os.path.getsize(f_name) > 0:

        df = pd.read_csv(f_name, index_col=0)
    else:
        df = pd.DataFrame()

    latest = pd.DataFrame(data, columns=map(str, range(10000)))
    latest["name"] = name

    df = pd.concat((df, latest), ignore_index=True, sort=False)

    df.to_csv(f_name)