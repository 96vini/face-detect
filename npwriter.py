import pandas as pd
import numpy as np
import os

f_name = "face_data.csv"

def write(name: str, data: list):
    # Verifica se o arquivo já existe e se está vazio
    if os.path.isfile(f_name) and os.path.getsize(f_name) > 0:
        # Se o arquivo não estiver vazio, lê os dados existentes
        df = pd.read_csv(f_name, index_col=0)
    else:
        # Se o arquivo estiver vazio, cria um DataFrame vazio
        df = pd.DataFrame()

    # Cria um novo DataFrame com os dados mais recentes
    latest = pd.DataFrame(data, columns=map(str, range(10000)))
    latest["name"] = name

    # Concatena o novo DataFrame com os dados existentes (ou DataFrame vazio)
    df = pd.concat((df, latest), ignore_index=True, sort=False)

    # Escreve o DataFrame atualizado no arquivo CSV
    df.to_csv(f_name)
