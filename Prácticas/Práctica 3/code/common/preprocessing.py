import pandas as pd
import numpy as np
from .vars import SEED

np.random.seed(SEED)

#from sklearn.decomposition import PCA
#from sklearn.preprocessing import StandardScaler


def trans_discretize(df, attr, discrete_dict):
  return df.replace({attr: discrete_dict})

def trans_onehot(df, attr):
  dummies = pd.get_dummies(df[attr], prefix=attr).astype(np.int64)
  df_trans = df.drop(labels=attr, axis=1)
  return pd.concat([df_trans, dummies], axis=1)

def trans_discretize(df, attr, discrete_dict=None):
  if discrete_dict:
    return df.replace({attr: discrete_dict})
  else:
    discrete_dict = {f[1]: f[0] for f in enumerate(df[attr].unique())}
    return trans_discretize(df, attr, discrete_dict), discrete_dict

def impute(df, strategy, attrs=None, fill=None):
  df = df.copy()
  if attrs is None: attrs = df.columns
  for attr in attrs:
    try:
      if strategy == 'mean':
        df[attr] = df[attr].fillna(df[attr].mean())
      if strategy == 'median':
        df[attr] = df[attr].fillna(df[attr].median())
      if strategy == 'mode':
        df[attr] = df[attr].fillna(df[attr].mode()[0])
      if strategy == 'fill':
        if fill:
          df[attr] = df[attr].fillna(fill)
        else:
          print(f"Para aplicar la estrategia 'fill' debe especificar un valor de 'fill'")
    except Exception as e:
      print(f"No se pudo aplicar la estrategia '{strategy}' sobre el atributo '{attr}'")
  return df
