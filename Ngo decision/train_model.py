import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.cluster import AgglomerativeClustering
def pca_performed():
        Col = ColumnTransformer([

            ('one-Hot-encoder', OneHotEncoder(sparse_output=False),['country']),
            ('num',StandardScaler(),['child_mort','health','inflation','life_expec','total_fer','gdpp','net_exports']),

            ],remainder='passthrough')

        piplines=Pipeline([('scalar',Col),('feature_reduce',PCA(n_components=2)),('model',AgglomerativeClustering(n_clusters=2))])
        

        return piplines