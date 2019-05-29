import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.decomposition import PCA


def interaction_features(data):
    """
    Outputs an extended array with all possible interaction features x_i * x_j

    Parameters
    ----------
    data : DataFrame
        2D DataFrame of dimensions (n observations) x (p features)

    Returns
    -------
    comb_data : DataFrame
        2D DataFrame containing original p features plus C(p+1,p-1) new
        interaction features

    Examples
    --------
    >>> A
        0   1
    0   1   0
    1   0   1

    >>> interaction_features(A)
        0   1   0:0 0:1 1:1
    0   1   0   1   0   0
    1   0   1   0   0   1

    """
    new_cols = []
    new_data = pd.DataFrame([])
    for i, xi in enumerate(data.columns.tolist()):

        for xj in data.columns.tolist()[i:]:
            new_cols.append(str(xi) + ':' + str(xj))
            new_data[str(xi) + ':' + str(xj)] = data[xi] * data[xj]
            # print("Done with iteration {}".format(xi + ':' + xj))

    comb_data = pd.concat([data, new_data], axis=1)

    return comb_data


def pca_analysis(data, n_components=None):

    scaler = preprocessing.StandardScaler().fit(data)

    data_s = scaler.transform(data)
    data_s = pd.DataFrame(data=data_s)

    if n_components == None:
        pca_model = PCA(n_components=data_s.shape[1])
    else:
        pca_model = PCA(n_components=n_components)

    new_data= pca_model.fit(data_s).transform(data_s)
    new_data = pd.DataFrame(data=new_data)

    components = pd.DataFrame(pca_model.components_)
    return new_data, pca_model, components
