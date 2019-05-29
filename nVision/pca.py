import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.decomposition import PCA


def interaction_features(data):
    """

    """
    new_cols = []
    new_data = pd.DataFrame([])
    for i, xi in enumerate(data.columns.tolist()):

        for xj in data.columns.tolist()[i:]:
            new_cols.append(xi + ':' + xj)
            new_data[xi + ':' + xj] = data[xi] * data[xj]
            # print("Done with iteration {}".format(xi + ':' + xj))

        comb_data = pd.concat([data, new_data], axis=1)

    return comb_data


def pca_analysis(data, n_components=None):
    """
    Perform the principal component analysis (PCA) to the input dataset 
    to reduce the dimensionality and to project the data to a 
    lower dimensional space.

    Returns the projected data after PCA, the PCA model and the components.
    The data are centered and scaled before PCA analysis.    

    Parameters
    ----------
    data : pandas.core.frame.DataFrame
        An m-by-n dataset with m observations and n features.
    n_components : None or int, optional
        The number of components. The default value is the number of features 
        of the dataset. Must be the lesser value of the number of observations
        and the number of features.

    Returns
    -------
    new_data : pandas.core.frame.DataFrame, returns to the data after 
    dimensionality reduction.
    pca_model : sklearn.decomposition.pca.PCA, returns to all the options 
    used in the PCA analysis.
    components : pandas.core.frame.DataFrame, returns to the calculated 
    principal axes in feature space, representing the directions of 
    maximum variance in the data. 
    
    Examples
    --------
    >>> data = pd.DataFrame(np.array([[2,0],[0,2]]))
    >>> new_data, pca_model, comps = pca.pca_analysis(data)
    
    
    """

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
