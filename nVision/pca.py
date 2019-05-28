import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def interaction_features(data):

    new_cols = []
    new_data = pd.DataFrame([])
    for i, xi in enumerate(data.columns.tolist()):

        for xj in data.columns.tolist()[i:]:
            new_cols.append(xi + ':' + xj)
            new_data[xi + ':' + xj] = data[xi] * data[xj]
            # print("Done with iteration {}".format(xi + ':' + xj))

        comb_data = pd.concat([data, new_data], axis=1)

    return comb_data


def pca_analysis(data):

    return new_data, pca_model, components
