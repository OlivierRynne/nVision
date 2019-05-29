import numpy as np
import sys
import pandas as pd
import numpy.testing as npt
from pandas.util.testing import assert_frame_equal
from nVision import pca


def test_pca_analysis():
    expected = np.array([[-np.sqrt(2), 0], [np.sqrt(2), 0]])
    actual, pca_model, comps = pca.pca_analysis(data = pd.DataFrame(np.array([[2,0],[0,2]])))
    npt.assert_almost_equal(expected, actual)


def test_interaction_features():
    expected = pd.DataFrame({'A':[1,1,1],'B':[2,2,2],'C':[3,3,3],'A:A':[1,1,1],'A:B':[2,2,2],'A:C':[3,3,3],'B:B':[4,4,4],'B:C':[6,6,6],'C:C':[9,9,9]})
    data = pd.DataFrame({'A':[1,1,1],'B':[2,2,2],'C':[3,3,3]})
    actual = pca.interaction_features(data)

    assert_frame_equal(expected, actual)
