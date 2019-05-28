import numpy as np
import sys
import pandas as pd
import numpy.testing as npt

sys.path.insert(0, '.\\nVision')
import pca as pca


def test_pca_analysis():
    expected = np.array([[-np.sqrt(2), 0], [np.sqrt(2), 0]])
    actual, pca_model, comps = pca.pca_analysis(data = pd.DataFrame(np.array([[2,0],[0,2]])))
    npt.assert_almost_equal(expected, actual)
