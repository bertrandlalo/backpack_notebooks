import warnings

import numpy as np
from pyriemann.classification import LogisticRegression
from pyriemann.estimation import ERPCovariances
from pyriemann.estimation import XdawnCovariances, Xdawn
from pyriemann.tangentspace import TangentSpace
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.pipeline import make_pipeline

import moabb

warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=RuntimeWarning)

moabb.set_log_level('info')


# This is an auxiliary transformer that allows one to vectorize data
# structures in a pipeline For instance, in the case of a X with dimensions
# Nt x Nc x Ns, one might be interested in a new data structure with
# dimensions Nt x (Nc.Ns)
class Vectorizer(BaseEstimator, TransformerMixin):

    def __init__(self):
        pass

    def fit(self, X, y):
        """fit."""
        return self

    def transform(self, X):
        """transform. """
        return np.reshape(X, (X.shape[0], -1))


##############################################################################
# Create pipelines
# ----------------
# Pipelines must be a dict of sklearn pipeline transformer.
pipelines = {}

# we have to do this because the classes are called 'Target' and 'NonTarget'
# but the evaluation function uses a LabelEncoder, transforming them
# to 0 and 1
labels_dict = {'Target': 1, 'NonTarget': 0}

pipelines['RG + LDA'] = make_pipeline(
    XdawnCovariances(
        nfilter=2,
        classes=[
            labels_dict['Target']],
        estimator='lwf',
        xdawn_estimator='lwf'),
    TangentSpace(),
    LDA(solver='lsqr', shrinkage='auto'))

pipelines['Xdw + LDA'] = make_pipeline(Xdawn(nfilter=2, estimator='lwf'),
                                       Vectorizer(), LDA(solver='lsqr',
                                                         shrinkage='auto'))
pipelines['ERPCov + TS'] = make_pipeline(ERPCovariances(classes=[0, 1], estimator='oas', svd=None),
                                         TangentSpace(metric='riemann'),
                                         LogisticRegression(solver='lbfgs'))
