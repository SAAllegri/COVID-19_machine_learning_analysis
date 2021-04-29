import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import MDS


class Reduction:

    def __init__(self, X):
        self.N, self.D = X.shape
        scaler = StandardScaler()
        X_normal = scaler.fit_transform(X)
        self.X = X_normal

    def pca(self, variance_ratio):
        for i in range(self.D):
            pca = PCA(n_components=i)
            pca.fit(self.X)
            total_variance = np.sum(pca.explained_variance_ratio_)
            if total_variance >= variance_ratio:
                break

        return pca.fit_transform(self.X) @ pca.components_, i

    def mds(self, components):
        embedding = MDS(n_components=components)
        X_transformed = embedding.fit_transform(self.X)

        return X_transformed
