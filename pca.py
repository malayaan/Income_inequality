import numpy as np
import matplotlib.pyplot as plt

class PCA:

    def __init__(self):
        self.eigenvalues = None
        self.eigenvectors = None

    def fit(self, X, D=None):
        if D is None:
            D = np.diag([1/len(X)]*len(X))

        # Center the data
        mean_X = np.mean(X, axis=0)
        Z = X - mean_X

        # Compute the covariance matrix
        S = Z.T @ D @ Z

        # Compute eigenvalues and eigenvectors
        self.eigenvalues, eigenvectors = np.linalg.eig(S)
        
        # Sort eigenvalues and eigenvectors
        idx = self.eigenvalues.argsort()[::-1]
        self.eigenvalues = self.eigenvalues[idx]
        self.eigenvectors = eigenvectors[:,idx]

        # Normalize eigenvectors
        for i in range(len(self.eigenvectors)):
            self.eigenvectors[:, i] = self.eigenvectors[:, i] / np.linalg.norm(self.eigenvectors[:, i])

    def transform(self, X):
        mean_X = np.mean(X, axis=0)
        Z = X - mean_X
        return Z @ self.eigenvectors

    def fit_transform(self, X, D=None):
        self.fit(X, D)
        return self.transform(X)

    def explained_variance_ratio(self):
        return self.eigenvalues / np.sum(self.eigenvalues)

if __name__ == "__main__":

    # Example usage:
    # Create an instance of PCA
    pca = PCA()

    # Fit and transform the data
    X = np.array([[0, 1, 0, 1, 1, 0], [0, 1, 1, 0, 1, 0]]).T
    factorial_coordinates = pca.fit_transform(X)

    # Creating the plot
    plt.figure(figsize=(8, 8))
    plt.scatter(X[:, 0], X[:, 1], color='blue', label='Data')
    plt.quiver(np.mean(X[:, 0]), np.mean(X[:, 1]), pca.eigenvectors[0, 0], pca.eigenvectors[1, 0], angles='xy', scale_units='xy', scale=1, color='red', label='Principal Axis 1')
    plt.quiver(np.mean(X[:, 0]), np.mean(X[:, 1]), pca.eigenvectors[0, 1], pca.eigenvectors[1, 1], angles='xy', scale_units='xy', scale=1, color='green', label='Principal Axis 2')
    plt.axhline(0, color='black', linewidth=0.5)
    plt.axvline(0, color='black', linewidth=0.5)
    plt.grid(color='gray', linestyle='--', linewidth=0.5)
    plt.title('Graph of the 2 principal axes D1 and D2 extended')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.legend()
    plt.axis('equal')
    plt.show()

    # Print explained variance ratio for each component
    print(pca.explained_variance_ratio())
