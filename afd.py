import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

class AFD:
    def __init__(self, data, target_column):
        self.data = data
        self.target_column = target_column
        self.groups = data.groupby(target_column)
        self.numeric_columns = data.select_dtypes(include=[np.number]).columns  # updated to np.number to include all numeric types
        self.n = data.shape[0]

    def compute_matrices(self):
        # Calculating mean vectors for each group
        self.mean_vectors = {group: df[self.numeric_columns].mean() for group, df in self.groups}

        # Calculating the overall mean
        self.overall_mean = self.data[self.numeric_columns].mean()

        # Initializing B matrix
        B = np.zeros((len(self.numeric_columns), len(self.numeric_columns)))
        for group, df in self.groups:
            group_size = df.shape[0]
            mean_diff = (self.mean_vectors[group] - self.overall_mean).values.reshape(-1, 1)
            B += group_size * mean_diff @ mean_diff.T
        self.B = B / self.n

        # Initializing W matrix
        W = np.zeros((len(self.numeric_columns), len(self.numeric_columns)))
        for group, df in self.groups:
            group_size = df.shape[0]
            mean_diff = df[self.numeric_columns].sub(self.mean_vectors[group])
            Wk = mean_diff.T @ mean_diff / group_size
            W += group_size * Wk
        self.W = W / self.n

        # Calculating V matrix as the sum of B and W
        self.V = self.B + self.W


    def compute_factorial_axis(self):
        V_inv = np.linalg.inv(self.V)
        c_kj = {}
        print(self.mean_vectors)
        for group, mean_vector in self.mean_vectors.items():
            group_size = self.groups.get_group(group).shape[0]
            c_kj[group] = np.sqrt(group_size / self.n) * (mean_vector - self.overall_mean)
        c_matrix = np.array([c_kj[group].values for group in c_kj])

        # Computing the matrix C * V_inv * C.T
        C_V_inv_Ct = c_matrix @ V_inv @ c_matrix.T

        # Computing eigenvalues and eigenvectors
        eigenvalues, eigenvectors = np.linalg.eig(C_V_inv_Ct)

        # Sorting eigenvalues and eigenvectors by the absolute values of eigenvalues in descending order
        sorted_indices = np.argsort(np.abs(eigenvalues))[::-1]
        self.sorted_eigenvalues = eigenvalues[sorted_indices]  # Storing the sorted eigenvalues
        sorted_eigenvectors = eigenvectors[:, sorted_indices]
        self.sorted_eigenvectors = V_inv @ c_matrix.T @ sorted_eigenvectors
        # Selecting the K-1 largest eigenvalues and corresponding eigenvectors
        K_minus_one = len(self.groups) - 1
        largest_eigenvectors = sorted_eigenvectors[:, :K_minus_one]

        # Projecting the selected eigenvectors
        self.a = V_inv @ c_matrix.T @ largest_eigenvectors

    def project_data(self, dimensions=0):
        if isinstance(dimensions, int):
            # Single dimension
            a1 = self.a[:, dimensions]
            self.projections1 = self.data[self.numeric_columns].dot(a1)
            self.data_with_projections = self.data.copy()
            self.data_with_projections['Projection_X'] = self.projections1
        elif isinstance(dimensions, tuple) and len(dimensions) == 2:
            # Two dimensions
            i, j = dimensions
            a1 = self.a[:, i]
            a2 = self.a[:, j]
            self.projections1 = self.data[self.numeric_columns].dot(a1)
            self.projections2 = self.data[self.numeric_columns].dot(a2)
            self.data_with_projections = self.data.copy()
            self.data_with_projections['Projection_X'] = self.projections1
            self.data_with_projections['Projection_Y'] = self.projections2




    def plot_projections(self):
        plt.figure(figsize=(10, 6))
        if 'Projection_Y' in self.data_with_projections.columns:
            # Two-dimensional plot
            sns.scatterplot(data=self.data_with_projections, x='Projection_X', y='Projection_Y', hue=self.target_column, style=self.target_column)
            plt.xlabel('Projection on Axis 1')
            plt.ylabel('Projection on Axis 2')
        else:
            # One-dimensional plot
            sns.scatterplot(data=self.data_with_projections, x='Projection_X', y=[0]*self.data_with_projections.shape[0], hue=self.target_column, style=self.target_column)
            plt.xlabel('Projection on Axis')
            plt.ylabel('')
            plt.yticks([])
        plt.title('Projection of Individuals on Selected Factorial Axes by ' + self.target_column)
        plt.show()



if __name__ == "__main__":
    # Usage example
    file_path = 'chienloup.csv'  # Ensure this path is correct
    data = pd.read_csv(file_path, sep=';')  # Adjust the separator if needed

    # Initialize the AFD object with the dataset and the target column
    afd = AFD(data, 'GENRE')

    # Compute the matrices B, W, and V
    afd.compute_matrices()

    # Compute the factorial axes
    afd.compute_factorial_axis()

    # Project the data onto the first two factorial axes (or any two axes of your choice)
    afd.project_data(dimensions=0)

    # Plot the projections
    afd.plot_projections()
    
    print(afd.sorted_eigenvalues)