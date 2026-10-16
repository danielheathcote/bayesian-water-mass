import pygeoinf as inf 
import numpy as np
from typing import List, Any
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist


class GridSpace(inf.HilbertSpace):
    def __init__(self, shape: tuple[int, ...]):
        if isinstance(shape, int):
            self._shape = (shape,)
        else:
            self._shape = tuple(shape)
        self._dim = int(np.prod(self._shape))
        
    @property
    def dim(self) -> int:
        return self._dim

    @property
    def shape(self) -> tuple[int, ...]:
        return self._shape

    def to_components(self, x: np.ndarray) -> np.ndarray:
        return x.flatten()

    def from_components(self, c: np.ndarray) -> np.ndarray:
        return c.reshape(self.shape)
    
    def add(self, x: np.ndarray, y: np.ndarray) -> np.ndarray: # todo: should this be necessary?
        # Ensure both operands are in the same shape
        if x.shape != y.shape:
            # Convert both to component form, add, then convert back
            x_flat = self.to_components(x) if x.shape == self.shape else x
            y_flat = self.to_components(y) if y.shape == self.shape else y
            result_flat = x_flat + y_flat
            return self.from_components(result_flat)
        return x + y

    def to_dual(self, x: np.ndarray) -> "inf.LinearForm":
        return inf.LinearForm(self, components=self.to_components(x))

    def from_dual(self, xp: "inf.LinearForm") -> np.ndarray:
        return self.from_components(xp.components)

    def __eq__(self, other: object):
        if not isinstance(other, GridSpace):
            return NotImplemented
        return self.shape == other.shape
    

class WaterMassForwardOperator:


    def __init__(self, mat: np.ndarray, ):
        self._mat = mat
        self._number_of_observation_variables = mat.shape[0]
        self._number_of_water_masses = mat.shape[1]

    @property
    def mat(self) -> np.ndarray:
        return self._mat

    @property
    def number_of_observation_variables(self) -> int:
        return self._number_of_observation_variables

    @property
    def number_of_water_masses(self) -> int:
        return self._number_of_water_masses

    def form_water_mass_matrix():
        # 
        pass

    def apply_matrix(self, v: np.ndarray) -> np.ndarray:
        # Ensure the matrix and vector dimensions are compatible
        assert self._mat.shape[1] == v.shape[1], "Matrix columns must match number of input variables"
        
        # Perform the matrix multiplication
        result = self._mat @ v.T  # Shape will be (m, n)
        return result.T  # Return shape (n, m)

    


class OldWaterMassForwardOperator:
    def __init__(self, space: GridSpace, mat: np.ndarray):
        self._space = space
        self._mat = mat
        self._number_of_observation_variables = mat.shape[0]
        self._number_of_water_masses = mat.shape[1]

    @property
    def space(self) -> GridSpace:
        return self._space

    @property
    def mat(self) -> np.ndarray:
        return self._mat

    def form_stacked_vector(self, x: List[np.ndarray]) -> np.ndarray:
        # Use space's to_components method instead of flatten
        return np.column_stack([self.space.to_components(arr) for arr in x])
    
    def apply_matrix(self, v: np.ndarray) -> np.ndarray:
        # Ensure the matrix and vector dimensions are compatible
        assert self.mat.shape[1] == v.shape[1], "Matrix columns must match number of input variables"
        
        # Perform the matrix multiplication
        result = self.mat @ v.T  # Shape will be (m, n)
        return result.T  # Return shape (n, m)
    
    def separate_stacked_vector(self, v: np.ndarray) -> List[np.ndarray]:
        # Use space's from_components method instead of reshape
        return [self.space.from_components(v[:, i]) for i in range(v.shape[1])]
    
    def transform(self, x: List[np.ndarray]) -> List[np.ndarray]:
        # Validate input length matches matrix expectation
        assert len(x) == self._number_of_water_masses, f"Expected {self._number_of_water_masses} input arrays, got {len(x)}"

        # Combine all operations in one method
        stacked = self.form_stacked_vector(x)
        transformed = self.apply_matrix(stacked)
        return self.separate_stacked_vector(transformed)
    
    def as_linear_operator(self) -> inf.LinearOperator:
        # Create product space for multiple inputs
        input_space = inf.HilbertSpaceDirectSum([self.space] * self._number_of_water_masses)
        output_space = inf.HilbertSpaceDirectSum([self.space] * self._number_of_observation_variables)

        def forward_mapping(x_list):
            return self.transform(x_list)
        
        return inf.LinearOperator(
            input_space,
            output_space,
            forward_mapping
        )
    

def create_spatial_covariance_matrix(
    shape: tuple[int, ...],
    std: float,
    length_scale: float,
    taper: bool = True,
    taper_distance_multiplier: float = 2.0,
) -> np.ndarray:
    """
    Create a covariance matrix with squared exponential (Gaussian) decay using vectorized operations.
    Optionally apply a compactly supported PSD taper to reduce long-range correlations.
    """
    # Generate coordinates for any dimensionality (1D, 2D, 3D, etc.)
    coords = np.indices(shape).reshape(len(shape), -1).T
    # Compute the pairwise Squared Euclidean Distance matrix
    dists_sq = cdist(coords, coords, 'sqeuclidean')
    # Apply the Squared Exponential Kernel: 
    # K(r) = sigma^2 * exp(-d^2 / (2 * l^2))
    cov_matrix = (std**2) * np.exp(-dists_sq / (2 * length_scale**2))

    if taper:
        # Wendland C2 taper (compact support, PSD in low dimensions)
        cutoff = taper_distance_multiplier * length_scale
        dists = np.sqrt(dists_sq)
        r = dists / cutoff
        taper_vals = np.where(r < 1.0, (1.0 - r) ** 4 * (4.0 * r + 1.0), 0.0)
        cov_matrix = cov_matrix * taper_vals

    return cov_matrix


def create_spatial_covariance_matrix_old(shape: tuple[int, ...], std: float, length_scale: float) -> np.ndarray:
    """
    Create a covariance matrix with squared exponential (Gaussian) decay based on spatial distance.
    
    Args:
        shape: Grid shape (e.g., (4, 5) for 2D grid)
        std: Standard deviation for all grid points  
        length_scale: Length scale for squared exponential decay
    
    Returns:
        Covariance matrix of size (prod(shape), prod(shape))
    """
    # Get total number of grid points
    n_points = np.prod(shape)
    
    # Create coordinate arrays for each grid point
    if len(shape) == 2:
        rows, cols = shape
        coords = np.array([[i, j] for i in range(rows) for j in range(cols)])
    elif len(shape) == 3:
        d1, d2, d3 = shape
        coords = np.array([[i, j, k] for i in range(d1) for j in range(d2) for k in range(d3)])
    else:
        # General case for arbitrary dimensions
        indices = np.meshgrid(*[range(dim) for dim in shape], indexing='ij')
        coords = np.column_stack([idx.ravel() for idx in indices])
    
    # Initialize covariance matrix
    cov_matrix = np.zeros((n_points, n_points))
    
    # Fill covariance matrix
    for i in range(n_points):
        for j in range(n_points):
            # Calculate Euclidean distance between grid points
            distance = np.linalg.norm(coords[i] - coords[j])
            
            # Squared exponential (Gaussian) covariance
            if i == j:
                # Diagonal elements (variance)
                cov_matrix[i, j] = std**2
            else:
                # Off-diagonal elements (squared exponential decay)
                cov_matrix[i, j] = std**2 * np.exp(-(distance**2) / (2 * length_scale**2))
    
    return cov_matrix


def create_point_measurement_operator(xs: List[float], ys: List[float], space: inf.HilbertSpaceDirectSum) -> inf.LinearOperator:
    """
    Create a linear operator that maps from a Hilbert space of multiple subspaces to point measurements.
    
    Args:
        xs: List of x-coordinates for point measurements
        ys: List of y-coordinates for point measurements
        space: Hilbert space with multiple subspaces (one for each variable)
    
    Returns:
        Linear operator mapping from the input space to point measurements
    """

    assert len(xs) == len(ys), "xs and ys must have the same length"
    number_of_measurements = len(xs)
    
    def mapping(variable_fields: List[np.ndarray]) -> np.ndarray:

        # Create separate arrays for each subspace
        measurements = [np.zeros(number_of_measurements) for _ in range(space.number_of_subspaces)]
        for i, (x, y) in enumerate(zip(xs, ys)):
            for j, field in enumerate(variable_fields):
                measurements[j][i] = field[int(y), int(x)]
        return measurements

    data_space = inf.HilbertSpaceDirectSum([inf.EuclideanSpace(number_of_measurements) for _ in range(space.number_of_subspaces)])

    return inf.LinearOperator(space, data_space, mapping)