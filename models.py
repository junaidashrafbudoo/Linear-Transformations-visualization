# models.py

import numpy as np

class Matrix:
    def __init__(self, data):
        self.data = np.array(data, dtype=float)
        self.rows, self.cols = self.data.shape

    def __matmul__(self, other):
        if isinstance(other, Vector):
            return Vector(self.data @ other.data)
        elif isinstance(other, Matrix):
            return Matrix(self.data @ other.data)
        else:
            return self.data @ other

    def eigenvalues(self):
        """
        Compute the eigenvalues and eigenvectors of the matrix.
        """
        vals, vecs = np.linalg.eig(self.data)
        return vals, vecs

    def is_symmetric(self):
        """
        Check if the matrix is symmetric.
        """
        return np.allclose(self.data, self.data.T)

    def is_orthogonal(self):
        """
        Check if the matrix is orthogonal.
        """
        return np.allclose(self.data.T @ self.data, np.eye(self.rows))

    def determinant(self):
        """
        Compute the determinant of the matrix.
        """
        return np.linalg.det(self.data)

    def inverse(self):
        """
        Compute the inverse of the matrix.
        """
        return Matrix(np.linalg.inv(self.data))

    def rank(self):
        """
        Compute the rank of the matrix.
        """
        return np.linalg.matrix_rank(self.data)

    def trace(self):
        """
        Compute the trace of the matrix.
        """
        return np.trace(self.data)

    def singular_values(self):
        """
        Compute the singular values of the matrix.
        """
        return np.linalg.svd(self.data, compute_uv=False)

    def condition_number(self):
        """
        Compute the condition number of the matrix.
        """
        sv = self.singular_values()
        return sv.max() / sv.min()

    def null_space(self):
        """
        Compute an orthonormal basis for the null space of the matrix.
        """
        u, s, vh = np.linalg.svd(self.data)
        tol = max(self.rows, self.cols) * np.spacing(s[0])
        nnz = (s >= tol).sum()
        ns = vh[nnz:].conj().T
        return ns

    def __str__(self):
        return str(self.data)

    def __repr__(self):
        return f"Matrix(rows={self.rows}, cols={self.cols})"

class Vector:
    def __init__(self, data):
        self.data = np.array(data, dtype=float)

    def magnitude(self):
        """
        Compute the magnitude (length) of the vector.
        """
        return np.linalg.norm(self.data)

    def normalize(self):
        """
        Return a normalized (unit length) vector.
        """
        mag = self.magnitude()
        if mag == 0:
            return Vector(self.data)
        return Vector(self.data / mag)

    def dot(self, other):
        """
        Compute the dot product with another vector.
        """
        return np.dot(self.data, other.data)

    def cross(self, other):
        """
        Compute the cross product with another vector (only for 3D vectors).
        """
        return Vector(np.cross(self.data, other.data))

    def angle_with(self, other):
        """
        Compute the angle between this vector and another vector.
        """
        dot_prod = self.dot(other)
        mag_self = self.magnitude()
        mag_other = other.magnitude()
        if mag_self == 0 or mag_other == 0:
            return 0
        cos_theta = dot_prod / (mag_self * mag_other)
        return np.arccos(np.clip(cos_theta, -1.0, 1.0))

    def projection_onto(self, other):
        """
        Compute the projection of this vector onto another vector.
        """
        other_norm = other.normalize()
        proj_length = self.dot(other_norm)
        return Vector(proj_length * other_norm.data)

    def __str__(self):
        return str(self.data)

    def __repr__(self):
        return f"Vector(dim={len(self.data)})"

class Transformation:
    def __init__(self, name, matrix, latex, explanation):
        """
        Initialize a transformation with a name, matrix, LaTeX representation, and explanation.
        """
        self.name = name
        self.matrix = Matrix(matrix)
        self.latex = latex
        self.explanation = explanation

    def apply(self, vector):
        """
        Apply the transformation to a vector.
        """
        return self.matrix @ vector

# Predefined 2D transformations
TRANSFORMATIONS_2D = {
    'Rotation 45°': Transformation(
        name='Rotation 45°',
        matrix=[
            [np.cos(np.pi / 4), -np.sin(np.pi / 4)],
            [np.sin(np.pi / 4), np.cos(np.pi / 4)]
        ],
        latex=r'\begin{pmatrix} \cos 45^\circ & -\sin 45^\circ \\ \sin 45^\circ & \cos 45^\circ \end{pmatrix}',
        explanation='This matrix rotates vectors by 45 degrees.'
    ),
    'Scaling': Transformation(
        name='Scaling',
        matrix=[
            [2, 0],
            [0, 1.5]
        ],
        latex=r'\begin{pmatrix} 2 & 0 \\ 0 & 1.5 \end{pmatrix}',
        explanation='This matrix scales vectors by different factors along x and y axes.'
    ),
    'Shear': Transformation(
        name='Shear',
        matrix=[
            [1, 1],
            [0, 1]
        ],
        latex=r'\begin{pmatrix} 1 & 1 \\ 0 & 1 \end{pmatrix}',
        explanation='This matrix shears the shape along the x-axis.'
    ),
    'Reflection over X-axis': Transformation(
        name='Reflection over X-axis',
        matrix=[
            [1, 0],
            [0, -1]
        ],
        latex=r'\begin{pmatrix} 1 & 0 \\ 0 & -1 \end{pmatrix}',
        explanation='This matrix reflects vectors over the x-axis.'
    ),
    'Reflection over Y-axis': Transformation(
        name='Reflection over Y-axis',
        matrix=[
            [-1, 0],
            [0, 1]
        ],
        latex=r'\begin{pmatrix} -1 & 0 \\ 0 & 1 \end{pmatrix}',
        explanation='This matrix reflects vectors over the y-axis.'
    ),
    'Projection onto X-axis': Transformation(
        name='Projection onto X-axis',
        matrix=[
            [1, 0],
            [0, 0]
        ],
        latex=r'\begin{pmatrix} 1 & 0 \\ 0 & 0 \end{pmatrix}',
        explanation='This matrix projects vectors onto the x-axis.'
    ),
    'Projection onto Y-axis': Transformation(
        name='Projection onto Y-axis',
        matrix=[
            [0, 0],
            [0, 1]
        ],
        latex=r'\begin{pmatrix} 0 & 0 \\ 0 & 1 \end{pmatrix}',
        explanation='This matrix projects vectors onto the y-axis.'
    ),
    # Add more predefined 2D transformations as needed
}

# Predefined 3D transformations
TRANSFORMATIONS_3D = {
    'Rotation about X-axis': Transformation(
        name='Rotation about X-axis',
        matrix=[
            [1, 0, 0],
            [0, np.cos(np.pi / 4), -np.sin(np.pi / 4)],
            [0, np.sin(np.pi / 4), np.cos(np.pi / 4)]
        ],
        latex=r'\begin{pmatrix} 1 & 0 & 0 \\ 0 & \cos \theta & -\sin \theta \\ 0 & \sin \theta & \cos \theta \end{pmatrix}',
        explanation='This matrix rotates vectors around the X-axis by 45 degrees.'
    ),
    'Rotation about Y-axis': Transformation(
        name='Rotation about Y-axis',
        matrix=[
            [np.cos(np.pi / 4), 0, np.sin(np.pi / 4)],
            [0, 1, 0],
            [-np.sin(np.pi / 4), 0, np.cos(np.pi / 4)]
        ],
        latex=r'\begin{pmatrix} \cos \theta & 0 & \sin \theta \\ 0 & 1 & 0 \\ -\sin \theta & 0 & \cos \theta \end{pmatrix}',
        explanation='This matrix rotates vectors around the Y-axis by 45 degrees.'
    ),
    'Rotation about Z-axis': Transformation(
        name='Rotation about Z-axis',
        matrix=[
            [np.cos(np.pi / 4), -np.sin(np.pi / 4), 0],
            [np.sin(np.pi / 4), np.cos(np.pi / 4), 0],
            [0, 0, 1]
        ],
        latex=r'\begin{pmatrix} \cos \theta & -\sin \theta & 0 \\ \sin \theta & \cos \theta & 0 \\ 0 & 0 & 1 \end{pmatrix}',
        explanation='This matrix rotates vectors around the Z-axis by 45 degrees.'
    ),
    'Scaling': Transformation(
        name='Scaling',
        matrix=[
            [2, 0, 0],
            [0, 1.5, 0],
            [0, 0, 1]
        ],
        latex=r'\begin{pmatrix} 2 & 0 & 0 \\ 0 & 1.5 & 0 \\ 0 & 0 & 1 \end{pmatrix}',
        explanation='This matrix scales vectors along x, y, and z axes.'
    ),
    'Shear': Transformation(
        name='Shear',
        matrix=[
            [1, 0.5, 0],
            [0, 1, 0],
            [0, 0, 1]
        ],
        latex=r'\begin{pmatrix} 1 & k & 0 \\ 0 & 1 & 0 \\ 0 & 0 & 1 \end{pmatrix}',
        explanation='This matrix shears the shape along the x-axis in 3D.'
    ),
    'Reflection over XY-plane': Transformation(
        name='Reflection over XY-plane',
        matrix=[
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, -1]
        ],
        latex=r'\begin{pmatrix} 1 & 0 & 0 \\ 0 & 1 & 0 \\ 0 & 0 & -1 \end{pmatrix}',
        explanation='This matrix reflects vectors over the XY-plane.'
    ),
    'Projection onto XY-plane': Transformation(
        name='Projection onto XY-plane',
        matrix=[
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, 0]
        ],
        latex=r'\begin{pmatrix} 1 & 0 & 0 \\ 0 & 1 & 0 \\ 0 & 0 & 0 \end{pmatrix}',
        explanation='This matrix projects vectors onto the XY-plane.'
    ),
    # Add more predefined 3D transformations as needed
}

# Symmetric transformations in 2D
SYMMETRIC_TRANSFORMATIONS_2D = {
    'Symmetric Matrix': Transformation(
        name='Symmetric Matrix',
        matrix=[
            [2, 1],
            [1, 2]
        ],
        latex=r'\begin{pmatrix} 2 & 1 \\ 1 & 2 \end{pmatrix}',
        explanation='This is a symmetric 2x2 matrix with real eigenvalues and orthogonal eigenvectors.'
    ),
    'Diagonal Matrix': Transformation(
        name='Diagonal Matrix',
        matrix=[
            [3, 0],
            [0, 1]
        ],
        latex=r'\begin{pmatrix} 3 & 0 \\ 0 & 1 \end{pmatrix}',
        explanation='A diagonal matrix with eigenvalues on the diagonal.'
    ),
    # Add more symmetric 2D transformations
}

# Symmetric transformations in 3D
SYMMETRIC_TRANSFORMATIONS_3D = {
    'Symmetric Matrix 3D': Transformation(
        name='Symmetric Matrix 3D',
        matrix=[
            [2, 1, 0],
            [1, 2, 1],
            [0, 1, 2]
        ],
        latex=r'\begin{pmatrix} 2 & 1 & 0 \\ 1 & 2 & 1 \\ 0 & 1 & 2 \end{pmatrix}',
        explanation='A symmetric 3x3 matrix with real eigenvalues and orthogonal eigenvectors.'
    ),
    'Diagonal Matrix': Transformation(
        name='Diagonal Matrix',
        matrix=[
            [3, 0, 0],
            [0, 2, 0],
            [0, 0, 1]
        ],
        latex=r'\begin{pmatrix} 3 & 0 & 0 \\ 0 & 2 & 0 \\ 0 & 0 & 1 \end{pmatrix}',
        explanation='A diagonal matrix with eigenvalues on the diagonal.'
    ),
    # Add more symmetric 3D transformations
}

# Additional functions for transformations
def rotation_matrix_2d(theta):
    """
    Returns a 2D rotation matrix for a given angle theta.
    """
    return np.array([
        [np.cos(theta), -np.sin(theta)],
        [np.sin(theta), np.cos(theta)]
    ])

def rotation_matrix_3d(axis, theta):
    """
    Returns a 3D rotation matrix for a given axis and angle theta.
    """
    axis = np.array(axis, dtype=float)
    axis = axis / np.linalg.norm(axis)
    x, y, z = axis
    c = np.cos(theta)
    s = np.sin(theta)
    C = 1 - c
    return np.array([
        [c + x*x*C, x*y*C - z*s, x*z*C + y*s],
        [y*x*C + z*s, c + y*y*C, y*z*C - x*s],
        [z*x*C - y*s, z*y*C + x*s, c + z*z*C]
    ])

def create_custom_transformation_2d(a, b, c, d):
    """
    Creates a custom 2D transformation matrix with user-defined elements.
    """
    return np.array([
        [a, b],
        [c, d]
    ])

def create_custom_transformation_3d(a, b, c, d, e, f, g, h, i):
    """
    Creates a custom 3D transformation matrix with user-defined elements.
    """
    return np.array([
        [a, b, c],
        [d, e, f],
        [g, h, i]
    ])
