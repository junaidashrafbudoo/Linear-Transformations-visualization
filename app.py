# app.py

import streamlit as st
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
import plotly.graph_objects as go
from utils import (
    create_vector_animation_frames,
    create_grid_animation_frames,
    generate_3d_shape,
    generate_2d_shape,
    create_vector_plot,
    create_grid_plot,
    perform_pca,
    plot_pca_results,
    plot_explained_variance,
    create_pca_animation_frames,
    plot_pca_animation,
    perform_svd,
    reconstruct_matrix,
    plot_svd_variance,
    create_svd_3d_visualization,
    create_svd_animation_frames,
    plot_svd_animation
)
from models import (
    Matrix,
    Vector,
    TRANSFORMATIONS_2D,
    TRANSFORMATIONS_3D,
    SYMMETRIC_TRANSFORMATIONS_2D,
    SYMMETRIC_TRANSFORMATIONS_3D
)

# --- Streamlit App Setup ---
st.set_page_config(page_title="Matrix Transformation, PCA, and SVD Visualization", layout="wide")
st.title("Matrix Transformation, PCA, and SVD Visualization")
st.write(
    "This app visualizes how matrices transform vectors and grids in 2D and 3D space, and demonstrates Principal Component Analysis (PCA) and Singular Value Decomposition (SVD) for dimensionality reduction."
)

# --- Sidebar ---
# Function Selection
st.sidebar.header("Select Function")
function_choice = st.sidebar.radio(
    "Choose a function:",
    ("2D Transformation", "3D Transformation", "Principal Component Analysis (PCA)", "Singular Value Decomposition (SVD) Visualization")
)

# Plot Theme Selection
st.sidebar.header("Plot Theme")
available_themes = [
    "plotly", "plotly_white", "plotly_dark", "ggplot2", "seaborn",
    "simple_white", "none"
]
selected_theme = st.sidebar.selectbox("Select Plotly Theme",
                                      options=available_themes,
                                      index=available_themes.index(
                                          "plotly_white"))

# Point Size Adjustment
st.sidebar.header("Plot Customization")
point_size = st.sidebar.slider("Data Point Size",
                               min_value=2,
                               max_value=20,
                               value=5,
                               step=1)
st.session_state['marker_size'] = point_size

if function_choice == "2D Transformation":
    # --- 2D Transformation ---
    # --- Sidebar Options ---
    st.sidebar.header("2D Grid Settings")
    grid_shape_2d = st.sidebar.selectbox("Select 2D Shape",
                                         options=["Rectangle", "Triangle", "Octagon"])
    point_density = st.sidebar.slider("Point Density",
                                      min_value=5,
                                      max_value=50,
                                      value=21,
                                      step=2)
    range_min = st.sidebar.number_input("Range Min", value=-10.0, step=1.0)
    range_max = st.sidebar.number_input("Range Max", value=10.0, step=1.0)

    # Generate 2D Grid
    original_points = generate_2d_shape(grid_shape_2d, int(point_density), range_min, range_max)

    # --- 2D Transformation Examples ---
    st.header("2D Transformation Visualization")

    st.sidebar.header("2D Transformation Examples")
    selected_example_2d_name = st.sidebar.selectbox(
        'Select a 2D Transformation Example',
        options=['None'] + list(TRANSFORMATIONS_2D.keys()),
        index=0)
    selected_example_2d = TRANSFORMATIONS_2D.get(selected_example_2d_name)

    # Initialize 2D Matrix
    if 'matrix_2d' not in st.session_state:
        st.session_state['matrix_2d'] = np.eye(2)

    # Apply or Update 2D Matrix
    if selected_example_2d:
        st.session_state['matrix_2d'] = selected_example_2d.matrix.data
    else:
        st.sidebar.header("2D Matrix Input")
        matrix_entries_2d = {}
        for i in range(2):
            for j in range(2):
                key = f'a{i+1}{j+1}'
                value = float(st.session_state['matrix_2d'][i, j])
                matrix_entries_2d[key] = st.sidebar.number_input(
                    key, value=value, step=0.1, format="%.2f")
        st.session_state['matrix_2d'] = np.array([
            [matrix_entries_2d['a11'], matrix_entries_2d['a12']],
            [matrix_entries_2d['a21'], matrix_entries_2d['a22']]
        ])

    # Display 2D Matrix
    st.sidebar.write("Current 2D Matrix:")
    st.sidebar.latex(
        r'''
        \begin{pmatrix}
        %.2f & %.2f \\
        %.2f & %.2f
        \end{pmatrix}
        ''' % (st.session_state['matrix_2d'][0, 0],
               st.session_state['matrix_2d'][0, 1],
               st.session_state['matrix_2d'][1, 0],
               st.session_state['matrix_2d'][1, 1]))

    # 2D Vector Input
    st.sidebar.header("2D Vector Input")
    vector_2d = np.zeros(2)
    for i in range(2):
        key = f'v{i+1}'
        default_value = 1.0 if i == 0 else 0.0
        vector_2d[i] = st.sidebar.number_input(key,
                                               value=float(default_value),
                                               step=0.1, format="%.2f")

    # 2D Vector Transformation
    matrix_2d = Matrix(st.session_state['matrix_2d'])
    vector_2d = Vector(vector_2d)
    transformed_vector_2d = matrix_2d @ vector_2d

    # 2D Vector Animation
    frames_2d = create_vector_animation_frames(matrix_2d, vector_2d)
    create_vector_plot(vector_2d, transformed_vector_2d, frames_2d,
                       selected_theme)

    # --- 2D Grid Transformation ---
    st.header(" Geometric Transformations in ℝ²")
    # Apply Transformation
    transformed_points = st.session_state['matrix_2d'] @ original_points

    # Calculate Axis Limits
    all_x = np.concatenate([original_points[0, :], transformed_points[0, :]])
    all_y = np.concatenate([original_points[1, :], transformed_points[1, :]])
    x_min, x_max = all_x.min(), all_x.max()
    y_min, y_max = all_y.min(), all_y.max()
    padding = max((x_max - x_min), (y_max - y_min)) * 0.1
    x_range = [x_min - padding, x_max + padding]
    y_range = [y_min - padding, y_max + padding]

    # Grid Animation
    frames_grid = create_grid_animation_frames(matrix_2d, original_points)
    create_grid_plot(original_points, transformed_points, frames_grid,
                     selected_theme, x_range, y_range)

    # --- 2D Eigenvalues and Eigenvectors ---
    st.header("Eigenvalues and Eigenvectors (2D)")
    eigenvalues, eigenvectors = matrix_2d.eigenvalues()
    st.subheader("Eigenvalues")
    for idx, val in enumerate(eigenvalues):
        st.latex(r"\lambda_%d = %.2f" % (idx + 1, val.real))
    st.subheader("Eigenvectors")
    for idx, vec in enumerate(eigenvectors.T):
        st.latex(r"""\vec{{v}}_{%d} = \begin{pmatrix} %.2f \\ %.2f \end{pmatrix}""" %
                 (idx + 1, vec[0].real, vec[1].real))
    st.write(
        r"""An **eigenvector** of a matrix \( A \) is a non-zero vector \( \vec{v} \) that, when \( A \) is applied to it, does not change direction. Instead, it is only scaled by a scalar factor called the **eigenvalue** \( \lambda \). This relationship is expressed as:"""
    )
    st.latex(r"A \vec{v} = \lambda \vec{v}")

    # --- Symmetric Matrices in 2D ---
    st.header("Symmetric Matrices in 2D")
    st.sidebar.header("Symmetric Matrices (2D)")
    selected_symmetric_example_2d_name = st.sidebar.selectbox(
        'Select a Symmetric 2D Transformation',
        options=['None'] + list(SYMMETRIC_TRANSFORMATIONS_2D.keys()),
        index=0)
    selected_symmetric_example_2d = SYMMETRIC_TRANSFORMATIONS_2D.get(selected_symmetric_example_2d_name)

    if selected_symmetric_example_2d:
        st.session_state['matrix_2d'] = selected_symmetric_example_2d.matrix.data
        # Display Symmetric Matrix
        st.write(f"**{selected_symmetric_example_2d.name}**:")
        st.latex(selected_symmetric_example_2d.latex)
        st.write(selected_symmetric_example_2d.explanation)
        # Recalculate transformations
        matrix_2d = Matrix(st.session_state['matrix_2d'])
        transformed_vector_2d = matrix_2d @ vector_2d
        transformed_points = st.session_state['matrix_2d'] @ original_points
        # Recalculate animations
        frames_2d = create_vector_animation_frames(matrix_2d, vector_2d)
        create_vector_plot(vector_2d, transformed_vector_2d, frames_2d,
                           selected_theme)
        frames_grid = create_grid_animation_frames(matrix_2d, original_points)
        create_grid_plot(original_points, transformed_points, frames_grid,
                         selected_theme, x_range, y_range)
        # Eigenvalues and Eigenvectors
        st.subheader("Eigenvalues and Eigenvectors of Symmetric Matrix")
        eigenvalues, eigenvectors = matrix_2d.eigenvalues()
        st.subheader("Eigenvalues")
        for idx, val in enumerate(eigenvalues):
            st.latex(r"\lambda_%d = %.2f" % (idx + 1, val.real))
        st.subheader("Eigenvectors")
        for idx, vec in enumerate(eigenvectors.T):
            st.latex(r"""\vec{{v}}_{%d} = \begin{pmatrix} %.2f \\ %.2f \end{pmatrix}""" %
                     (idx + 1, vec[0].real, vec[1].real))
        st.write(
            "Symmetric matrices have real eigenvalues and orthogonal eigenvectors. They represent transformations that are symmetrical with respect to the basis vectors."
        )

    # --- Transformation Explanations ---
    if selected_example_2d:
        st.header("2D Transformation Example Explanation")
        st.write(f"**{selected_example_2d.name}**:")
        st.latex(selected_example_2d.latex)
        st.write(selected_example_2d.explanation)

    if selected_symmetric_example_2d:
        st.header("Symmetric Transformation Explanation")
        st.write(f"**{selected_symmetric_example_2d.name}**:")
        st.latex(selected_symmetric_example_2d.latex)
        st.write(selected_symmetric_example_2d.explanation)

elif function_choice == "3D Transformation":
    # --- 3D Transformation ---
    # --- Sidebar Options ---
    st.sidebar.header("3D Transformation Examples")
    selected_example_3d_name = st.sidebar.selectbox(
        'Select a 3D Transformation Example',
        options=['None'] + list(TRANSFORMATIONS_3D.keys()),
        index=0)
    selected_example_3d = TRANSFORMATIONS_3D.get(selected_example_3d_name)

    # Initialize 3D Matrix
    if 'matrix_3d' not in st.session_state:
        st.session_state['matrix_3d'] = np.eye(3)

    # Apply or Update 3D Matrix
    if selected_example_3d:
        st.session_state['matrix_3d'] = selected_example_3d.matrix.data
    else:
        st.sidebar.header("3D Matrix Input")
        matrix_entries_3d = {}
        for i in range(3):
            for j in range(3):
                key = f'a{i+1}{j+1}_3d'
                value = float(st.session_state['matrix_3d'][i, j])
                matrix_entries_3d[key] = st.sidebar.number_input(
                    key, value=value, step=0.1, format="%.2f")
        st.session_state['matrix_3d'] = np.array([
            [
                matrix_entries_3d['a11_3d'], matrix_entries_3d['a12_3d'],
                matrix_entries_3d['a13_3d']
            ],
            [
                matrix_entries_3d['a21_3d'], matrix_entries_3d['a22_3d'],
                matrix_entries_3d['a23_3d']
            ],
            [
                matrix_entries_3d['a31_3d'], matrix_entries_3d['a32_3d'],
                matrix_entries_3d['a33_3d']
            ]
        ])

    # Display 3D Matrix
    st.sidebar.write("Current 3D Matrix:")
    st.sidebar.latex(
        r'''
        \begin{pmatrix}
        %.2f & %.2f & %.2f \\
        %.2f & %.2f & %.2f \\
        %.2f & %.2f & %.2f
        \end{pmatrix}
        ''' % tuple(st.session_state['matrix_3d'].flatten()))

    # 3D Vector Input
    st.sidebar.header("3D Vector Input")
    vector_3d = np.zeros(3)
    for i in range(3):
        key = f'v{i+1}_3d'
        default_value = 1.0 if i == 0 else 0.0
        vector_3d[i] = st.sidebar.number_input(key,
                                               value=float(default_value),
                                               step=0.1, format="%.2f")

    # 3D Vector Transformation
    matrix_3d = Matrix(st.session_state['matrix_3d'])
    vector_3d = Vector(vector_3d)
    transformed_vector_3d = matrix_3d @ vector_3d

    # 3D Vector Animation
    frames_3d = create_vector_animation_frames(matrix_3d, vector_3d, dims=3)
    create_vector_plot(vector_3d, transformed_vector_3d, frames_3d,
                       selected_theme, dims=3)

    # --- 3D Grid Transformation ---
    st.header("Geometric Transformations in ℝ³")
    # 3D Grid Settings
    st.sidebar.header("3D Grid Settings")
    grid_shape = st.sidebar.selectbox("Select 3D Shape",
                                      options=["Cube", "Sphere", "Cylinder"])
    grid_density = st.sidebar.slider("Grid Density",
                                     min_value=5,
                                     max_value=30,
                                     value=10,
                                     step=1)
    # Generate 3D Grid
    original_points_3d = generate_3d_shape(grid_shape, grid_density)
    transformed_points_3d = st.session_state['matrix_3d'] @ original_points_3d

    # 3D Grid Animation
    frames_grid_3d = create_grid_animation_frames(matrix_3d,
                                                  original_points_3d,
                                                  dims=3)
    create_grid_plot(original_points_3d,
                     transformed_points_3d,
                     frames_grid_3d,
                     selected_theme,
                     None,
                     None,
                     dims=3)

    # --- 3D Eigenvalues and Eigenvectors ---
    st.header("Eigenvalues and Eigenvectors (3D)")
    eigenvalues_3d, eigenvectors_3d = matrix_3d.eigenvalues()
    st.subheader("Eigenvalues")
    for idx, val in enumerate(eigenvalues_3d):
        st.latex(r"\lambda_%d = %.2f" % (idx + 1, val.real))
    st.subheader("Eigenvectors")
    for idx, vec in enumerate(eigenvectors_3d.T):
        st.latex(
            r"""\vec{{v}}_{%d} = \begin{pmatrix} %.2f \\ %.2f \\ %.2f \end{pmatrix}""" %
            (idx + 1, vec[0].real, vec[1].real, vec[2].real))
    st.write(
        r"""An **eigenvector** of a matrix \( A \) is a non-zero vector \( \vec{v} \) that, when \( A \) is applied to it, does not change direction. Instead, it is only scaled by a scalar factor called the **eigenvalue** \( \lambda \). This relationship is expressed as:"""
    )
    st.latex(r"A \vec{v} = \lambda \vec{v}")

    # --- Symmetric Matrices in 3D ---
    st.header("Symmetric Matrices in 3D")
    st.sidebar.header("Symmetric Matrices (3D)")
    selected_symmetric_example_3d_name = st.sidebar.selectbox(
        'Select a Symmetric 3D Transformation',
        options=['None'] + list(SYMMETRIC_TRANSFORMATIONS_3D.keys()),
        index=0)
    selected_symmetric_example_3d = SYMMETRIC_TRANSFORMATIONS_3D.get(selected_symmetric_example_3d_name)

    if selected_symmetric_example_3d:
        st.session_state['matrix_3d'] = selected_symmetric_example_3d.matrix.data
        # Display Symmetric Matrix
        st.write(f"**{selected_symmetric_example_3d.name}**:")
        st.latex(selected_symmetric_example_3d.latex)
        st.write(selected_symmetric_example_3d.explanation)
        # Recalculate transformations
        matrix_3d = Matrix(st.session_state['matrix_3d'])
        transformed_vector_3d = matrix_3d @ vector_3d
        transformed_points_3d = st.session_state['matrix_3d'] @ original_points_3d
        # Recalculate animations
        frames_3d = create_vector_animation_frames(matrix_3d, vector_3d, dims=3)
        create_vector_plot(vector_3d, transformed_vector_3d, frames_3d,
                           selected_theme, dims=3)
        frames_grid_3d = create_grid_animation_frames(matrix_3d,
                                                      original_points_3d,
                                                      dims=3)
        create_grid_plot(original_points_3d,
                         transformed_points_3d,
                         frames_grid_3d,
                         selected_theme,
                         None,
                         None,
                         dims=3)
        # Eigenvalues and Eigenvectors
        st.subheader("Eigenvalues and Eigenvectors of Symmetric Matrix")
        eigenvalues_3d, eigenvectors_3d = matrix_3d.eigenvalues()
        st.subheader("Eigenvalues")
        for idx, val in enumerate(eigenvalues_3d):
            st.latex(r"\lambda_%d = %.2f" % (idx + 1, val.real))
        st.subheader("Eigenvectors")
        for idx, vec in enumerate(eigenvectors_3d.T):
            st.latex(
                r"""\vec{{v}}_{%d} = \begin{pmatrix} %.2f \\ %.2f \\ %.2f \end{pmatrix}""" %
                (idx + 1, vec[0].real, vec[1].real, vec[2].real))
        st.write(
            "Symmetric matrices in 3D have real eigenvalues and orthogonal eigenvectors. They represent transformations that are symmetrical in all three dimensions."
        )

    # --- Transformation Explanations ---
    if selected_example_3d:
        st.header("3D Transformation Example Explanation")
        st.write(f"**{selected_example_3d.name}**:")
        st.latex(selected_example_3d.latex)
        st.write(selected_example_3d.explanation)

    if selected_symmetric_example_3d:
        st.header("Symmetric Transformation Explanation (3D)")
        st.write(f"**{selected_symmetric_example_3d.name}**:")
        st.latex(selected_symmetric_example_3d.latex)
        st.write(selected_symmetric_example_3d.explanation)

elif function_choice == "Principal Component Analysis (PCA)":
    # --- PCA Section ---
    st.header("Principal Component Analysis (PCA)")
    st.write("""
    PCA is a statistical procedure that uses an orthogonal transformation to convert a set of observations of possibly correlated variables into a set of values of linearly uncorrelated variables called principal components.

    PCA helps in reducing the dimensionality of data while retaining as much variance as possible.
    """)

    # PCA Options
    st.sidebar.header("PCA Settings")
    pca_option = st.sidebar.radio(
        "Choose the dimensionality reduction:",
        ("2D to 1D", "3D to 2D", "3D to 1D")
    )

    # Data Input
    st.sidebar.header("Data Input")
    data_option = st.sidebar.selectbox("Select Data Input Method",
                                       options=["Generate Synthetic Data", "Upload CSV File"])

    if data_option == "Generate Synthetic Data":
        # Generate Synthetic Data
        num_samples = st.sidebar.number_input("Number of Samples", min_value=10, max_value=1000, value=200, step=10)
        if pca_option == "2D to 1D":
            mean = [0, 0]
            cov = [[1, 0.9], [0.9, 1]]  # High positive correlation
            data = np.random.multivariate_normal(mean, cov, num_samples)
        elif pca_option == "3D to 2D" or pca_option == "3D to 1D":
            mean = [0, 0, 0]
            cov = [[1, 0.9, 0.7],
                   [0.9, 1, 0.5],
                   [0.7, 0.5, 1]]
            data = np.random.multivariate_normal(mean, cov, num_samples)
    else:
        # Upload CSV File
        uploaded_file = st.sidebar.file_uploader("Upload your CSV file", type=["csv"])
        if uploaded_file is not None:
            import pandas as pd
            data = pd.read_csv(uploaded_file)
            data = data.values
        else:
            st.warning("Please upload a CSV file.")
            st.stop()

    # Perform PCA
    if pca_option == "2D to 1D":
        data = data[:, :2]
        n_components = 2  # Keep both components to allow dynamic slider
    elif pca_option == "3D to 2D":
        data = data[:, :3]
        n_components = 3  # Keep all components for slider
    elif pca_option == "3D to 1D":
        data = data[:, :3]
        n_components = 3  # Keep all components for slider

    # PCA Computation
    pca_results = perform_pca(data, n_components)
    pca_model = pca_results['pca_model']
    components = pca_results['components']
    explained_variance = pca_results['explained_variance']
    cumulative_variance = pca_results['cumulative_variance']
    transformed_data = pca_results['transformed_data']

    # Show PCA Results
    st.subheader("Explained Variance by Principal Components")
    plot_explained_variance(explained_variance, cumulative_variance, selected_theme)

    # Dimension Slider
    max_dims = data.shape[1]
    dims_to_keep = st.slider("Select Number of Principal Components",
                             min_value=1,
                             max_value=max_dims,
                             value=max_dims,
                             step=1)

    # Adjust transformed data based on selected dimensions
    transformed_data_reduced = transformed_data[:, :dims_to_keep]
    components_reduced = components[:dims_to_keep, :]

    # Visualization
    st.subheader("Data Visualization")
    plot_pca_results(data, transformed_data_reduced, dims_to_keep, selected_theme, point_size)

    # Play Animation
    st.subheader("Projection Animation")
    animate_option = st.checkbox("Show Projection Animation")
    if animate_option:
        frames = create_pca_animation_frames(data, transformed_data_reduced, pca_model)
        plot_pca_animation(frames, selected_theme)

    # Allow users to input their own transformation matrix (components)
    st.sidebar.header("Custom Transformation Matrix")
    custom_matrix_option = st.sidebar.checkbox("Use Custom Transformation Matrix")
    if custom_matrix_option:
        st.write("Enter your custom transformation matrix:")
        num_rows = dims_to_keep
        num_cols = data.shape[1]
        custom_matrix = np.zeros((num_rows, num_cols))
        for i in range(num_rows):
            for j in range(num_cols):
                key = f"c_{i+1}{j+1}"
                default_value = components_reduced[i, j]
                custom_matrix[i, j] = st.number_input(key, value=float(default_value), step=0.1, format="%.2f")
        # Apply custom transformation
        transformed_data_custom = (data - np.mean(data, axis=0)) @ custom_matrix.T
        st.subheader("Custom Transformation Results")
        st.write("Custom Transformation Matrix:")
        st.write(custom_matrix)
        plot_pca_results(data, transformed_data_custom, dims_to_keep, selected_theme, point_size, custom=True)

    # PCA Explanation Sidebar
    st.sidebar.header("Understanding PCA")
    st.sidebar.write("""
    **Principal Component Analysis (PCA)** is a technique used to emphasize variation and bring out strong patterns in a dataset. It's often used to make data easy to explore and visualize.

    - **Principal Components (PCs):** New variables constructed as linear combinations of the original variables. They are orthogonal (uncorrelated) and capture the maximum variance.
    - **Explained Variance:** The amount of variance captured by each principal component.
    - **Dimensionality Reduction:** Reducing the number of variables under consideration by obtaining a set of principal variables.
    """)

    st.sidebar.subheader("Explained Variance Chart")
    st.sidebar.write("The bar chart shows the variance captured by each principal component. The cumulative variance line indicates the total variance captured up to that component.")

    st.sidebar.subheader("How to Use This App")
    st.sidebar.write("""
    1. **Select Dimensionality Reduction:** Choose how you want to reduce your data dimensions.
    2. **Input Data:** Generate synthetic data or upload your own CSV file.
    3. **Adjust Principal Components:** Use the slider to select the number of principal components to retain.
    4. **Visualize Results:** Observe the data projection in reduced dimensions.
    5. **Custom Transformation (Optional):** Input your own transformation matrix to see custom projections.
    """)

    # Aesthetic Customization
    st.sidebar.header("Aesthetic Customization")
    marker_size = st.sidebar.slider("Marker Size", min_value=2, max_value=20, value=point_size, step=1)
    st.session_state['marker_size'] = marker_size

    # Additional Explanations
    st.subheader("Understanding the Visualization")
    st.write(r"""
    - **Original Data Plot:** Shows the data in its original dimensions.
    - **Transformed Data Plot:** Displays the data projected onto the selected principal components.
    - **Projection Animation:** Animates the projection of data from the original space to the reduced space.
    - **Explained Variance Chart:** Highlights how much variance each principal component captures.
    """)

elif function_choice == "Singular Value Decomposition (SVD) Visualization":
    # --- SVD Visualization Section ---
    st.header("Singular Value Decomposition (SVD) Visualization")
    # Include educational information
    st.write("""
    ## What is SVD?
    Singular Value Decomposition (SVD) is a matrix factorization technique that decomposes a matrix into three component matrices:

    $$ A = U \Sigma V^T $$

    - **\( U \)**: Contains the left singular vectors.
    - **\( \Sigma \)**: Contains the singular values.
    - **\( V^T \)**: Contains the right singular vectors.

    ### Singular Values and Their Role
    Singular values represent the amount of variance captured along each corresponding singular vector. Truncating small singular values can reduce noise and help in dimensionality reduction.

    ### Applications of SVD
    SVD is widely used in:
    - Dimensionality Reduction
    - Image Compression
    - Noise Reduction
    - Recommender Systems
    - Natural Language Processing
    """)

    # SVD Options
    st.sidebar.header("SVD Settings")
    # Matrix Input
    st.sidebar.header("Matrix Input")
    matrix_option = st.sidebar.selectbox("Select Matrix Input Method",
                                         options=["Random Matrix", "Upload CSV File", "Pre-loaded Examples"])

    if matrix_option == "Random Matrix":
        # Matrix Size Selection
        matrix_size = st.sidebar.slider("Select Matrix Size (n x n)", min_value=2, max_value=7, value=3, step=1)
        # Generate Random Matrix
        np.random.seed(42)  # For reproducibility
        matrix_data = np.random.randn(matrix_size, matrix_size)
    elif matrix_option == "Upload CSV File":
        # Upload CSV File
        uploaded_file = st.sidebar.file_uploader("Upload your CSV file", type=["csv"])
        if uploaded_file is not None:
            matrix_df = pd.read_csv(uploaded_file, header=None)
            matrix_data = matrix_df.values
        else:
            st.warning("Please upload a CSV file.")
            st.stop()
    elif matrix_option == "Pre-loaded Examples":
        preloaded_options = ["Image Matrix", "Text Data Matrix"]
        selected_preloaded = st.sidebar.selectbox("Select Pre-loaded Matrix", options=preloaded_options)
        if selected_preloaded == "Image Matrix":
            from sklearn.datasets import load_sample_image
            image = load_sample_image("china.jpg")
            # Convert to grayscale
            image_gray = np.mean(image, axis=2)
            # Downsample for faster computation
            image_gray = image_gray[::2, ::2]
            matrix_data = image_gray
        elif selected_preloaded == "Text Data Matrix":
            # For simplicity, use a small term-document matrix
            matrix_data = np.array([
                [1, 2, 0, 0],
                [3, 0, 0, 0],
                [0, 0, 4, 5],
                [0, 0, 6, 7],
            ])
    else:
        st.warning("Please select a valid matrix input method.")
        st.stop()

    # Show Matrix Customization
    st.subheader("Input Matrix")
    if matrix_option == "Random Matrix":
        # Allow users to modify the matrix
        st.write("Modify the matrix entries:")
        matrix_entries = {}
        rows, cols = matrix_data.shape
        for i in range(rows):
            for j in range(cols):
                key = f'a{i+1}{j+1}'
                value = float(matrix_data[i, j])
                matrix_entries[key] = st.number_input(
                    key, value=value, step=0.1, format="%.2f")
        matrix_data = np.array([[matrix_entries[f'a{i+1}{j+1}'] for j in range(cols)] for i in range(rows)])
    st.write("Current Matrix:")
    st.write(matrix_data)

    # Perform SVD
    U, S, VT = perform_svd(matrix_data)

    # Component Selection
    st.sidebar.header("Component Selection")
    num_components = st.sidebar.slider("Number of Singular Values to Retain",
                                       min_value=1, max_value=len(S), value=len(S), step=1)

    # Truncated SVD
    reconstructed_matrix = reconstruct_matrix(U, S, VT, num_components)

    # Variance and Truncation Visualization
    st.subheader("Singular Values")
    st.write("Singular Values and their contribution to variance:")
    plot_svd_variance(S, selected_theme, num_components)

    # Reconstructed Matrix Display
    st.subheader("Reconstructed Matrix")
    st.write("Matrix reconstructed using selected singular values:")
    st.write(reconstructed_matrix)

    # Show Original and Reconstructed Matrix Side by Side
    st.subheader("Original vs Reconstructed Matrix")
    col1, col2 = st.columns(2)
    with col1:
        st.write("Original Matrix")
        st.write(matrix_data)
    with col2:
        st.write("Reconstructed Matrix")
        st.write(reconstructed_matrix)

    # Error Metrics and Visual Reconstruction Quality
    reconstruction_error = np.linalg.norm(matrix_data - reconstructed_matrix) / np.linalg.norm(matrix_data)
    st.write(f"Reconstruction Error (Relative Frobenius Norm): {reconstruction_error:.4f}")

    # 3D Dynamic Visualization of SVD Transformations
    st.subheader("3D Visualization of SVD Transformations")
    st.write("Visualizing the effect of SVD transformations (Rotation and Scaling)")

    # Create 3D visualization
    if matrix_data.shape[0] == 2 and matrix_data.shape[1] == 2:
        create_svd_3d_visualization(matrix_data, U, S, VT, selected_theme)
    else:
        st.write("3D visualization is only available for 2x2 matrices.")

    # Step-by-Step SVD Walkthrough with Descriptions
    st.subheader("Step-by-Step SVD Walkthrough")
    svd_step = st.slider("Select Step", min_value=1, max_value=4, value=1, step=1)
    if svd_step == 1:
        st.write("**Step 1: Original Matrix (A):**")
        st.write(matrix_data)
    elif svd_step == 2:
        st.write("**Step 2: Left Singular Vectors (U):**")
        st.write(U)
    elif svd_step == 3:
        st.write("**Step 3: Singular Values (Σ):**")
        st.write(np.diag(S))
    elif svd_step == 4:
        st.write("**Step 4: Right Singular Vectors (V^T):**")
        st.write(VT)

    # Pre-loaded Real-world Data Examples
    if matrix_option == "Pre-loaded Examples" and selected_preloaded == "Image Matrix":
        st.subheader("Original vs Reconstructed Image")
        col1, col2 = st.columns(2)
        with col1:
            st.write("Original Image")
            st.image(matrix_data, use_column_width=True, clamp=True)
        with col2:
            st.write("Reconstructed Image")
            st.image(reconstructed_matrix, use_column_width=True, clamp=True)

    # Comparison with PCA Visualization
    st.subheader("Comparison with PCA Visualization")
    comparison_option = st.checkbox("Show PCA Comparison")
    if comparison_option:
        # Perform PCA
        pca = PCA(n_components=num_components)
        pca_transformed = pca.fit_transform(matrix_data)
        pca_reconstructed = pca.inverse_transform(pca_transformed)
        st.write("**PCA Reconstructed Matrix:**")
        st.write(pca_reconstructed)
        # Show Error Metrics
        pca_reconstruction_error = np.linalg.norm(matrix_data - pca_reconstructed) / np.linalg.norm(matrix_data)
        st.write(f"PCA Reconstruction Error (Relative Frobenius Norm): {pca_reconstruction_error:.4f}")

    # Eigenvalues and Eigenvectors Explanation
    st.subheader("Eigenvalues and Eigenvectors Explanation")
    st.write("For symmetric matrices, the singular values are the absolute values of the eigenvalues.")

    # Modified Section Starts Here
    if matrix_data.shape[0] == matrix_data.shape[1]:
        if np.allclose(matrix_data, matrix_data.T):
            st.write("The matrix is symmetric.")
            eigenvalues, eigenvectors = np.linalg.eigh(matrix_data)
            st.write("Eigenvalues:")
            st.write(eigenvalues)
            st.write("Eigenvectors:")
            st.write(eigenvectors)
        else:
            st.write("The matrix is not symmetric.")
            eigenvalues, eigenvectors = np.linalg.eig(matrix_data)
            st.write("Eigenvalues (may be complex):")
            st.write(eigenvalues)
            st.write("Eigenvectors:")
            st.write(eigenvectors)
    else:
        st.write("The matrix is not square. Computing eigenvalues of A^T A.")
        ATA = np.dot(matrix_data.T, matrix_data)
        eigenvalues, eigenvectors = np.linalg.eigh(ATA)
        st.write("Eigenvalues of A^T A:")
        st.write(eigenvalues)
        st.write("Eigenvectors of A^T A:")
        st.write(eigenvectors)
    # Modified Section Ends Here

    # Error Metrics and Visual Reconstruction Quality
    st.subheader("Error Metrics and Visual Reconstruction Quality")
    st.write(f"Reconstruction Error (Frobenius Norm): {reconstruction_error:.4f}")
    error_metric_option = st.checkbox("Show Error Heatmap")
    if error_metric_option:
        error_matrix = np.abs(matrix_data - reconstructed_matrix)
        fig_error = go.Figure(data=go.Heatmap(
            z=error_matrix,
            colorscale='Reds'
        ))
        fig_error.update_layout(
            template=selected_theme,
            title='Reconstruction Error Heatmap'
        )
        st.plotly_chart(fig_error, use_container_width=True)

    # Save and Export Options
    st.subheader("Save and Export Options")
    save_option = st.checkbox("Save Reconstructed Matrix as CSV")
    if save_option:
        reconstructed_df = pd.DataFrame(reconstructed_matrix)
        csv = reconstructed_df.to_csv(index=False)
        st.download_button(label="Download CSV",
                           data=csv,
                           file_name='reconstructed_matrix.csv',
                           mime='text/csv')

    # Interface Customization
    st.sidebar.header("Interface Customization")
    interface_theme = st.sidebar.selectbox("Select Interface Theme",
                                           options=["Light", "Dark", "Dyslexia-Friendly"])
    if interface_theme == "Dyslexia-Friendly":
        st.write('<style>@import url("https://fonts.googleapis.com/css?family=OpenDyslexic");html, body, [class*="css"] {font-family: "OpenDyslexic", sans-serif;}</style>', unsafe_allow_html=True)
    elif interface_theme == "Dark":
        st.write('<style>body {background-color: #333; color: #fff;}</style>', unsafe_allow_html=True)

    # Animation Speed Control
    st.sidebar.header("Animation Controls")
    animation_speed = st.sidebar.slider("Animation Speed (ms)", min_value=100, max_value=2000, value=500, step=100)

    # User-friendly Controls
    st.write("Use the sliders and inputs in the sidebar to adjust settings and visualize different aspects of the SVD.")

    # SVD Animation
    st.subheader("SVD Transformation Animation")
    frames = create_svd_animation_frames(matrix_data, U, S, VT, steps=int(animation_speed / 100))
    plot_svd_animation(frames, selected_theme)

    # Keyboard Shortcuts and Accessibility
    st.write("Press 'P' to play/pause the animation.")
