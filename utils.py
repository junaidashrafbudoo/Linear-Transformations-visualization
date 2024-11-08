# utils.py

import plotly.graph_objects as go
import numpy as np
import streamlit as st

def create_vector_animation_frames(matrix, vector, steps=20, dims=2):
    """
    Creates animation frames for the transformation of a vector by a matrix.
    """
    frames = []
    for i in range(steps + 1):
        t = i / steps
        intermediate_matrix = np.eye(matrix.rows) * (1 - t) + matrix.data * t
        intermediate_vector = intermediate_matrix @ vector.data
        if dims == 2:
            frame_data = {
                'x': [0, intermediate_vector[0]],
                'y': [0, intermediate_vector[1]],
                'mode': 'lines+markers',
                'line': dict(color='red', width=3),
                'marker': dict(size=8)
            }
            frames.append(go.Frame(data=[go.Scatter(**frame_data)]))
        elif dims == 3:
            frame_data = {
                'x': [0, intermediate_vector[0]],
                'y': [0, intermediate_vector[1]],
                'z': [0, intermediate_vector[2]],
                'mode': 'lines+markers',
                'line': dict(color='red', width=5),
                'marker': dict(size=5)
            }
            frames.append(go.Frame(data=[go.Scatter3d(**frame_data)]))
    return frames

def create_grid_animation_frames(matrix, original_points, steps=20, dims=2):
    """
    Creates animation frames for the transformation of a grid of points by a matrix.
    """
    frames = []
    for i in range(steps + 1):
        t = i / steps
        intermediate_matrix = np.eye(matrix.rows) * (1 - t) + matrix.data * t
        intermediate_points = intermediate_matrix @ original_points
        if dims == 2:
            frame_data = {
                'x': intermediate_points[0, :],
                'y': intermediate_points[1, :],
                'mode': 'markers',
                'marker': dict(color='red', size=4)
            }
            frames.append(go.Frame(data=[go.Scatter(**frame_data)]))
        elif dims == 3:
            frame_data = {
                'x': intermediate_points[0, :],
                'y': intermediate_points[1, :],
                'z': intermediate_points[2, :],
                'mode': 'markers',
                'marker': dict(color='red', size=2)
            }
            frames.append(go.Frame(data=[go.Scatter3d(**frame_data)]))
    return frames

def generate_3d_shape(shape, density):
    """
    Generates a 3D grid of points for a given shape and density.
    """
    if shape == "Cube":
        lin = np.linspace(-1, 1, density)
        X, Y, Z = np.meshgrid(lin, lin, lin)
        points = np.vstack([X.ravel(), Y.ravel(), Z.ravel()])
    elif shape == "Sphere":
        phi = np.linspace(0, np.pi, density)
        theta = np.linspace(0, 2 * np.pi, density)
        phi, theta = np.meshgrid(phi, theta)
        X = np.sin(phi) * np.cos(theta)
        Y = np.sin(phi) * np.sin(theta)
        Z = np.cos(phi)
        points = np.vstack([X.ravel(), Y.ravel(), Z.ravel()])
    elif shape == "Cylinder":
        z = np.linspace(-1, 1, density)
        theta = np.linspace(0, 2 * np.pi, density)
        theta, z = np.meshgrid(theta, z)
        X = np.cos(theta)
        Y = np.sin(theta)
        Z = z
        points = np.vstack([X.ravel(), Y.ravel(), Z.ravel()])
    else:
        points = np.zeros((3, 0))
    return points

def generate_2d_shape(shape, density, range_min=-1, range_max=1):
    """
    Generates a 2D grid of points for a given shape and density.
    """
    if shape == "Rectangle":
        x = np.linspace(range_min, range_max, density)
        y = np.linspace(range_min, range_max, density)
        X, Y = np.meshgrid(x, y)
        points = np.vstack([X.ravel(), Y.ravel()])
    elif shape == "Triangle":
        x = np.linspace(range_min, range_max, density)
        y = np.linspace(range_min, range_max, density)
        X, Y = np.meshgrid(x, y)
        mask = Y >= X  # Condition for a right-angled triangle
        points = np.vstack([X[mask].ravel(), Y[mask].ravel()])
    elif shape == "Octagon":
        angles = np.linspace(0, 2 * np.pi, 8, endpoint=False)
        radius = (range_max - range_min) / 2
        x = radius * np.cos(angles)
        y = radius * np.sin(angles)
        points = np.vstack([x, y])
    else:
        points = np.zeros((2, 0))
    return points

def create_vector_plot(vector, transformed_vector, frames, theme, dims=2):
    """
    Creates a Plotly figure to visualize the transformation of a 2D or 3D vector.
    """
    max_val = max(np.abs(vector.data).max(),
                  np.abs(transformed_vector.data).max(), 1) * 1.2
    if dims == 2:
        fig = go.Figure(
            data=[
                go.Scatter(x=[0, vector.data[0]],
                           y=[0, vector.data[1]],
                           mode='lines+markers',
                           name='Original Vector',
                           line=dict(color='blue', width=3),
                           marker=dict(size=8))
            ],
            frames=frames)
        fig.update_layout(template=theme,
                          title='2D Vector Transformation Animation',
                          xaxis=dict(range=[-max_val, max_val],
                                     zeroline=True,
                                     showgrid=True,
                                     scaleanchor='y',
                                     scaleratio=1),
                          yaxis=dict(range=[-max_val, max_val],
                                     zeroline=True,
                                     showgrid=True),
                          updatemenus=[
                              dict(type='buttons',
                                   showactive=False,
                                   buttons=[
                                       dict(label='Play',
                                            method='animate',
                                            args=[
                                                None, {
                                                    'frame': {
                                                        'duration': 100,
                                                        'redraw': True
                                                    },
                                                    'fromcurrent': True,
                                                    'transition': {
                                                        'duration': 0
                                                    }
                                                }
                                            ]),
                                       dict(label='Pause',
                                            method='animate',
                                            args=[
                                                [None], {
                                                    'frame': {
                                                        'duration': 0,
                                                        'redraw': False
                                                    },
                                                    'mode': 'immediate',
                                                    'transition': {
                                                        'duration': 0
                                                    }
                                                }
                                            ])
                                   ])
                          ])
    elif dims == 3:
        fig = go.Figure(
            data=[
                go.Scatter3d(x=[0, vector.data[0]],
                             y=[0, vector.data[1]],
                             z=[0, vector.data[2]],
                             mode='lines+markers',
                             name='Original Vector',
                             line=dict(color='blue', width=5),
                             marker=dict(size=5))
            ],
            frames=frames)
        fig.update_layout(template=theme,
                          title='3D Vector Transformation Animation',
                          scene=dict(xaxis=dict(range=[-5, 5]),
                                     yaxis=dict(range=[-5, 5]),
                                     zaxis=dict(range=[-5, 5]),
                                     aspectmode='cube'),
                          updatemenus=[
                              dict(type='buttons',
                                   showactive=False,
                                   buttons=[
                                       dict(label='Play',
                                            method='animate',
                                            args=[
                                                None, {
                                                    'frame': {
                                                        'duration': 100,
                                                        'redraw': True
                                                    },
                                                    'fromcurrent': True,
                                                    'transition': {
                                                        'duration': 0
                                                    }
                                                }
                                            ]),
                                       dict(label='Pause',
                                            method='animate',
                                            args=[
                                                [None], {
                                                    'frame': {
                                                        'duration': 0,
                                                        'redraw': False
                                                    },
                                                    'mode': 'immediate',
                                                    'transition': {
                                                        'duration': 0
                                                    }
                                                }
                                            ])
                                   ])
                          ])
    st.plotly_chart(fig, use_container_width=True)

def create_grid_plot(original_points,
                     transformed_points,
                     frames,
                     theme,
                     x_range,
                     y_range,
                     dims=2):
    """
    Create a grid plot for 2D or 3D transformations.
    """
    if dims == 2:
        orig_fig = go.Figure()
        orig_fig.add_trace(go.Scatter(x=original_points[0, :],
                                      y=original_points[1, :],
                                      mode='markers',
                                      marker=dict(color='blue', size=4),
                                      name='Original Grid'))
        orig_fig.update_layout(template=theme,
                               title='Original Grid',
                               xaxis=dict(range=x_range,
                                          zeroline=True,
                                          showgrid=True,
                                          scaleanchor='y',
                                          scaleratio=1),
                               yaxis=dict(range=y_range,
                                          zeroline=True,
                                          showgrid=True),
                               margin=dict(l=50, r=50, t=50, b=50))

        trans_fig = go.Figure(
            data=[
                go.Scatter(x=original_points[0, :],
                           y=original_points[1, :],
                           mode='markers',
                           marker=dict(color='blue', size=4),
                           name='Original Grid')
            ],
            frames=frames)
        trans_fig.update_layout(template=theme,
                                title='Transformed Grid Animation',
                                xaxis=dict(range=x_range,
                                           zeroline=True,
                                           showgrid=True,
                                           scaleanchor='y',
                                           scaleratio=1),
                                yaxis=dict(range=y_range,
                                           zeroline=True,
                                           showgrid=True),
                                updatemenus=[
                                    dict(type='buttons',
                                         showactive=False,
                                         buttons=[
                                             dict(label='Play',
                                                  method='animate',
                                                  args=[
                                                      None, {
                                                          'frame': {
                                                              'duration': 100,
                                                              'redraw': True
                                                          },
                                                          'fromcurrent': True,
                                                          'transition': {
                                                              'duration': 0
                                                          }
                                                      }
                                                  ]),
                                             dict(
                                                 label='Pause',
                                                 method='animate',
                                                 args=[
                                                     [None], {
                                                         'frame': {
                                                             'duration': 0,
                                                             'redraw': False
                                                         },
                                                         'mode': 'immediate',
                                                         'transition': {
                                                             'duration': 0
                                                         }
                                                     }
                                                 ])
                                         ])
                                ],
                                margin=dict(l=50, r=50, t=50, b=50))
        col1, col2 = st.columns(2)
        with col1:
            st.plotly_chart(orig_fig, use_container_width=True)
            st.caption("Original Grid")
        with col2:
            st.plotly_chart(trans_fig, use_container_width=True)
            st.caption("Transformed Grid Animation")

    elif dims == 3:
        orig_fig_3d = go.Figure()
        orig_fig_3d.add_trace(go.Scatter3d(x=original_points[0, :],
                                           y=original_points[1, :],
                                           z=original_points[2, :],
                                           mode='markers',
                                           marker=dict(color='blue', size=2),
                                           name='Original Shape'))
        orig_fig_3d.update_layout(template=theme,
                                  title='Original 3D Shape',
                                  scene=dict(xaxis=dict(range=[-2, 2]),
                                             yaxis=dict(range=[-2, 2]),
                                             zaxis=dict(range=[-2, 2]),
                                             aspectmode='cube'),
                                  margin=dict(l=50, r=50, t=50, b=50))

        trans_fig_3d = go.Figure(
            data=[
                go.Scatter3d(x=original_points[0, :],
                             y=original_points[1, :],
                             z=original_points[2, :],
                             mode='markers',
                             marker=dict(color='blue', size=2),
                             name='Original Shape')
            ],
            frames=frames)
        trans_fig_3d.update_layout(template=theme,
                                   title='Transformed 3D Shape Animation',
                                   scene=dict(xaxis=dict(range=[-2, 2]),
                                              yaxis=dict(range=[-2, 2]),
                                              zaxis=dict(range=[-2, 2]),
                                              aspectmode='cube'),
                                   updatemenus=[
                                       dict(type='buttons',
                                            showactive=False,
                                            buttons=[
                                                dict(label='Play',
                                                     method='animate',
                                                     args=[
                                                         None, {
                                                             'frame': {
                                                                 'duration':
                                                                 100,
                                                                 'redraw': True
                                                             },
                                                             'fromcurrent':
                                                             True,
                                                             'transition': {
                                                                 'duration': 0
                                                             }
                                                         }
                                                     ]),
                                                dict(
                                                    label='Pause',
                                                    method='animate',
                                                    args=[
                                                        [None], {
                                                            'frame': {
                                                                'duration': 0,
                                                                'redraw': False
                                                            },
                                                            'mode': 'immediate',
                                                            'transition': {
                                                                'duration': 0
                                                            }
                                                        }
                                                    ])
                                            ])
                                   ],
                                   margin=dict(l=50, r=50, t=50, b=50))

        col1, col2 = st.columns(2)
        with col1:
            st.plotly_chart(orig_fig_3d, use_container_width=True)
            st.caption("Original 3D Shape")
        with col2:
            st.plotly_chart(trans_fig_3d, use_container_width=True)
            st.caption("Transformed 3D Shape Animation")

def perform_pca(data, n_components):
    """
    Perform PCA on the given data.
    """
    from sklearn.decomposition import PCA

    pca = PCA(n_components=n_components)
    transformed_data = pca.fit_transform(data)
    components = pca.components_
    explained_variance = pca.explained_variance_ratio_
    cumulative_variance = np.cumsum(explained_variance)

    return {
        'pca_model': pca,
        'components': components,
        'explained_variance': explained_variance,
        'cumulative_variance': cumulative_variance,
        'transformed_data': transformed_data
    }

def plot_pca_results(original_data, transformed_data, n_components, theme, point_size, custom=False):
    """
    Plot the original data and the PCA transformed data.
    """
    cols = st.columns(2)
    with cols[0]:
        # Original Data Plot
        if original_data.shape[1] == 2:
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=original_data[:, 0],
                y=original_data[:, 1],
                mode='markers',
                name='Original Data',
                marker=dict(color='blue', size=point_size),
                hovertemplate='(%{x:.2f}, %{y:.2f})<extra></extra>'
            ))
            fig.update_layout(
                template=theme,
                title='Original Data (2D)',
                xaxis_title='X-axis',
                yaxis_title='Y-axis',
                showlegend=False
            )
            st.plotly_chart(fig, use_container_width=True)
        elif original_data.shape[1] == 3:
            fig = go.Figure()
            fig.add_trace(go.Scatter3d(
                x=original_data[:, 0],
                y=original_data[:, 1],
                z=original_data[:, 2],
                mode='markers',
                name='Original Data',
                marker=dict(color='blue', size=point_size),
                hovertemplate='(%{x:.2f}, %{y:.2f}, %{z:.2f})<extra></extra>'
            ))
            fig.update_layout(
                template=theme,
                title='Original Data (3D)',
                scene=dict(
                    xaxis_title='X-axis',
                    yaxis_title='Y-axis',
                    zaxis_title='Z-axis'
                ),
                showlegend=False
            )
            st.plotly_chart(fig, use_container_width=True)

    with cols[1]:
        # Transformed Data Plot
        if n_components == 1:
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=transformed_data[:, 0],
                y=np.zeros_like(transformed_data[:, 0]),
                mode='markers',
                name='Transformed Data',
                marker=dict(color='red', size=point_size),
                hovertemplate='PC1: %{x:.2f}<extra></extra>'
            ))
            fig.update_layout(
                template=theme,
                title='Transformed Data (1D)',
                xaxis_title='Principal Component 1',
                yaxis_title='',
                showlegend=False,
                yaxis=dict(showticklabels=False)
            )
            st.plotly_chart(fig, use_container_width=True)
        elif n_components == 2:
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=transformed_data[:, 0],
                y=transformed_data[:, 1],
                mode='markers',
                name='Transformed Data',
                marker=dict(color='red', size=point_size),
                hovertemplate='PC1: %{x:.2f}<br>PC2: %{y:.2f}<extra></extra>'
            ))
            fig.update_layout(
                template=theme,
                title='Transformed Data (2D)',
                xaxis_title='Principal Component 1',
                yaxis_title='Principal Component 2',
                showlegend=False
            )
            st.plotly_chart(fig, use_container_width=True)
        elif n_components == 3:
            fig = go.Figure()
            fig.add_trace(go.Scatter3d(
                x=transformed_data[:, 0],
                y=transformed_data[:, 1],
                z=transformed_data[:, 2],
                mode='markers',
                name='Transformed Data',
                marker=dict(color='red', size=point_size),
                hovertemplate='PC1: %{x:.2f}<br>PC2: %{y:.2f}<br>PC3: %{z:.2f}<extra></extra>'
            ))
            fig.update_layout(
                template=theme,
                title='Transformed Data (3D)',
                scene=dict(
                    xaxis_title='Principal Component 1',
                    yaxis_title='Principal Component 2',
                    zaxis_title='Principal Component 3'
                ),
                showlegend=False
            )
            st.plotly_chart(fig, use_container_width=True)

def plot_explained_variance(explained_variance, cumulative_variance, theme):
    """
    Plot the explained variance bar chart.
    """
    num_components = len(explained_variance)
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=[f'PC{i+1}' for i in range(num_components)],
        y=explained_variance,
        name='Individual Explained Variance',
        marker_color='blue',
        hovertemplate='PC%{x}: %{y:.2%}<extra></extra>'
    ))
    fig.add_trace(go.Scatter(
        x=[f'PC{i+1}' for i in range(num_components)],
        y=cumulative_variance,
        name='Cumulative Explained Variance',
        marker_color='red',
        mode='lines+markers',
        hovertemplate='Cumulative Variance: %{y:.2%}<extra></extra>'
    ))
    fig.update_layout(
        template=theme,
        yaxis=dict(title='Explained Variance Ratio'),
        xaxis=dict(title='Principal Components'),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        showlegend=True
    )
    st.plotly_chart(fig, use_container_width=True)

def create_pca_animation_frames(original_data, transformed_data, pca_model, steps=20):
    """
    Create frames for PCA projection animation.
    """
    frames = []
    n_components = transformed_data.shape[1]
    for i in range(steps + 1):
        t = i / steps

        # Reconstruct the transformed data back to original feature space
        transformed_data_full = transformed_data @ pca_model.components_[:n_components, :]
        intermediate_data = (1 - t) * original_data + t * (transformed_data_full + pca_model.mean_)

        if original_data.shape[1] == 2:
            frame = go.Frame(
                data=[
                    go.Scatter(
                        x=intermediate_data[:, 0],
                        y=intermediate_data[:, 1],
                        mode='markers',
                        marker=dict(color='green', size=5),
                        hovertemplate='(%{x:.2f}, %{y:.2f})<extra></extra>'
                    )
                ],
                name=str(i)
            )
        elif original_data.shape[1] == 3:
            frame = go.Frame(
                data=[
                    go.Scatter3d(
                        x=intermediate_data[:, 0],
                        y=intermediate_data[:, 1],
                        z=intermediate_data[:, 2],
                        mode='markers',
                        marker=dict(color='green', size=5),
                        hovertemplate='(%{x:.2f}, %{y:.2f}, %{z:.2f})<extra></extra>'
                    )
                ],
                name=str(i)
            )
        frames.append(frame)
    return frames

def plot_pca_animation(frames, theme):
    """
    Plot the PCA projection animation.
    """
    if not frames:
        st.write("Animation not available for the selected data.")
        return

    first_frame = frames[0].data[0]
    if isinstance(first_frame, go.Scatter):
        fig = go.Figure(
            data=frames[0].data,
            frames=frames
        )
        fig.update_layout(
            template=theme,
            xaxis_title='X-axis',
            yaxis_title='Y-axis',
            updatemenus=[
                dict(type='buttons',
                     showactive=False,
                     buttons=[
                         dict(label='Play',
                              method='animate',
                              args=[
                                  None, {
                                      'frame': {'duration': 100, 'redraw': True},
                                      'fromcurrent': True,
                                      'transition': {'duration': 0}
                                  }
                              ]),
                         dict(label='Pause',
                              method='animate',
                              args=[
                                  [None], {
                                      'frame': {'duration': 0, 'redraw': False},
                                      'mode': 'immediate',
                                      'transition': {'duration': 0}
                                  }
                              ])
                     ])
            ]
        )
    elif isinstance(first_frame, go.Scatter3d):
        fig = go.Figure(
            data=frames[0].data,
            frames=frames
        )
        fig.update_layout(
            template=theme,
            scene=dict(
                xaxis_title='X-axis',
                yaxis_title='Y-axis',
                zaxis_title='Z-axis'
            ),
            updatemenus=[
                dict(type='buttons',
                     showactive=False,
                     buttons=[
                         dict(label='Play',
                              method='animate',
                              args=[
                                  None, {
                                      'frame': {'duration': 100, 'redraw': True},
                                      'fromcurrent': True,
                                      'transition': {'duration': 0}
                                  }
                              ]),
                         dict(label='Pause',
                              method='animate',
                              args=[
                                  [None], {
                                      'frame': {'duration': 0, 'redraw': False},
                                      'mode': 'immediate',
                                      'transition': {'duration': 0}
                                  }
                              ])
                     ])
            ]
        )
    st.plotly_chart(fig, use_container_width=True)

# SVD Functions
def perform_svd(matrix_data):
    """
    Perform Singular Value Decomposition (SVD) on the given matrix.
    """
    U, S, VT = np.linalg.svd(matrix_data, full_matrices=False)
    return U, S, VT

def reconstruct_matrix(U, S, VT, num_components):
    """
    Reconstruct the matrix using a specified number of singular values.
    """
    U_truncated = U[:, :num_components]
    S_truncated = S[:num_components]
    VT_truncated = VT[:num_components, :]
    reconstructed_matrix = U_truncated @ np.diag(S_truncated) @ VT_truncated
    return reconstructed_matrix

def plot_svd_variance(S, theme, num_components):
    """
    Plot the singular values and their explained variance.
    """
    explained_variance = S**2 / np.sum(S**2)
    cumulative_variance = np.cumsum(explained_variance)
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=[f'SV{i+1}' for i in range(len(S))],
        y=explained_variance,
        name='Individual Explained Variance',
        marker_color=['blue' if i < num_components else 'gray' for i in range(len(S))],
        hovertemplate='SV%{x}: %{y:.2%}<extra></extra>'
    ))
    fig.add_trace(go.Scatter(
        x=[f'SV{i+1}' for i in range(len(S))],
        y=cumulative_variance,
        name='Cumulative Explained Variance',
        marker_color='red',
        mode='lines+markers',
        hovertemplate='Cumulative Variance: %{y:.2%}<extra></extra>'
    ))
    fig.update_layout(
        template=theme,
        yaxis=dict(title='Explained Variance Ratio'),
        xaxis=dict(title='Singular Values'),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        showlegend=True
    )
    st.plotly_chart(fig, use_container_width=True)

def create_svd_3d_visualization(matrix_data, U, S, VT, theme):
    """
    Create a 3D visualization of SVD transformations.
    """
    # For visualization, we'll assume the matrix is 2x2
    if matrix_data.shape[0] != 2 or matrix_data.shape[1] != 2:
        st.write("3D visualization is only available for 2x2 matrices.")
        return

    # Generate a circle of points
    theta = np.linspace(0, 2 * np.pi, 100)
    circle = np.vstack((np.cos(theta), np.sin(theta)))

    # Apply transformations
    V_points = VT.T @ circle
    S_points = np.diag(S) @ V_points
    U_points = U @ S_points

    # Create interactive 3D plot
    fig = go.Figure()

    # Original Circle
    fig.add_trace(go.Scatter3d(
        x=circle[0, :],
        y=circle[1, :],
        z=np.zeros(circle.shape[1]),
        mode='lines',
        name='Original Circle',
        line=dict(color='blue')
    ))

    # After V^T Transformation
    fig.add_trace(go.Scatter3d(
        x=V_points[0, :],
        y=V_points[1, :],
        z=np.zeros(V_points.shape[1]) + 1,
        mode='lines',
        name='After V^T',
        line=dict(color='green')
    ))

    # After Σ Transformation
    fig.add_trace(go.Scatter3d(
        x=S_points[0, :],
        y=S_points[1, :],
        z=np.zeros(S_points.shape[1]) + 2,
        mode='lines',
        name='After Σ',
        line=dict(color='orange')
    ))

    # After U Transformation
    fig.add_trace(go.Scatter3d(
        x=U_points[0, :],
        y=U_points[1, :],
        z=np.zeros(U_points.shape[1]) + 3,
        mode='lines',
        name='After U',
        line=dict(color='red')
    ))

    fig.update_layout(
        template=theme,
        scene=dict(
            xaxis_title='X-axis',
            yaxis_title='Y-axis',
            zaxis_title='Transformation Step',
            aspectmode='cube'
        ),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    st.plotly_chart(fig, use_container_width=True)

def create_svd_animation_frames(matrix_data, U, S, VT, steps=20):
    """
    Create frames for SVD animation.
    """
    # For visualization, we'll assume the matrix is 2x2
    if matrix_data.shape[0] != 2 or matrix_data.shape[1] != 2:
        return []

    # Generate a circle of points
    theta = np.linspace(0, 2 * np.pi, 100)
    circle = np.vstack((np.cos(theta), np.sin(theta)))

    frames = []
    for i in range(steps + 1):
        t = i / steps

        # Interpolate between transformations
        Vt_interp = np.eye(2) * (1 - t) + VT.T * t
        V_points = Vt_interp @ circle

        S_interp = np.eye(2) * (1 - t) + np.diag(S) * t
        S_points = S_interp @ V_points

        U_interp = np.eye(2) * (1 - t) + U * t
        U_points = U_interp @ S_points

        frame = go.Frame(
            data=[
                go.Scatter(
                    x=U_points[0, :],
                    y=U_points[1, :],
                    mode='lines',
                    line=dict(color='red')
                )
            ],
            name=str(i)
        )
        frames.append(frame)
    return frames

def plot_svd_animation(frames, theme):
    """
    Plot the SVD transformation animation.
    """
    if not frames:
        st.write("Animation not available for the selected matrix.")
        return

    fig = go.Figure(
        data=frames[0].data,
        frames=frames
    )
    fig.update_layout(
        template=theme,
        xaxis_title='X-axis',
        yaxis_title='Y-axis',
        xaxis=dict(range=[-3, 3]),
        yaxis=dict(range=[-3, 3]),
        updatemenus=[
            dict(type='buttons',
                 showactive=False,
                 buttons=[
                     dict(label='Play',
                          method='animate',
                          args=[
                              None, {
                                  'frame': {'duration': 100, 'redraw': True},
                                  'fromcurrent': True,
                                  'transition': {'duration': 0}
                              }
                          ]),
                     dict(label='Pause',
                          method='animate',
                          args=[
                              [None], {
                                  'frame': {'duration': 0, 'redraw': False},
                                  'mode': 'immediate',
                                  'transition': {'duration': 0}
                              }
                          ])
                 ])
        ]
    )
    st.plotly_chart(fig, use_container_width=True)
