"""
Scientific Visualization for Spatial Accuracy Patterns.

Adapted from OG_transformer.plot.accuracy_sparsity_map
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import gridspec
from sklearn.metrics import r2_score


# ============================================================
# Helper Functions
# ============================================================

def calculate_station_r2(df_analysis, model_name, sufficiency=None):
    """Calculate R² for each station."""
    df_plot = df_analysis.copy()
    
    if sufficiency is not None and 'sufficiency' in df_plot.columns:
        df_plot = df_plot[df_plot['sufficiency'] == sufficiency]
    
    predicted_col = f'predicted_{model_name}'
    if predicted_col not in df_plot.columns:
        raise ValueError(f"Column '{predicted_col}' not found")
    
    if 'observed' not in df_plot.columns:
        raise ValueError("Column 'observed' not found")
    
    station_r2_list = []
    for (lon, lat), group in df_plot.groupby(['longitude', 'latitude']):
        if len(group) > 1:
            r2 = r2_score(group['observed'], group[predicted_col])
            station_r2_list.append({'longitude': lon, 'latitude': lat, 'r2': r2})
    
    return pd.DataFrame(station_r2_list)


def plot_observation_points(ax, station_data, cmap, vmin, vmax, marker_size=15):
    """Plot station observation points."""
    if len(station_data) == 0:
        return None
    
    # Filter outliers
    station_data = station_data[(station_data['r2'] >= -0.5) & (station_data['r2'] <= 1.0)].copy()
    if len(station_data) == 0:
        return None
    
    scatter = ax.scatter(
        station_data['longitude'], station_data['latitude'],
        c=station_data['r2'], cmap=cmap, vmin=vmin, vmax=vmax,
        s=marker_size, alpha=0.8, edgecolors='black', linewidths=0.5
    )
    return scatter


def plot_grid_average(ax, df_analysis, model_name, sufficiency, cmap, vmin, vmax, 
                      grid_shape=(15, 20), lon_range=None, lat_range=None):
    """Plot grid-averaged R²."""
    df_plot = df_analysis.copy()
    
    if sufficiency is not None and 'sufficiency' in df_plot.columns:
        df_plot = df_plot[df_plot['sufficiency'] == sufficiency]
    
    predicted_col = f'predicted_{model_name}'
    if predicted_col not in df_plot.columns or len(df_plot) == 0:
        return None
    
    n_lat_bins, n_lon_bins = grid_shape
    
    if lon_range is None:
        lon_min, lon_max = df_plot['longitude'].min(), df_plot['longitude'].max()
    else:
        lon_min, lon_max = lon_range
    if lat_range is None:
        lat_min, lat_max = df_plot['latitude'].min(), df_plot['latitude'].max()
    else:
        lat_min, lat_max = lat_range
    
    lon_bins = np.linspace(lon_min, lon_max, n_lon_bins + 1)
    lat_bins = np.linspace(lat_min, lat_max, n_lat_bins + 1)
    
    df_plot = df_plot.copy()
    df_plot['lon_bin'] = np.digitize(df_plot['longitude'], lon_bins) - 1
    df_plot['lat_bin'] = np.digitize(df_plot['latitude'], lat_bins) - 1
    df_plot['lon_bin'] = df_plot['lon_bin'].clip(0, n_lon_bins - 1)
    df_plot['lat_bin'] = df_plot['lat_bin'].clip(0, n_lat_bins - 1)
    
    grid_r2 = np.full((n_lat_bins, n_lon_bins), np.nan)
    
    for (lat_bin, lon_bin), group in df_plot.groupby(['lat_bin', 'lon_bin']):
        if len(group) > 1:
            r2 = r2_score(group['observed'], group[predicted_col])
            grid_r2[int(lat_bin), int(lon_bin)] = np.clip(r2, -0.5, 1.0)
    
    mesh = ax.pcolormesh(lon_bins, lat_bins, grid_r2, cmap=cmap,
                         vmin=vmin, vmax=vmax, alpha=0.8, shading='flat')
    return mesh


def plot_interpolation(ax, df_analysis, model_name, sufficiency, cmap, vmin, vmax,
                       grid_shape=(30, 40), lon_range=None, lat_range=None, power=2):
    """Plot IDW interpolation of station R² values."""
    station_df = calculate_station_r2(df_analysis, model_name, sufficiency)
    
    if len(station_df) < 3:
        return None
    
    # Filter outliers
    station_df = station_df[(station_df['r2'] >= -0.5) & (station_df['r2'] <= 1.0)].copy()
    if len(station_df) < 3:
        return None
    
    if lon_range is None:
        lon_range = (station_df['longitude'].min() - 1, station_df['longitude'].max() + 1)
    if lat_range is None:
        lat_range = (station_df['latitude'].min() - 1, station_df['latitude'].max() + 1)
    
    n_lat, n_lon = grid_shape
    lon_grid = np.linspace(lon_range[0], lon_range[1], n_lon)
    lat_grid = np.linspace(lat_range[0], lat_range[1], n_lat)
    
    station_lons = station_df['longitude'].values
    station_lats = station_df['latitude'].values
    station_r2 = station_df['r2'].values.astype(np.float64)
    
    r2_grid = np.zeros((n_lat, n_lon))
    
    for i in range(n_lat):
        for j in range(n_lon):
            grid_lon = lon_grid[j]
            grid_lat = lat_grid[i]
            distances = np.sqrt((station_lons - grid_lon)**2 + (station_lats - grid_lat)**2)
            
            if np.any(distances == 0):
                r2_grid[i, j] = station_r2[distances == 0][0]
            else:
                weights = 1.0 / (distances ** power)
                weights = weights / np.sum(weights)
                r2_grid[i, j] = np.sum(weights * station_r2)
    
    levels = np.linspace(vmin, vmax, 21)
    mesh = ax.contourf(lon_grid, lat_grid, r2_grid, levels=levels, cmap=cmap, extend='both')
    ax.scatter(station_lons, station_lats, c='black', s=5, alpha=0.3)
    
    return mesh


def plot_spatial_model(ax, df_analysis, model_name, sufficiency, cmap, vmin, vmax,
                       grid_shape=(30, 40), lon_range=None, lat_range=None):
    """Plot SVM spatial regression prediction."""
    from sklearn.svm import SVR
    from sklearn.preprocessing import StandardScaler
    
    station_df = calculate_station_r2(df_analysis, model_name, sufficiency)
    station_df = station_df[(station_df['r2'] >= -0.5) & (station_df['r2'] <= 1.0)].copy()
    
    if len(station_df) < 5:
        return None
    
    X = station_df[['longitude', 'latitude']].values
    y = station_df['r2'].values
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    model = SVR(kernel='rbf', C=1, gamma='scale')
    model.fit(X_scaled, y)
    
    if lon_range is None:
        lon_range = (X[:, 0].min() - 1, X[:, 0].max() + 1)
    if lat_range is None:
        lat_range = (X[:, 1].min() - 1, X[:, 1].max() + 1)
    
    n_lat, n_lon = grid_shape
    lon_grid = np.linspace(lon_range[0], lon_range[1], n_lon)
    lat_grid = np.linspace(lat_range[0], lat_range[1], n_lat)
    lon_mesh, lat_mesh = np.meshgrid(lon_grid, lat_grid)
    
    grid_coords = np.column_stack([lon_mesh.ravel(), lat_mesh.ravel()])
    grid_scaled = scaler.transform(grid_coords)
    
    r2_pred = model.predict(grid_scaled)
    r2_grid = np.clip(r2_pred.reshape(n_lat, n_lon), vmin, vmax)
    
    levels = np.linspace(vmin, vmax, 21)
    mesh = ax.contourf(lon_grid, lat_grid, r2_grid, levels=levels, cmap=cmap, extend='both')
    ax.scatter(station_df['longitude'], station_df['latitude'], c='black', s=5, alpha=0.3)
    
    return mesh


# ============================================================
# Main Visualization Functions
# ============================================================

def plot_accuracy_map(
    df_analysis,
    model_name,
    mode='observation',
    sufficiency=None,
    states=None,
    accuracy_range=(0, 1),
    lon_range=None,
    lat_range=None,
    grid_shape=(20, 25),
    cmap_name='Spectral_r',
    figsize=(8, 6),
    title=None,
    save_path=None,
    show_plot=True
):
    """
    Plot accuracy map with different visualization modes.
    """
    vmin, vmax = accuracy_range
    cmap = plt.get_cmap(cmap_name)
    
    fig, ax = plt.subplots(figsize=figsize)
    
    if states is not None:
        plot_multipolygon_edges(ax, states, edge_color='grey', linewidth=1, alpha=0.3)
    
    if lon_range is None:
        lon_range = (df_analysis['longitude'].min() - 1, df_analysis['longitude'].max() + 1)
    if lat_range is None:
        lat_range = (df_analysis['latitude'].min() - 1, df_analysis['latitude'].max() + 1)
    
    if mode == 'observation':
        station_data = calculate_station_r2(df_analysis, model_name, sufficiency)
        scatter = plot_observation_points(ax, station_data, cmap, vmin, vmax)
        if scatter:
            plt.colorbar(scatter, ax=ax, label='R²')
        mode_label = 'Observation'
        
    elif mode == 'interpolation':
        mesh = plot_interpolation(ax, df_analysis, model_name, sufficiency, cmap, vmin, vmax,
                                  grid_shape=grid_shape, lon_range=lon_range, lat_range=lat_range)
        if mesh:
            plt.colorbar(mesh, ax=ax, label='R² (IDW)')
        mode_label = 'IDW Interpolation'
        
    elif mode.startswith('average'):
        if '_' in mode:
            try:
                grid_str = mode.split('_')[1]
                n_lat, n_lon = map(int, grid_str.split('x'))
                grid_shape = (n_lat, n_lon)
            except:
                pass
        
        mesh = plot_grid_average(ax, df_analysis, model_name, sufficiency, cmap, vmin, vmax,
                                 grid_shape=grid_shape, lon_range=lon_range, lat_range=lat_range)
        if mesh:
            plt.colorbar(mesh, ax=ax, label='R² (Grid Average)')
        mode_label = f'Grid Average ({grid_shape[0]}×{grid_shape[1]})'
        
    elif mode == 'spatial_model':
        mesh = plot_spatial_model(ax, df_analysis, model_name, sufficiency, cmap, vmin, vmax,
                                  grid_shape=grid_shape, lon_range=lon_range, lat_range=lat_range)
        if mesh:
            plt.colorbar(mesh, ax=ax, label='R² (SVM)')
        mode_label = 'Spatial Model (SVM)'
        
    else:
        raise ValueError(f"Unknown mode: {mode}")
    
    ax.set_xlim(lon_range)
    ax.set_ylim(lat_range)
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    ax.set_title(title if title else f'{model_name} - {mode_label}')
    ax.grid(True, alpha=0.2)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    if show_plot:
        plt.show()
    
    return fig, ax


def plot_accuracy_comparison(
    df_analysis,
    model_name,
    modes=['observation', 'interpolation', 'average', 'spatial_model'],
    sufficiency=None,
    accuracy_range=(0, 1),
    lon_range=None,
    lat_range=None,
    grid_shape=(20, 25),
    cmap_name='Spectral_r',
    figsize_per_plot=4,
    height=4,
    save_path=None,
    show_plot=True,
    ts_model=None,
    station_lons=None,
    station_lats=None,
    radius=500
):
    """Plot multiple visualization modes side by side.
    
    For 'two_stage' mode, ts_model, station_lons, station_lats are required.
    """
    n_plots = len(modes)
    vmin, vmax = accuracy_range
    cmap = plt.get_cmap(cmap_name)
    
    fig = plt.figure(figsize=(figsize_per_plot * n_plots + 0.5, height))
    gs = gridspec.GridSpec(1, n_plots + 1, width_ratios=[1] * n_plots + [0.05], wspace=0.03)
    
    axes = [fig.add_subplot(gs[0, i]) for i in range(n_plots)]
    cax = fig.add_subplot(gs[0, -1])
    
    if lon_range is None:
        lon_range = (df_analysis['longitude'].min() - 1, df_analysis['longitude'].max() + 1)
    if lat_range is None:
        lat_range = (df_analysis['latitude'].min() - 1, df_analysis['latitude'].max() + 1)
    
    for i, (ax, mode) in enumerate(zip(axes, modes)):
        if mode == 'observation':
            station_data = calculate_station_r2(df_analysis, model_name, sufficiency)
            plot_observation_points(ax, station_data, cmap, vmin, vmax)
            title = 'Observation'
            
        elif mode == 'interpolation':
            plot_interpolation(ax, df_analysis, model_name, sufficiency, cmap, vmin, vmax,
                              grid_shape=grid_shape, lon_range=lon_range, lat_range=lat_range)
            title = 'IDW'
            
        elif mode.startswith('average'):
            gs_local = grid_shape
            if '_' in mode:
                try:
                    grid_str = mode.split('_')[1]
                    n_lat, n_lon = map(int, grid_str.split('x'))
                    gs_local = (n_lat, n_lon)
                except:
                    pass
            plot_grid_average(ax, df_analysis, model_name, sufficiency, cmap, vmin, vmax,
                             grid_shape=gs_local, lon_range=lon_range, lat_range=lat_range)
            title = f'Average ({gs_local[0]}×{gs_local[1]})'
            
        elif mode == 'spatial_model':
            plot_spatial_model(ax, df_analysis, model_name, sufficiency, cmap, vmin, vmax,
                              grid_shape=grid_shape, lon_range=lon_range, lat_range=lat_range)
            title = 'SVM'
        
        elif mode == 'two_stage':
            if ts_model is None:
                raise ValueError("ts_model is required for 'two_stage' mode")
            if station_lons is None or station_lats is None:
                raise ValueError("station_lons and station_lats are required for 'two_stage' mode")
            
            # Import density calculation
            try:
                from .data import calculate_density_at_locations
            except ImportError:
                from data import calculate_density_at_locations
            
            # Create grid
            n_lon, n_lat = grid_shape[1], grid_shape[0]
            lon_grid = np.linspace(lon_range[0], lon_range[1], n_lon)
            lat_grid = np.linspace(lat_range[0], lat_range[1], n_lat)
            lon_mesh, lat_mesh = np.meshgrid(lon_grid, lat_grid)
            
            # Calculate density for each pixel
            pixel_densities = calculate_density_at_locations(
                lon_mesh.ravel(), lat_mesh.ravel(),
                station_lons, station_lats, radius=radius
            )
            
            # Predict
            suff = sufficiency if sufficiency is not None else df_analysis['sufficiency'].iloc[0]
            r2_grid = ts_model.predict(
                longitude=lon_mesh.ravel(),
                latitude=lat_mesh.ravel(),
                density=pixel_densities,
                sufficiency=suff
            )
            r2_grid = np.clip(r2_grid.reshape(n_lat, n_lon), vmin, vmax)
            
            levels = np.linspace(vmin, vmax, 21)
            ax.contourf(lon_grid, lat_grid, r2_grid, levels=levels, cmap=cmap, extend='both')
            title = 'TwoStage'
        
        ax.set_xlim(lon_range)
        ax.set_ylim(lat_range)
        ax.set_xlabel('Longitude')
        ax.set_title(title)
        ax.grid(True, alpha=0.2)
        
        if i == 0:
            ax.set_ylabel('Latitude')
        else:
            ax.set_yticklabels([])
    
    norm = plt.Normalize(vmin, vmax)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    plt.colorbar(sm, cax=cax, label='R²')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    if show_plot:
        plt.show()
    
    return fig, axes
