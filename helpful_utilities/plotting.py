import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm
import cartopy.crs as ccrs


def easy_map(da, n_colors=10, cmap=None, levels=None, figsize=(8, 4), switch_to_180=False, **kwargs):
    """
    Plot a DataArray (lat, lon) on a map with discrete colors,
    centered on the data, no extra whitespace, and labeled gridlines.
    """

    if not {'lat', 'lon'}.issubset(da.dims):
        raise ValueError("DataArray must have 'lat' and 'lon' dimensions.")

    # Determine data range and whether it spans zero
    vmin, vmax = float(da.min()), float(da.max())
    spans_zero = (vmin < 0) and (vmax > 0)

    # Choose colormap
    if cmap is None:
        cmap = 'RdBu_r' if spans_zero else 'viridis'

    # Symmetric or linear color levels
    if levels is None:
        if spans_zero:
            v = max(abs(vmin), abs(vmax))
            levels = np.linspace(-v, v, n_colors + 1)
            print(levels)
        else:
            levels = np.linspace(vmin, vmax, n_colors + 1)

    norm = BoundaryNorm(levels, ncolors=256, clip=False)

    if switch_to_180:
        if da['lon'].max() > 180:
            lon = ((da['lon'].values + 180) % 360) - 180
            da = da.assign_coords(lon=lon).sortby('lon')

    # Projection centered on data midpoint
    lon_center = float(da.lon.mean())
    proj = ccrs.PlateCarree(central_longitude=lon_center)

    fig, ax = plt.subplots(subplot_kw={'projection': proj}, figsize=figsize)

    # Plot
    mesh = da.plot(transform=ccrs.PlateCarree(), cmap=cmap, norm=norm, shading='auto', add_colorbar=False)

    # Add coastlines and gridlines
    ax.coastlines(linewidth=0.8)
    gl = ax.gridlines(draw_labels=True, linestyle='--', color='gray', alpha=0.5, linewidth=0.5)
    gl.top_labels = False
    gl.right_labels = False

    # Set extent exactly to data (no whitespace)
    ax.set_extent([
        float(da.lon.min()), float(da.lon.max()),
        float(da.lat.min()), float(da.lat.max())
    ], crs=ccrs.PlateCarree())

    # Colorbar: discrete, rounded ticks
    cb = plt.colorbar(
        mesh, ax=ax, orientation='horizontal',
        pad=0.1, shrink=0.9, extend='both'
    )

    if 'cbar_name' in kwargs:
        cb.set_label(kwargs['cbar_name'])

    ax.set_title(da.name or '', loc='left', fontsize=12)
    plt.tight_layout()
    return fig, ax
