import rasterio
import numpy as np
import cv2
import fiona
import os
import geopandas as gpd
import xml.etree.ElementTree as ET
from rasterio.windows import from_bounds
from rasterio.transform import xy
from shapely.geometry import Polygon, mapping, shape, box
from pyproj import Transformer
from rasterio.warp import transform_geom
from rasterio.features import rasterize
from rasterio.windows import transform as window_transform

def thermal_conversion(array, xml_path):
        
    tree = ET.parse(xml_path)
    root = tree.getroot()
    
    radiometric = root.find('LEVEL2_SURFACE_TEMPERATURE_PARAMETERS')
    ML = float(radiometric.find("TEMPERATURE_MULT_BAND_ST_B10").text)
    AL = float(radiometric.find("TEMPERATURE_ADD_BAND_ST_B10").text)
    
    Thermal = (array * ML) + AL
    

    return Thermal

def shapefile_mask(
    shapefile,
    tif_path,
    *,
    all_touched=True,
    coast_buffer_px=0.5,
    use_true_footprint=True,
    do_vector_clip=True
):
    """
    Build a boolean mask (H, W) where True = inside shapefile geometries.

    Speed strategy:
      - Transform the raster footprint to the vector CRS
      - Read only features that intersect that bbox (gpd.read_file(..., bbox=...))
      - (Optional) clip those features by the footprint polygon in vector CRS
      - Reproject the small subset to raster CRS and rasterize

    Parameters
    ----------
    shapefile : str
    tif_path  : str
    all_touched : bool         # include edge-touching pixels
    coast_buffer_px : float    # buffer in *pixels* to catch coastlines (0 to disable)
    use_true_footprint : bool  # use 4-corner footprint (handles rotation)
    do_vector_clip : bool      # perform a true vector clip before rasterizing

    Returns
    -------
    mask : np.ndarray of bool, shape (height, width)
    """
    
    def _raster_footprint_poly(src, use_true_footprint=True):
        """
        Return raster footprint polygon in raster CRS.
        If use_true_footprint=True, handles rotation/skew via corner coords.
        Otherwise uses axis-aligned bounds.
        """
        if not use_true_footprint:
            b = src.bounds
            return Polygon([(b.left, b.top), (b.right, b.top), (b.right, b.bottom), (b.left, b.bottom)])
        h, w = src.height, src.width
        corners = [(0, 0), (0, w), (h, w), (h, 0)]
        xs, ys = zip(*[xy(src.transform, r, c, offset="ul") for r, c in corners])
        return Polygon(zip(xs, ys))
    
    with rasterio.open(tif_path) as src:
        r_crs = src.crs
        transform = src.transform
        out_shape = (src.height, src.width)
        px = max(abs(transform.a), abs(transform.e))  # pixel size in map units
        r_foot = _raster_footprint_poly(src, use_true_footprint=use_true_footprint)

    # Get vector CRS quickly without loading all features
    with fiona.open(shapefile) as lyr:
        v_crs = lyr.crs_wkt or lyr.crs
    if not v_crs:
        raise ValueError("Shapefile has no CRS; set it before masking.")

    # Transform raster footprint into vector CRS
    to_vec = Transformer.from_crs(r_crs, v_crs, always_xy=True).transform
    xs, ys = r_foot.exterior.xy
    xs_v, ys_v = to_vec(xs, ys)
    r_foot_vec = Polygon(zip(xs_v, ys_v))
    xmin, ymin, xmax, ymax = r_foot_vec.bounds

    # Load only intersecting features from disk
    gdf = gpd.read_file(shapefile, bbox=(xmin, ymin, xmax, ymax))

    if gdf.empty:
        return np.zeros(out_shape, dtype=bool)

    # Optional precise clip in vector CRS (still much cheaper than global)
    if do_vector_clip:
        # buffer(0) as a harmless fix for minor topology issues
        r_foot_vec_clean = r_foot_vec.buffer(0)
        gdf = gpd.clip(gdf, r_foot_vec_clean)
        if gdf.empty:
            return np.zeros(out_shape, dtype=bool)

    # Reproject *only* this subset to raster CRS (robust via rasterio.warp)
    geoms_ras = []
    for g in gdf.geometry:
        if g is None or g.is_empty:
            continue
        try:
            gj = mapping(g)  # GeoJSON-like
            tg = transform_geom(gdf.crs.to_string(), r_crs.to_string(), gj, precision=6)
            sg = shape(tg)
            if sg.is_empty:
                continue
            # Small positive buffer to catch coastline/edge pixels (½ pixel default)
            if coast_buffer_px and coast_buffer_px > 0:
                sg = sg.buffer(px * coast_buffer_px)
            geoms_ras.append(sg)
        except Exception:
            # skip any geometry that fails to transform
            continue

    if not geoms_ras:
        return np.zeros(out_shape, dtype=bool)

    # Rasterize to boolean mask (True = inside)
    mask = rasterize(
        [(geom, 1) for geom in geoms_ras],
        out_shape=out_shape,
        transform=transform,
        fill=0,
        dtype="uint8",
        all_touched=all_touched,
    ).astype(bool)

    return mask

def decode_landsat_qa_basic(qa):
    """Minimal QA decode: uses only bits 0-7."""
    
    with rasterio.open(qa) as src:
        qa_array = src.read()
    qa_array = qa_array.squeeze()
    
    qa = (qa_array & 255).astype(np.uint16)

    def bit(a, n):
        return (a & (1 << n)) != 0

    fill          = bit(qa, 0)
    dilated_cloud = bit(qa, 1)
    cirrus        = bit(qa, 2)
    cloud         = bit(qa, 3)
    shadow        = bit(qa, 4)
    snow          = bit(qa, 5)
    clear         = bit(qa, 6)
    water         = bit(qa, 7)

    cloud_any = cloud | cirrus | dilated_cloud

    return {
        "fill": fill,
        "cloud_any": cloud_any,
        "shadow": shadow,
        "snow": snow,
        "clear": clear,
        "water": water,
    }


def temperature_aware_cloud_mask_local(
        T,
        qa_path,
        hot_thresh_K: float = 0.3,
        block_size: int = 384,
        min_valid_frac: float = 0.1
):
    """
    Cloud mask adjusted with LOCAL sea temperature, to avoid
    misclassifying ships in colder/warmer subregions.

    Logic:
      - Start from QA cloud mask (cloud_any).
      - Estimate local sea mean per block from clear-water pixels.
      - A QA-cloud pixel is *unmasked* (not cloud) if:
            T(pixel) >= local_sea_mean(block) + hot_thresh_K

    Parameters
    ----------
    T : 2D array-like (float)
        Surface temperature or BT in Kelvin.
    qa : 2D array-like (uint16)
        QA_PIXEL band.
    hot_thresh_K : float
        Threshold above *local* sea mean (K) to treat cloud-flagged pixel as NOT cloud.
    block_size : int
        Size of grid blocks in pixels (e.g. 64, 128, 256).
    min_valid_frac : float in [0,1]
        Minimum fraction of valid sea pixels in a block to trust its local mean.
        Otherwise falls back to water-only, then to global sea mean.

    Returns
    -------
    cloud_mask_adj : 2D bool
        True = treat as cloud, False = not cloud.
    sea_mean_local : 2D float32
        Map of local sea means used (same shape as T).
    """

    qa_dict = decode_landsat_qa_basic(qa_path)

    # --- base masks ---
    finite = np.isfinite(T)
    sea = qa_dict['water'] & ~qa_dict['fill'] & finite               # all water pixels
    sea_clear = sea & ~qa_dict['cloud_any']

    H, W = T.shape

    # --- global fallback mean ---
    if sea_clear.sum() > 0:
        sea_vals_global = T[sea_clear]
    else:
        sea_vals_global = T[sea]
    if sea_vals_global.size == 0:
        # no usable sea: return raw QA cloud mask
        return qa_dict['cloud_any'].copy(), np.full_like(T, np.nan, dtype=np.float32)

    sea_mean_global = float(np.nanmean(sea_vals_global))

    # --- allocate local mean map ---
    sea_mean_local = np.full_like(T, np.nan, dtype=np.float32)

    # --- loop over grid blocks ---
    for y0 in range(0, H, block_size):
        y1 = min(y0 + block_size, H)
        for x0 in range(0, W, block_size):
            x1 = min(x0 + block_size, W)

            sl = np.s_[y0:y1, x0:x1]

            sea_clear_blk = sea_clear[sl]
            sea_blk = sea[sl]
            T_blk = T[sl]

            n_pix = (y1 - y0) * (x1 - x0)

            # 1) prefer clear sea (no cloud_any/shadow/snow)
            m1 = sea_clear_blk
            if m1.sum() >= min_valid_frac * n_pix:
                mean_blk = float(np.nanmean(T_blk[m1]))
            else:
                # 2) fallback: all sea pixels
                m2 = sea_blk
                if m2.sum() >= min_valid_frac * n_pix:
                    mean_blk = float(np.nanmean(T_blk[m2]))
                else:
                    # 3) fallback: global sea mean
                    mean_blk = sea_mean_global

            sea_mean_local[sl] = mean_blk

    # fill any leftover NaNs (paranoia) with global mean
    nan_mask = ~np.isfinite(sea_mean_local)
    if nan_mask.any():
        sea_mean_local[nan_mask] = sea_mean_global

    # --- start from QA cloud mask ---
    cloud_mask = qa_dict['cloud_any'].copy()

    # --- unmask 'too hot to be cloud' pixels using LOCAL mean ---
    too_hot_for_cloud = cloud_mask & (
        (T - sea_mean_local) >= float(hot_thresh_K)
    )

    cloud_mask_adj = cloud_mask & ~too_hot_for_cloud

    return cloud_mask_adj, sea_mean_local.astype(np.float32)

def std_func(arr, mask, ksize=3, *, min_valid_frac=0.0, fill_value=0.0,
        border_type=cv2.BORDER_REPLICATE):
    """
    Local std-dev (brightness texture) with double-precision math (float64).

    Parameters
    ----------
    arr : 2D array-like
        Input image (e.g., brightness temperature).
    mask : 2D bool array
        True = invalid pixel to ignore.
    ksize : int
        Window size (odd), e.g. 3, 5, 7.
    min_valid_frac : float in [0,1]
        Minimum fraction of valid pixels required to compute a value.
    fill_value : float
        Value to put where coverage is insufficient or invalid.
    border_type : cv2 border mode
        e.g., cv2.BORDER_REPLICATE (default).

    Returns
    -------
    sigma : 2D np.ndarray (float32)
        Local standard deviation image with no NaNs.
    """
    # --- Convert to double precision for numerical safety ---
    arr = np.asarray(arr, dtype=np.float64)
    mask = np.asarray(mask, dtype=bool)

    finite = np.isfinite(arr)
    valid = (~mask) & finite

    k = (ksize, ksize)
    win_area = float(ksize * ksize)

    # Zero-out invalids for convolution
    arr_filled = np.where(valid, arr, 0.0).astype(np.float64)
    valid_f = valid.astype(np.float64)

    # Box filter (sums, not averages) in float64
    sum_x  = cv2.boxFilter(arr_filled, ddepth=cv2.CV_64F, ksize=k,
                           normalize=False, borderType=border_type)
    sum_x2 = cv2.boxFilter(arr_filled * arr_filled, ddepth=cv2.CV_64F, ksize=k,
                           normalize=False, borderType=border_type)
    cnt    = cv2.boxFilter(valid_f, ddepth=cv2.CV_64F, ksize=k,
                           normalize=False, borderType=border_type)

    # Minimum coverage mask
    min_valid = min_valid_frac * win_area
    ok = cnt >= min_valid

    # Safe division
    mean  = np.divide(sum_x,  cnt, out=np.full_like(sum_x, fill_value),
                      where=ok & (cnt > 0))
    mean2 = np.divide(sum_x2, cnt, out=np.full_like(sum_x2, fill_value),
                      where=ok & (cnt > 0))

    # Variance = E[x²] - (E[x])², clipped to non-negative
    var = np.maximum(mean2 - mean * mean, 0.0)

    # Standard deviation
    sigma = np.sqrt(var, dtype=np.float64)

    # Fill invalids
    sigma[~ok] = fill_value

    # Final cleanup: replace any accidental non-finite values
    bad = ~np.isfinite(sigma)
    if bad.any():
        sigma[bad] = fill_value

    # --- Return as float32 to save memory ---
    return sigma.astype(np.float32)

def gradient_magnitude(arr, border_type=cv2.BORDER_REPLICATE):
    """
    Compute gradient magnitude |∇T| from a 2D thermal array.

    Parameters
    ----------
    arr : 2D np.ndarray
        Brightness or surface temperature (float32 or float64).
    border_type : cv2 border mode
        e.g., cv2.BORDER_REPLICATE to avoid artifacts at edges.

    Returns
    -------
    grad_mag : 2D np.ndarray (float32)
        Gradient magnitude image (same size as input).
    """
    arr = np.asarray(arr, dtype=np.float32)
    
    # Sobel operator approximates ∂T/∂x and ∂T/∂y
    gx = cv2.Sobel(arr, cv2.CV_32F, 1, 0, ksize=3, borderType=border_type)
    gy = cv2.Sobel(arr, cv2.CV_32F, 0, 1, ksize=3, borderType=border_type)
    
    grad_mag = cv2.magnitude(gx, gy)   # sqrt(gx² + gy²)
    return grad_mag

def clip_raster_by_rect_buffer(
    shp_path: str,
    tif_path: str,
    directory_path: str,
    n_pixels: int = 50,
    attribute_for_name: str | None = None,  # e.g., "id" or "name"; falls back to the row index
    overwrite: bool = False
):
    """
    For each feature in `shp_path`, create an axis-aligned rectangle expanded by ~`n_pixels`
    on all sides, then clip `tif_path` to that rectangle (grid-aligned window read), and write
    one GeoTIFF per feature into `out_dir`.

    Parameters
    ----------
    shp_path : str
        Path to the input shapefile.
    tif_path : str
        Path to the input GeoTIFF.
    out_dir : str
        Output directory where per-feature clips will be written.
    n_pixels : int
        Number of pixels to expand in all directions to form the rectangle buffer.
    attribute_for_name : str or None
        Optional attribute in the shapefile used to name outputs. If None, uses row index.
    overwrite : bool
        Whether to overwrite existing files.
    """

    clip_name = os.path.splitext(os.path.basename(tif_path))[0]
    os.makedirs(f"{directory_path}/Clipped_T/{clip_name}", exist_ok=True)

    # Load raster once to fetch transform, resolution, CRS, and profile
    with rasterio.open(tif_path) as src:
        r_crs = src.crs
        transform = src.transform
        # Pixel size. Note transform.a is pixel width, transform.e is negative pixel height in most north-up rasters
        px_w = abs(transform.a)
        px_h = abs(transform.e)

        # Expand distances in map units corresponding to n_pixels
        dx = n_pixels * px_w
        dy = n_pixels * px_h

        # Load vector and reproject if necessary
        gdf = gpd.read_file(shp_path)
        if gdf.crs is None:
            raise ValueError("Shapefile has no CRS. Please define or set it before running.")

        if r_crs is not None and gdf.crs != r_crs:
            gdf = gdf.to_crs(r_crs)

        # Iterate features
        clip_paths = []
        for idx, row in gdf.iterrows():
            geom = row.geometry
            if geom is None or geom.is_empty:
                continue

            # Build rectangle buffer around the feature bounds
            minx, miny, maxx, maxy = geom.bounds
            rect = box(minx - dx, miny - dy, maxx + dx, maxy + dy)

            # Convert rectangle bounds to a grid-aligned window
            w = from_bounds(*rect.bounds, transform=transform, width=src.width, height=src.height)
            # Clip window to the raster extent
            w = w.intersection(rasterio.windows.Window(col_off=0, row_off=0, width=src.width, height=src.height))

            if w.width <= 0 or w.height <= 0:
                # Rectangle lies completely outside the raster
                continue

            # Read only the window to avoid loading the whole raster
            data = src.read(window=w)  # shape: (bands, height, width)
            out_transform = window_transform(w, transform)

            # Prepare output profile
            profile = src.profile.copy()
            profile.update(
                height=int(w.height),
                width=int(w.width),
                transform=out_transform,
                compress="lzw",
                BIGTIFF="IF_SAFER"
            )

            # Write the clipped raster
            with rasterio.open(f"{directory_path}/Clipped_T/{clip_name}/{clip_name}_{idx}.tif", "w", **profile) as dst:
                dst.write(data)
                
            clip_paths.append(f"{directory_path}/Clipped_T/{clip_name}/{clip_name}_{idx}.tif")
            
        return clip_paths
    
def relabel_tiles_with_all_ships_from_paths(
    full_tif_path: str,
    ship_shapefile_path: str,
    tile_paths: list[str],
    *,
    all_touched: bool = True
) -> list[np.ndarray]:
    """
    Rasterize all ship polygons onto each tile in tile_paths.

    Parameters
    ----------
    full_tif_path : str
        Path to the full raster scene (used only to match CRS).
    ship_shapefile_path : str
        Path to shapefile containing all ships for the scene.
    tile_paths : list[str]
        List of paths to the tile GeoTIFFs.
    all_touched : bool, optional
        Include pixels touched by the polygon edges (default: True).

    Returns
    -------
    masks : list[np.ndarray]
        A list of binary (H, W) ship masks (0 = water, 1 = ship),
        aligned to each tile in tile_paths.
    """
    # Read ships
    ship_gdf = gpd.read_file(ship_shapefile_path)

    # Read full image CRS to reproject if needed
    with rasterio.open(full_tif_path) as src_full:
        full_crs = src_full.crs

    # Reproject ship polygons to match raster CRS
    if ship_gdf.crs != full_crs:
        ship_gdf = ship_gdf.to_crs(full_crs)

    masks = []
    arrays = []

    for tpath in tile_paths:
        with rasterio.open(tpath) as tile_src:
            H, W = tile_src.height, tile_src.width
            T = tile_src.transform
            tile_crs = tile_src.crs
            tile_bounds = tile_src.bounds
            data = tile_src.read()
            
        arrays.append(data)

        # Create shapely bounding box for the tile
        tile_box = box(*tile_bounds)

        # Filter ships that intersect the tile bounds
        ship_subset = ship_gdf[ship_gdf.intersects(tile_box)]

        if ship_subset.empty:
            masks.append(np.zeros((H, W), dtype="uint8"))
            continue

        # Rasterize all ships in this tile
        geoms = [(geom, 1) for geom in ship_subset.geometry if not geom.is_empty]
        mask = rasterize(
            geoms,
            out_shape=(H, W),
            transform=T,
            fill=0,
            all_touched=all_touched,
            dtype="uint8"
        )
        masks.append(mask)

    return masks, arrays

def clip_raster_by_rect_buffer(
    shp_path: str,
    tif_path: str,
    directory_path: str,
    n_pixels: int = 50,
    attribute_for_name: str | None = None,  # e.g., "id" or "name"; falls back to the row index
    overwrite: bool = False
):
    """
    For each feature in `shp_path`, create an axis-aligned rectangle expanded by ~`n_pixels`
    on all sides, then clip `tif_path` to that rectangle (grid-aligned window read), and write
    one GeoTIFF per feature into `out_dir`.

    Parameters
    ----------
    shp_path : str
        Path to the input shapefile.
    tif_path : str
        Path to the input GeoTIFF.
    out_dir : str
        Output directory where per-feature clips will be written.
    n_pixels : int
        Number of pixels to expand in all directions to form the rectangle buffer.
    attribute_for_name : str or None
        Optional attribute in the shapefile used to name outputs. If None, uses row index.
    overwrite : bool
        Whether to overwrite existing files.
    """

    clip_name = os.path.splitext(os.path.basename(tif_path))[0]
    os.makedirs(f"{directory_path}/Clipped_T/{clip_name}", exist_ok=True)

    # Load raster once to fetch transform, resolution, CRS, and profile
    with rasterio.open(tif_path) as src:
        r_crs = src.crs
        transform = src.transform
        # Pixel size. Note transform.a is pixel width, transform.e is negative pixel height in most north-up rasters
        px_w = abs(transform.a)
        px_h = abs(transform.e)

        # Expand distances in map units corresponding to n_pixels
        dx = n_pixels * px_w
        dy = n_pixels * px_h

        # Load vector and reproject if necessary
        gdf = gpd.read_file(shp_path)
        if gdf.crs is None:
            raise ValueError("Shapefile has no CRS. Please define or set it before running.")

        if r_crs is not None and gdf.crs != r_crs:
            gdf = gdf.to_crs(r_crs)

        # Iterate features
        clip_paths = []
        for idx, row in gdf.iterrows():
            geom = row.geometry
            if geom is None or geom.is_empty:
                continue

            # Build rectangle buffer around the feature bounds
            minx, miny, maxx, maxy = geom.bounds
            rect = box(minx - dx, miny - dy, maxx + dx, maxy + dy)

            # Convert rectangle bounds to a grid-aligned window
            w = from_bounds(*rect.bounds, transform=transform, width=src.width, height=src.height)
            # Clip window to the raster extent
            w = w.intersection(rasterio.windows.Window(col_off=0, row_off=0, width=src.width, height=src.height))

            if w.width <= 0 or w.height <= 0:
                # Rectangle lies completely outside the raster
                continue

            # Read only the window to avoid loading the whole raster
            data = src.read(window=w)  # shape: (bands, height, width)
            out_transform = window_transform(w, transform)

            # Prepare output profile
            profile = src.profile.copy()
            profile.update(
                height=int(w.height),
                width=int(w.width),
                transform=out_transform,
                compress="lzw",
                BIGTIFF="IF_SAFER"
            )

            # Write the clipped raster
            with rasterio.open(f"{directory_path}/Clipped_T/{clip_name}/{clip_name}_{idx}.tif", "w", **profile) as dst:
                dst.write(data)
                
            clip_paths.append(f"{directory_path}/Clipped_T/{clip_name}/{clip_name}_{idx}.tif")
            
        return clip_paths