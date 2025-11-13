import rasterio
import numpy as np
import xml.etree.ElementTree as ET
import os
import glob
import joblib
import tracemalloc
import time
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.inspection import permutation_importance
from sklearn.metrics import f1_score, make_scorer
from typing import Tuple, Optional
from aux_functions_L2 import thermal_conversion, shapefile_mask, std_func, gradient_magnitude, relabel_tiles_with_all_ships_from_paths, clip_raster_by_rect_buffer, temperature_aware_cloud_mask_local

def TC_images_all(directory_path, land_shapefile_path):
    
    """
    Build thermal-composite images for all Landsat scenes in a directory.

    This function expects a **very strict folder structure** and **original Landsat 
    filenames**. The filenames produced by the USGS (e.g., 
    LC08_L1TP_XXXXX_XXXXX_ST_B10.tif) **must NOT be renamed** or modified, 
    because the function extracts the `scene_name` from the MTL/XML metadata and 
    uses it to locate the matching Band 10 and QA files.

    REQUIRED DIRECTORY STRUCTURE (inside `directory_path`):

        directory_path/
        ├── B10/
        │     └── <scene_name>_ST_B10.tif
        ├── QA/
        │     └── <scene_name>_QA_PIXEL.tif
        ├── xml/
        │     └── <scene_name>.xml   (MTL metadata in XML format)
        └── Thermal Composite/
              └── (output is written here)

    Where `<scene_name>` must be the original Landsat Scene ID that appears 
    inside the XML metadata under:
        PRODUCT_CONTENTS → LANDSAT_PRODUCT_ID

    EXAMPLE of expected naming:
        LC08_L2SP_190030_20230829_20230906_02_T1_ST_B10.tif
        LC08_L2SP_190030_20230829_20230906_02_T1_QA_PIXEL.tif
        LC08_L2SP_190030_20230829_20230906_02_T1.xml

    PARAMETERS
    ----------
    directory_path : str
        Path to the parent folder containing the subfolders: B10/, QA/, xml/, 
        and Thermal Composite/.
    
    land_shapefile_path : str
        Path to the land polygon shapefile used to mask out land pixels.

    NOTES
    -----
    • If the output file `<scene_name>_thermal_composite.tif` already exists, 
      the scene is skipped.
    • Generates a 3-band thermal composite containing:
          Band 1: Mean-centered thermal values
          Band 2: Local standard deviation
          Band 3: Gradient magnitude
    """
    
    xmlL = glob.glob(os.path.join(f"{directory_path}/xml", '*.xml'))
    compL = glob.glob(os.path.join(f"{directory_path}/Thermal Composite", '*.tif'))
    comp_names = [os.path.splitext(os.path.basename(f))[0] for f in compL]
     
    for xml_path in xmlL:
        tree = ET.parse(xml_path)
        root = tree.getroot()
        
        processing_level = root.find('PRODUCT_CONTENTS')
        scene_name = processing_level.find('LANDSAT_PRODUCT_ID').text
        
        comp_n = f"{scene_name}_thermal_composite"
        
        if comp_n in comp_names:
            print(f"Skipping {comp_n}, already processed.")
            continue
                
        B10 = f"{directory_path}/B10/{scene_name}_ST_B10.tif"
        qa = f"{directory_path}/QA/{scene_name}_QA_PIXEL.tif"
        
        with rasterio.open(B10) as src:
            arr = src.read()
            arr = arr.squeeze()
            meta = src.meta.copy()
        mask_10 = arr == 0
        poly_mask = shapefile_mask(land_shapefile_path, B10)
        arr = thermal_conversion(arr, xml_path)
        arr = arr.astype(np.float32)
        cloud_mask, water_mean = temperature_aware_cloud_mask_local(arr, qa)
        combined_mask = mask_10 | poly_mask
        combined_mask = combined_mask | cloud_mask
        arr[combined_mask] = np.nan
        m_arr = arr - water_mean
        std_B10 = std_func(arr, combined_mask)
        grad_B10 = gradient_magnitude(arr)

        meta.update({
        "dtype": "float32",
        "count": 3
        })
        with rasterio.open(f"{directory_path}/Thermal Composite/{scene_name}_thermal_composite.tif", "w", **meta) as dst:
            dst.write(m_arr, 1)  
            dst.write(std_B10, 2)  
            dst.write(grad_B10, 3)
    
            
def create_tiles_update_dict(directory_path):
    
    """
    Build a full scene dictionary containing all required file paths, clipped tile paths,
    tile pixel arrays, and relabeled ship masks for every Landsat scene in the dataset.
    
    IMPORTANT
    ---------
    This function requires a **strict folder structure** and **original Landsat filenames**.
    The Landsat scene ID extracted from the XML metadata (field:
        PRODUCT_CONTENTS → LANDSAT_PRODUCT_ID
    )
    is used to construct all expected file paths. Therefore:
    
    **The Landsat filenames MUST NOT be renamed or altered.**
    
    REQUIRED DIRECTORY STRUCTURE (inside `directory_path`)
    ------------------------------------------------------
    
        directory_path/
        ├── B10/
        │     └── <scene_name>_B10.tif
        ├── Shapefiles/
        │     └── <scene_name>_BB.shp         (bounding box or ship polygons)
        ├── xml/
        │     └── <scene_name>_MTL.xml        (MTL metadata file)
        ├── Thermal Composite/
        │     └── <scene_name>_thermal_composite.tif
        └── (output tiles will be stored automatically)
    
    Where `<scene_name>` is the official Landsat product ID fetched from the XML file.
    
    FUNCTION BEHAVIOR
    -----------------
    For each scene detected in `directory_path/xml/`:
    
    1. Parse the XML to obtain the Landsat scene name.
    2. Build a dictionary containing all required file paths:
           - Band 10 path
           - Shapefile path
           - XML path
           - Thermal composite image path
    3. Automatically generate clipped tiles by calling:
           clip_raster_by_rect_buffer()
       Resulting tile paths are added under:
           all_dict[scene_name]["tile_paths"]
    4. Relabel each tile with ship masks using:
           relabel_tiles_with_all_ships_from_paths()
       Results are stored under:
           all_dict[scene_name]["image_tiles_arr"]
           all_dict[scene_name]["ship_feat"]
    
    RETURNS
    -------
    all_dict : dict
        A nested dictionary containing, for each scene:
    
            all_dict[scene_name] = {
                "B10"           : path to Band 10 image,
                "Shapefiles"    : path to scene ship shapefile,
                "xml"           : path to MTL XML,
                "comp_image"    : path to 3-band thermal composite,
                "tile_paths"    : list of clipped tile GeoTIFF paths,
                "image_tiles_arr": list/array of tile pixel arrays,
                "ship_feat"     : list/array of ship masks per tile
            }
    
    NOTES
    -----
    • The function processes all scenes found in the xml/ folder.
    • Filenames must follow the Landsat standard naming convention.
    • Output is structured for immediate use in model training workflows.
    """

    xmlL = glob.glob(os.path.join(f"{directory_path}/xml", '*.xml'))
    
    all_dict = {}
    for xml_path in xmlL:
        tree = ET.parse(xml_path)
        root = tree.getroot()
        
        processing_level = root.find('PRODUCT_CONTENTS')
        scene_name = processing_level.find('LANDSAT_PRODUCT_ID').text
             
        scene_partsD = {"B10": f"{directory_path}/B10/{scene_name}_B10.tif",
                        "Shapefiles": f"{directory_path}/Shapefiles/{scene_name}_BB.shp",
                        "xml": f"{directory_path}/xml/{scene_name}_MTL.xml",
                        "comp_image": f"{directory_path}/Thermal Composite/{scene_name}_thermal_composite.tif"
                        }
        
        all_dict[f"{scene_name}"] = scene_partsD
    
    for scene_name in all_dict.keys():
        clip_pathsL = clip_raster_by_rect_buffer(all_dict[scene_name]["Shapefiles"], all_dict[scene_name]["comp_image"], directory_path)
        all_dict[scene_name]["tile_paths"] = clip_pathsL
        
    for scene_name in all_dict.keys():      
        masksL, tilearrL = relabel_tiles_with_all_ships_from_paths(all_dict[scene_name]["comp_image"], all_dict[scene_name]["Shapefiles"], all_dict[scene_name]["tile_paths"])
        
        all_dict[scene_name]["image_tiles_arr"] = tilearrL
        all_dict[scene_name]["ship_feat"] = masksL
    
    return all_dict

def train_rf_from_tiled_scenes2(
    all_dict: dict,
    *,
    tile_key: str = "image_tiles_arr",
    mask_key: str = "ship_feat",
    channels_first: bool = True,
    label_mode: str = "mask",
    ship_radius_px: int = 5,
    bg_per_ship_ratio: int = 5,
    max_ship_pixels_per_tile: int | None = 2000,
    random_state: int = 42,
    test_size: float = 0.25,
    rf_kwargs: dict | None = None,
    feature_names: list[str] | None = None,     # <-- NEW
    perm_n_repeats: int = 5,                    # <-- NEW (speed/variance tradeoff)
    perm_subsample: int | None = 100_000        # <-- NEW (speed: None = use all test pixels)
):

    """
    Train a Random Forest ship detector from tiled scenes, using SCENE-BASED splitting.

    This function expects `all_dict` to be the output of the tiling/labeling pipeline
    (e.g. from `create_tiles_update_dict`), where each scene entry contains:
        - tile feature arrays (thermal composite tiles)
        - corresponding ship masks
        - metadata about which tiles belong to which scene

    SCENE-BASED SPLIT
    -----------------
    Rather than mixing pixels from all scenes, the function splits data at the
    scene level:
        - Each scene is assigned entirely to either the training set or the test set.
        - `train_test_split` is applied to the list of scene names.
    This prevents overly optimistic metrics that can occur when near-duplicate
    tiles from the same scene appear in both train and test sets.

    EXPECTED STRUCTURE OF `all_dict`
    --------------------------------
    all_dict is a nested dictionary of the form:

        all_dict[scene_name] = {
            tile_key   : list/array of tiles, each with shape (C, H, W) if channels_first=True,
                          or (H, W, C) if channels_first=False,
            mask_key   : list/array of ship masks, one per tile, shape (H, W),
            ...        : other metadata added upstream (paths, etc.)
        }

    PARAMETERS
    ----------
    all_dict : dict
        Dictionary of scenes, tiles, and masks as described above.

    tile_key : str, optional
        Key in `all_dict[scene_name]` that stores the tile feature arrays.
        Default is "image_tiles_arr".

    mask_key : str, optional
        Key in `all_dict[scene_name]` that stores the ship masks for each tile.
        Default is "ship_feat".

    channels_first : bool, optional
        If True, tiles are assumed to be in (C, H, W) format and are converted to
        (H, W, C). If False, tiles are assumed to already be (H, W, C).
        Default is True.

    label_mode : {"mask", "center_radius"}, optional
        How to construct the ship labels for each tile:
            - "mask": use the provided per-pixel ship mask (H, W) from `mask_key`.
            - "center_radius": ignore provided masks and instead define a ship
              region as a filled circle centered on the tile center with radius
              `ship_radius_px`.
        Default is "mask".

    ship_radius_px : int, optional
        Radius in pixels for the synthetic ship region when `label_mode="center_radius"`.
        Ignored if `label_mode="mask"`. Default is 5.

    bg_per_ship_ratio : int, optional
        For each ship pixel, this many background pixels (water) are randomly sampled.
        Controls the class balance during training. Default is 5 (1:5 ship:background).

    max_ship_pixels_per_tile : int or None, optional
        Maximum number of ship pixels sampled per tile. If a tile contains more than
        this number of ship pixels, a random subset of size `max_ship_pixels_per_tile`
        is used. If None, all ship pixels are used. Default is 2000.

    random_state : int, optional
        Seed used for all random operations (scene splitting, sampling).
        Default is 42.

    test_size : float, optional
        Fraction of scenes to hold out for testing. Passed to `train_test_split`.
        Default is 0.25 (25% of scenes in test set).

    rf_kwargs : dict or None, optional
        Optional dictionary of keyword arguments passed to `RandomForestClassifier`.
        If None, the following defaults are used:
            n_estimators   = 300
            max_depth      = None
            min_samples_leaf = 2
            class_weight   = "balanced_subsample"
            n_jobs         = -1
            random_state   = random_state

    feature_names : list of str or None, optional
        Names of the input features used for feature importance reporting.
        Length must match the number of channels C in the tile arrays.
        If None, a simple default list is used (e.g., ["BT10", "σBT10", "grad"]).
        This only affects the readability of the importance tables, not training.

    perm_n_repeats : int, optional
        Number of repeats for permutation importance. Higher values give more
        stable estimates but increase runtime. Default is 5.

    perm_subsample : int or None, optional
        Maximum number of test pixels used for permutation importance. If the
        test set contains more than `perm_subsample` pixels, a random subset
        of this size is used for speed. If None, all test pixels are used.
        Default is 100000.

    PROCESSING STEPS
    ----------------
    For each scene:
        1. Loop over all tiles belonging to the scene.
        2. Ensure consistent channel order and dimensionality (H, W, C).
        3. Build a per-pixel ship mask (from `mask_key` or from `center_radius`).
        4. Flatten tile into (N_pixels, C) and mask into (N_pixels,).
        5. Drop any pixels with non-finite feature values.
        6. Randomly sample ship and background pixels to enforce:
               - up to `max_ship_pixels_per_tile` ship pixels per tile
               - a bg:ship ratio of `bg_per_ship_ratio`.
        7. Aggregate all sampled pixels and labels per scene.
    After all scenes:
        - Concatenate per-scene arrays into X_train, y_train, X_test, y_test.
        - Train a RandomForestClassifier on the training set.
        - Evaluate on the test set and compute metrics.

    METRICS AND FEATURE IMPORTANCE
    ------------------------------
    The following evaluation outputs are computed:
        - Confusion matrix on the test set.
        - Classification report (precision, recall, F1) for water and ship.
        - Class counts in train/test splits.

    Feature importance is reported in two ways:
        1. Gini importance (`rf.feature_importances_`), quick but can be biased.
        2. Permutation importance based on drop in F1-score for the ship class
           when permuting each feature on the test set (or subsample).

    MODEL SAVING
    ------------
    The trained RandomForestClassifier is saved to disk using joblib:
        "rf_ship_detector.pkl"
    in the current working directory.

    RETURNS
    -------
    rf : RandomForestClassifier
        The trained Random Forest ship detector.

    metrics : dict
        Dictionary containing:
            - "confusion_matrix"
            - "classification_report"
            - "positives_in_train", "negatives_in_train"
            - "positives_in_test",  "negatives_in_test"
            - "feature_importance_gini"
            - "feature_importance_permutation_f1_ship"

    info : dict
        Additional metadata about the training run:
            - "train_scenes", "test_scenes"
            - "n_train_scenes", "n_test_scenes"
            - "n_samples_train", "n_samples_test"
            - "tiles_used_train", "tiles_used_test"
            - "feature_dim"
            - "label_mode"
            - "channels_first"
            - "perm_n_repeats"
            - "perm_subsample"
    """

    if rf_kwargs is None:
        rf_kwargs = dict(
            n_estimators=300,
            max_depth=None,
            min_samples_leaf=2,
            class_weight="balanced_subsample",
            n_jobs=-1,
            random_state=random_state,
        )

    rng = np.random.default_rng(random_state)
    feature_dim = None
    scene_names = list(all_dict.keys())

    # --- Split scenes (not pixels)
    train_scenes, test_scenes = train_test_split(
        scene_names, test_size=test_size, random_state=random_state
    )

    X_train_list, y_train_list = [], []
    X_test_list, y_test_list = [], []
    tiles_used_train = 0
    tiles_used_test = 0

    # --- Process scenes
    for scene_name, scene_dict in all_dict.items():
        tiles = scene_dict.get(tile_key, [])
        masks = scene_dict.get(mask_key, None)
        if not tiles:
            continue

        if label_mode == "mask" and (masks is None or len(masks) != len(tiles)):
            raise ValueError(
                f"{scene_name}: missing or misaligned '{mask_key}'. "
                f"Need one (H,W) mask per tile."
            )

        # Aggregate per-scene data
        X_scene, y_scene = [], []

        for i, tile in enumerate(tiles):
            arr = np.asarray(tile)
            if arr.ndim != 3:
                raise ValueError(f"{scene_name}: tile {i} must be 3D; got {arr.shape}")

            # Reorder channels if needed
            if channels_first:
                C, H, W = arr.shape
                arr = np.moveaxis(arr, 0, -1)
            else:
                H, W, C = arr.shape

            if feature_dim is None:
                feature_dim = C
            elif feature_dim != C:
                raise ValueError(f"Inconsistent channel count: expected {feature_dim}, got {C}")

            # Label mask
            if label_mode == "mask":
                ship_mask = np.asarray(masks[i]).astype(np.uint8)
                if ship_mask.shape != (H, W):
                    raise ValueError(f"{scene_name}: mask {i} shape {ship_mask.shape} != tile {(H,W)}")
            elif label_mode == "center_radius":
                cy, cx = H // 2, W // 2
                ys, xs = np.ogrid[:H, :W]
                ship_mask = ((ys - cy)**2 + (xs - cx)**2) <= (ship_radius_px**2)
                ship_mask = ship_mask.astype(np.uint8)
            else:
                raise ValueError("label_mode must be 'mask' or 'center_radius'.")

            # Flatten
            X_flat = arr.reshape(-1, C)
            y_flat = ship_mask.reshape(-1)

            # Drop invalids
            valid = np.isfinite(X_flat).all(axis=1)
            X_flat = X_flat[valid]
            y_flat = y_flat[valid]

            # Sampling
            ship_idx = np.where(y_flat == 1)[0]
            bg_idx = np.where(y_flat == 0)[0]
            if ship_idx.size == 0 or bg_idx.size == 0:
                continue

            if max_ship_pixels_per_tile and ship_idx.size > max_ship_pixels_per_tile:
                ship_idx = rng.choice(ship_idx, size=max_ship_pixels_per_tile, replace=False)

            n_bg = min(bg_idx.size, bg_per_ship_ratio * ship_idx.size)
            bg_idx = rng.choice(bg_idx, size=n_bg, replace=False)

            idx = np.concatenate([ship_idx, bg_idx])
            rng.shuffle(idx)

            X_scene.append(X_flat[idx])
            y_scene.append(y_flat[idx])

        # Skip empty scenes
        if not X_scene:
            continue

        X_scene = np.concatenate(X_scene, axis=0)
        y_scene = np.concatenate(y_scene, axis=0)

        # Assign to train or test group
        if scene_name in train_scenes:
            X_train_list.append(X_scene)
            y_train_list.append(y_scene)
            tiles_used_train += len(tiles)
        else:
            X_test_list.append(X_scene)
            y_test_list.append(y_scene)
            tiles_used_test += len(tiles)

    # --- Concatenate
    X_train = np.concatenate(X_train_list, axis=0)
    y_train = np.concatenate(y_train_list, axis=0)
    X_test  = np.concatenate(X_test_list, axis=0)
    y_test  = np.concatenate(y_test_list, axis=0)

    # --- Train
    rf = RandomForestClassifier(**rf_kwargs)
    rf.fit(X_train, y_train)

    # --- Metrics (predictions)
    y_pred = rf.predict(X_test)
    cm = confusion_matrix(y_test, y_pred, labels=[0, 1])
    report = classification_report(y_test, y_pred, target_names=["water", "ship"])

    # ---------------------------
    # Feature importance metrics
    # ---------------------------
    # Default names if not provided
    if feature_names is None:
        # Your 6-channel stack order (update if different)
        feature_names = ["BT10", "σBT10", "grad"]

    # 1) Gini importance (fast; can be biased by correlated features)
    gini_vals = rf.feature_importances_.astype(float)
    gini_tbl = sorted(
        [{"feature": n, "importance": float(v)} for n, v in zip(feature_names, gini_vals)],
        key=lambda d: d["importance"],
        reverse=True
    )

    # 2) Permutation importance on test set (drop in F1 for ship class)
    # Optionally subsample X_test for speed
    if perm_subsample is not None and X_test.shape[0] > perm_subsample:
        rng = np.random.default_rng(random_state)
        idx = rng.choice(X_test.shape[0], size=perm_subsample, replace=False)
        Xp, yp = X_test[idx], y_test[idx]
    else:
        Xp, yp = X_test, y_test

    # scorer focused on ship class (label=1); change average if you prefer macro
    ship_f1_scorer = make_scorer(f1_score, pos_label=1)

    perm = permutation_importance(
        rf,
        Xp,
        yp,
        scoring=ship_f1_scorer,
        n_repeats=perm_n_repeats,
        random_state=random_state,
        n_jobs=-1
    )

    perm_tbl = sorted(
        [
            {
                "feature": feature_names[i],
                "mean_drop_f1": float(perm.importances_mean[i]),
                "std_drop_f1": float(perm.importances_std[i]),
            }
            for i in range(len(feature_names))
        ],
        key=lambda d: d["mean_drop_f1"],
        reverse=True
    )

    metrics = {
        "confusion_matrix": cm,
        "classification_report": report,
        "positives_in_train": int((y_train == 1).sum()),
        "negatives_in_train": int((y_train == 0).sum()),
        "positives_in_test": int((y_test == 1).sum()),
        "negatives_in_test": int((y_test == 0).sum()),
        # NEW:
        "feature_importance_gini": gini_tbl,
        "feature_importance_permutation_f1_ship": perm_tbl,
    }

    info = {
        "train_scenes": train_scenes,
        "test_scenes": test_scenes,
        "n_train_scenes": len(train_scenes),
        "n_test_scenes": len(test_scenes),
        "n_samples_train": int(X_train.shape[0]),
        "n_samples_test": int(X_test.shape[0]),
        "tiles_used_train": tiles_used_train,
        "tiles_used_test": tiles_used_test,
        "feature_dim": int(feature_dim),
        "label_mode": label_mode,
        "channels_first": channels_first,
        "perm_n_repeats": perm_n_repeats,
        "perm_subsample": perm_subsample,
    }
    
    joblib.dump(rf, "rf_ship_detector.pkl")
    

    return rf, metrics, info

def detect_ships_with_rf(
    comp_tif_path: str,
    model_path: str = "rf_ship_detector.pkl",
    *,
    prob_threshold: float = 0.5,
    out_ship_mask_path: Optional[str] = None,
    out_prob_path: Optional[str] = None,
    profile: bool = True,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, float]]:
    """
    Run ship detection on a thermal composite image using a pre-trained RF,
    optionally measuring runtime and memory usage.

    This function expects the same input features that were used for training:
        Band 1: mean-centered thermal (e.g., BT10 - local sea mean)
        Band 2: local standard deviation (texture)
        Band 3: gradient magnitude

    It loads the trained RF model from disk, applies it per pixel, and produces:
        - a binary ship mask (0 = water, 1 = ship)
        - a probability map (P(ship) per pixel)
        - an optional metrics dict with runtime & memory usage

    PARAMETERS
    ----------
    comp_tif_path : str
        Path to the 3-band thermal composite GeoTIFF.

    model_path : str, optional
        Path to the saved RandomForest model (joblib/pickle file).
        Default is "rf_ship_detector.pkl".

    prob_threshold : float, optional
        Probability threshold for classifying a pixel as ship.
        Pixels with P(ship) >= prob_threshold are labeled as 1 (ship),
        otherwise 0 (water). Default is 0.5.

    out_ship_mask_path : str or None, optional
        If provided, the binary ship mask will be written to this path as a
        single-band GeoTIFF (uint8). If None, no file is written.

    out_prob_path : str or None, optional
        If provided, the ship probability map will be written to this path as a
        single-band GeoTIFF (float32). If None, no file is written.

    profile : bool, optional
        If True, measure wall-clock runtime and peak Python memory usage using
        `time.perf_counter` and `tracemalloc`. Metrics are returned in a dict.
        If False, metrics are still returned but with zeros. Default is True.

    RETURNS
    -------
    ship_mask : np.ndarray
        2D uint8 array of shape (H, W), where 1 = ship, 0 = water.

    ship_prob : np.ndarray
        2D float32 array of shape (H, W), containing P(ship) for each pixel.

    metrics : dict
        Dictionary with performance and image info:
            - "runtime_sec"
            - "peak_memory_mb"
            - "width"
            - "height"
            - "n_pixels"
            - "n_bands"

    NOTES
    -----
    • Any pixel with non-finite features (NaN/Inf in any band) is treated as
      invalid and set to ship_prob = 0.0, ship_mask = 0.
    • Geo-referencing (transform, CRS) is preserved in any outputs written.
    """

    # ---- Optional profiling start ----
    if profile:
        tracemalloc.start()
        t0 = time.perf_counter()
    else:
        t0 = time.perf_counter()  # still measure elapsed time, but no memory

    # ---- Load model ----
    rf = joblib.load(model_path)

    # ---- Read composite image (C, H, W) ----
    with rasterio.open(comp_tif_path) as src:
        comp = src.read(out_dtype="float32")  # (C, H, W)
        profile_r = src.profile.copy()
        width, height, n_bands = src.width, src.height, src.count

    if comp.ndim != 3 or comp.shape[0] < 3:
        raise ValueError(f"Expected a 3-band composite (C,H,W), got shape {comp.shape}")

    # Move channels last: (H, W, C)
    comp = np.moveaxis(comp, 0, -1)  # (H, W, C)
    H, W, C = comp.shape

    # Flatten to (N, C)
    X_flat = comp.reshape(-1, C)

    # Valid pixels = all finite features
    valid = np.isfinite(X_flat).all(axis=1)

    # Prepare outputs
    ship_prob_flat = np.zeros(X_flat.shape[0], dtype=np.float32)
    ship_mask_flat = np.zeros(X_flat.shape[0], dtype=np.uint8)

    # Predict only on valid pixels
    if valid.any():
        X_valid = X_flat[valid]
        proba_valid = rf.predict_proba(X_valid)[:, 1].astype(np.float32)
        ship_prob_flat[valid] = proba_valid
        ship_mask_flat[valid] = (proba_valid >= prob_threshold).astype(np.uint8)

    # Reshape back to (H, W)
    ship_prob = ship_prob_flat.reshape(H, W)
    ship_mask = ship_mask_flat.reshape(H, W)

    # ---- Optional: write outputs ----
    if out_ship_mask_path is not None:
        mask_profile = profile_r.copy()
        mask_profile.update(dtype="uint8", count=1, nodata=0)
        with rasterio.open(out_ship_mask_path, "w", **mask_profile) as dst:
            dst.write(ship_mask, 1)

    if out_prob_path is not None:
        prob_profile = profile_r.copy()
        prob_profile.update(dtype="float32", count=1, nodata=0.0)
        with rasterio.open(out_prob_path, "w", **prob_profile) as dst:
            dst.write(ship_prob, 1)

    # ---- Finish profiling ----
    runtime_sec = time.perf_counter() - t0

    if profile:
        current_bytes, peak_bytes = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        peak_mb = peak_bytes / (1024 ** 2)
    else:
        peak_mb = 0.0

    metrics = {
        "runtime_sec": runtime_sec,
        "peak_memory_mb": peak_mb,
        "width": width,
        "height": height,
        "n_pixels": width * height,
        "n_bands": n_bands,
    }

    return ship_mask, ship_prob, metrics
