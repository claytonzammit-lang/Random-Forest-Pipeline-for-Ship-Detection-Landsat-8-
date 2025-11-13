import rasterio
import numpy as np
import xml.etree.ElementTree as ET
import os
import glob
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.inspection import permutation_importance
from sklearn.metrics import f1_score, make_scorer
from aux_functions_L2 import thermal_conversion, shapefile_mask, std_func, gradient_magnitude, relabel_tiles_with_all_ships_from_paths, clip_raster_by_rect_buffer, temperature_aware_cloud_mask_local

def TC_images_all(directory_path, land_shapefile_path):
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
    Train a Random Forest from tiled scenes, with SCENE-BASED splitting.
    Each scene is entirely assigned to either training or testing.

    Returns
    -------
    rf : RandomForestClassifier
    metrics : dict
    info : dict
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