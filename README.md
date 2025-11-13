# Random Forest Ship Detection — Landsat 8 Thermal Pipeline

This repository contains a full pipeline of how this Random Forest (RF) model for ship detection using Landsat 8 Band 10 thermal imagery was trained. The workflow covers preprocessing, feature extraction, labeling, tiling, and model training.

---

## Required Files per Scene

Each Landsat L2 scene requires the following files:

- Landsat 8 Band 10 (thermal infrared)
- MTL metadata file (XML format)
- QA pixel file

---

## Feature Generation Pipeline

Three feature layers are generated from Band 10 and fed into the RF classifier.

### 1. Absolute Temperature (Kelvin)

Band 10 is originally provided as Digital Numbers (DN). These are converted into absolute temperature using:

$A = M \cdot a + L$

Where:

- M = radiance multiplier (from metadata)
- L = radiance additive term (from metadata)
- a = DN array

---

### 2. Mean-Centered Local Temperature

To obtain meaningful temperature differences:

1. Land pixels are masked using a global land polygon.
2. Clouds are masked using the QA layer.
3. To prevent ships from being incorrectly masked as clouds, any pixel that is 0.5 K hotter than the tile mean is ignored during cloud masking.
4. The scene is split into 384 × 384 tiles to compute a localized sea-surface mean.
5. Each pixel’s absolute temperature is subtracted from its tile’s local mean.

This produces the mean-centered localized thermal index (Layer 1).

---

### 3. Local Standard Deviation

A 3 × 3 sliding window computes the local standard deviation of thermal values.

$\sigma = \sqrt{E[x^2] - (E[x])^2}$

This measures local texture, which helps distinguish:

- Smooth water surface (low std)
- Ship structures and wakes (high std)

This becomes Layer 2.

---

### 4. Gradient Magnitude

The Sobel operator estimates local thermal gradients:

$|\nabla T| = \sqrt{\left(\frac{\partial T}{\partial x}\right)^2 + \left(\frac{\partial T}{\partial y}\right)^2}$

This highlights sharp temperature boundaries (e.g., ship edges).

This becomes Layer 3.

---

### 5. Creating the Feature Stack

The three layers are stacked into a single 3-band GeoTIFF using the original georeferencing metadata from Band 10.

---

## Labeling and Tiling System

Ships must be manually labeled using polygons (stored as a shapefile per scene).

To reduce processing cost and focus the model on relevant data:

- For each ship polygon, a 50 × 50 pixel tile is extracted.
- Pixels inside polygons are labeled as 1 (ship).
- For every ship pixel, 5 random ocean pixels are sampled and labeled as 0 (water).

This produces a dataset with a 1:5 ship-to-water ratio.

---

## Random Forest Model Configuration

| Parameter           | Value                | Description                                         |
|---------------------|----------------------|-----------------------------------------------------|
| n_estimators        | 300                  | Number of trees                                     |
| max_depth           | None                 | Allow trees to fully grow                           |
| min_samples_leaf    | 2                    | Minimum samples per leaf                            |
| class_weight        | balanced_subsample   | Rebalances each bootstrap sample                    |
| n_jobs              | -1                   | Use all CPU cores                                   |

---

## Example Model Performance

| Class | Precision | Recall | F1-score | Support |
|-------|-----------|--------|----------|---------|
| Water | 0.99      | 0.93   | 0.96     | 557105  |
| Ship  | 0.73      | 0.97   | 0.83     | 111421  |
| **Accuracy** | — | — | **0.93** | 668526 |
| **Macro Avg** | 0.86 | 0.95 | 0.89 | 668526 |
| **Weighted Avg** | 0.95 | 0.93 | 0.94 | 668526 |

---
### Download Model

The trained Random Forest model (185 MB) is available here:

[Download from Google Drive] (https://drive.google.com/file/d/1XSNjG1070Qqyp5DoD9F4n3mClL9gIre8/view?usp=drive_link)
