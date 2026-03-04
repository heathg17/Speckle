"""
Flash Disruption Index (FDI) - Alternative speckle impact metric

Measures the overall impact of flash reflections on the fluorescent yellow
security patch, capturing both localized specular speckle AND diffuse washout
that the existing speckle_area_percent misses.

The FDI combines five sub-metrics computed from the yellow patch images:
  1. Color Fidelity Loss  - blue channel contamination from flash
  2. Saturation Loss       - HSV saturation drop (yellow -> white)
  3. Specular Fraction     - fraction of near-white saturated pixels
  4. Texture Disruption    - loss of fine spatial detail
  5. Flash Differential    - abnormal response to flash intensity change

Usage:
    python flash_disruption_index.py
"""

import json
import re
import argparse
import numpy as np
from PIL import Image
from pathlib import Path
from datetime import datetime
import matplotlib.pyplot as plt

# ============================================================
# CONFIGURATION
# ============================================================

SCRIPT_DIR = Path(__file__).parent
OUTPUT_DIR = SCRIPT_DIR / 'figures'
IMAGE_DIR = Path('/tmp/speckle_images/DiscoMotion')
REFERENCE_TAG_HINTS = ('F53S2', 'FEDS2', 'FEDS2A', 'S2')

plt.rcParams.update({
    'figure.facecolor': 'white',
    'axes.grid': False,
    'font.size': 10,
})

# ============================================================
# IMAGE-BASED METRICS
# ============================================================

def load_yellow_images(scan_dir):
    """Load yellow patch images sorted by flash intensity."""
    server_dir = scan_dir / 'server_data'
    if not server_dir.exists():
        return None

    images = []
    for png in server_dir.glob('frame_*_flash_*.yellow.png'):
        # Parse flash intensity from filename
        name = png.stem  # e.g. frame_5_flash_0.99.yellow
        parts = name.split('_flash_')
        if len(parts) != 2:
            continue
        flash_str = parts[1].replace('.yellow', '')
        try:
            flash = float(flash_str)
        except ValueError:
            continue
        arr = np.array(Image.open(png))[:, :, :3].astype(np.float32) / 255.0
        images.append((flash, arr))

    if not images:
        return None

    images.sort(key=lambda x: x[0])
    return images


def compute_color_fidelity_loss(img_rgb):
    """Measure blue channel contamination (0=pure yellow, 1=fully washed)."""
    r, g, b = img_rgb[:, :, 0], img_rgb[:, :, 1], img_rgb[:, :, 2]
    # For a good yellow patch: B << R,G. Flash washout increases B.
    rg_mean = (r.mean() + g.mean()) / 2
    if rg_mean < 0.01:
        return 1.0
    # Ratio of blue to average of R,G. Good yellow: ~0.05-0.15, washed: ~0.5-0.8
    ratio = b.mean() / rg_mean
    # Normalize: 0.1 -> 0, 0.7 -> 1
    return np.clip((ratio - 0.1) / 0.6, 0, 1)


def compute_saturation_loss(img_rgb):
    """Measure saturation loss in HSV space (0=saturated yellow, 1=desaturated)."""
    r, g, b = img_rgb[:, :, 0], img_rgb[:, :, 1], img_rgb[:, :, 2]
    cmax = np.maximum(np.maximum(r, g), b)
    cmin = np.minimum(np.minimum(r, g), b)
    delta = cmax - cmin
    # Saturation = delta / cmax (where cmax > 0)
    sat = np.where(cmax > 0.01, delta / cmax, 0)
    mean_sat = sat.mean()
    # Good yellow: saturation ~0.85-0.95, washed: ~0.1-0.4
    return np.clip(1.0 - (mean_sat - 0.1) / 0.8, 0, 1)


def compute_contrast_loss(img_rgb):
    """Measure loss of local contrast / dynamic range in green channel.

    Good fluorescent patches have high local contrast from the textured surface.
    Flash washout compresses the dynamic range, making the image uniformly flat.

    Calibrated from surveying 30 scans across the dataset:
      - Good yellow patch: MAD ~ 0.025-0.037, dyn_range ~ 0.13-0.20
      - Moderate:          MAD ~ 0.012-0.025, dyn_range ~ 0.09-0.13
      - Washed out patch:  MAD ~ 0.005-0.010, dyn_range ~ 0.06-0.08
    """
    from scipy.ndimage import uniform_filter
    g = img_rgb[:, :, 1]
    if g.shape[0] < 11 or g.shape[1] < 11:
        return 0.5

    # Local mean over 11x11 neighborhood
    local_mean = uniform_filter(g, size=11)
    local_dev = np.abs(g - local_mean)

    # Mean absolute deviation from local mean
    mad = local_dev.mean()

    # Dynamic range: P95 - P5 of green channel
    p5, p95 = np.percentile(g, [5, 95])
    dyn_range = p95 - p5

    # Normalize using calibrated thresholds
    # MAD: 0.025+ = good (score 0), <0.006 = bad (score 1)
    mad_score = np.clip(1.0 - (mad - 0.006) / 0.024, 0, 1)

    # Dynamic range: 0.15+ = good (score 0), <0.07 = bad (score 1)
    range_score = np.clip(1.0 - (dyn_range - 0.065) / 0.10, 0, 1)

    return 0.6 * mad_score + 0.4 * range_score


def compute_texture_disruption(img_rgb):
    """Measure loss of fine spatial texture (fluorescent grain pattern).

    The fluorescent material has a characteristic granular texture visible
    as high-frequency variation. Flash washout destroys this texture.

    Calibrated from real data:
      - Good patch: gradient_var ~ 0.002-0.004 (log10 ~ -2.4 to -2.6)
      - Washed out:  gradient_var ~ 0.00005-0.0002 (log10 ~ -3.7 to -4.3)
    """
    g = img_rgb[:, :, 1]
    if g.shape[0] < 5 or g.shape[1] < 5:
        return 0.5

    dy = np.diff(g, axis=0)
    dx = np.diff(g, axis=1)
    local_var = (dy**2).mean() + (dx**2).mean()

    # Normalize on log scale using calibrated thresholds
    log_var = np.log10(max(local_var, 1e-8))
    # log10(0.003) = -2.52 -> good (score 0)
    # log10(0.0001) = -4.0 -> bad (score 1)
    return np.clip((-log_var - 2.3) / 1.7, 0, 1)


def compute_blue_spatial_variance(img_rgb):
    """Measure spatial non-uniformity of blue channel contamination.

    Diffuse flash washout raises blue uniformly across the patch.
    Localized speckle creates spatially varying blue contamination.
    This metric captures both patterns by looking at mean AND variance of
    the blue-to-green ratio across spatial blocks.

    Calibrated from real data:
      - Good patch: mean B/G ~ 0.15, block std ~ 0.02
      - Washed out:  mean B/G ~ 0.55+, block std ~ 0.01 (uniform wash)
    """
    g = img_rgb[:, :, 1]
    b = img_rgb[:, :, 2]

    # Blue-to-green ratio per pixel (green is stable reference)
    safe_g = np.maximum(g, 0.02)
    bg_ratio = b / safe_g

    # Global mean of B/G ratio
    mean_bg = bg_ratio.mean()

    # Divide into blocks and compute block-level statistics
    bh, bw = max(g.shape[0] // 4, 1), max(g.shape[1] // 4, 1)
    block_means = []
    for i in range(0, g.shape[0] - bh + 1, bh):
        for j in range(0, g.shape[1] - bw + 1, bw):
            block = bg_ratio[i:i+bh, j:j+bw]
            block_means.append(block.mean())

    block_std = np.std(block_means) if len(block_means) > 1 else 0

    # Score: high mean B/G = washout, high block_std = localized flash effects
    # Normalize: mean B/G 0.15->0, 0.60->1
    mean_score = np.clip((mean_bg - 0.15) / 0.45, 0, 1)
    # Block variance adds to the score for localized hot-spots
    var_score = np.clip(block_std / 0.10, 0, 1)

    return 0.7 * mean_score + 0.3 * var_score


def normalize_fluorescence_intensity(img_rgb, target_green=0.55):
    """Normalize global intensity to reduce bias from intentional fluorescence variation."""
    g = img_rgb[:, :, 1]
    med_g = float(np.median(g))
    if med_g < 1e-4:
        return img_rgb
    scale = np.clip(target_green / med_g, 0.4, 2.5)
    return np.clip(img_rgb * scale, 0, 1)


def compute_submetrics(img_rgb):
    """Compute all sub-metrics on a single image."""
    return {
        'color_fidelity_loss': compute_color_fidelity_loss(img_rgb),
        'saturation_loss': compute_saturation_loss(img_rgb),
        'contrast_loss': compute_contrast_loss(img_rgb),
        'texture_disruption': compute_texture_disruption(img_rgb),
        'blue_spatial_var': compute_blue_spatial_variance(img_rgb),
    }


def compute_experimental_distributions(img_rgb):
    """Compute experimental scalar distributions for bimodality search."""
    eps = 1e-6
    r = img_rgb[:, :, 0]
    g = img_rgb[:, :, 1]
    b = img_rgb[:, :, 2]
    lum = (r + g + b) / 3.0

    total = r + g + b + eps
    sat = (np.max(img_rgb, axis=2) - np.min(img_rgb, axis=2)) / np.maximum(np.max(img_rgb, axis=2), eps)

    # Minkowski-style distance from ideal yellow [1,1,0]
    dy_r = 1.0 - r
    dy_g = 1.0 - g
    dy_b = b

    l1 = np.mean(np.abs(dy_r) + np.abs(dy_g) + np.abs(dy_b))
    l2 = np.mean(np.sqrt(dy_r**2 + dy_g**2 + dy_b**2))
    l3 = np.mean((np.abs(dy_r)**3 + np.abs(dy_g)**3 + np.abs(dy_b)**3) ** (1.0 / 3.0))

    bg = b / (g + eps)
    br = b / (r + eps)
    rg_over_b = (0.5 * (r + g)) / (b + eps)
    chrom_b = b / total

    white_frac = np.mean((lum > 0.75) & (sat < 0.20))
    hot_blue_frac = np.mean(bg > 0.55)

    return {
        'lum_mean': float(np.mean(lum)),
        'lum_p95': float(np.percentile(lum, 95)),
        'lum_std': float(np.std(lum)),
        'bg_ratio_mean': float(np.mean(bg)),
        'bg_ratio_p90': float(np.percentile(bg, 90)),
        'br_ratio_mean': float(np.mean(br)),
        'rg_over_b_mean': float(np.mean(rg_over_b)),
        'blue_chromaticity_mean': float(np.mean(chrom_b)),
        'white_fraction': float(white_frac),
        'hot_blue_fraction': float(hot_blue_frac),
        'minkowski_l1_yellow': float(l1),
        'minkowski_l2_yellow': float(l2),
        'minkowski_l3_yellow': float(l3),
    }


def compute_fdi(scan_dir):
    """Compute Flash Disruption Index for a scan directory.

    Returns dict with sub-metrics and composite FDI score, or None if images missing.

    Sub-metrics:
      - color_fidelity_loss: Blue contamination from flash (0=pure yellow, 1=washed)
      - saturation_loss: HSV saturation drop (0=vivid, 1=desaturated)
      - contrast_loss: Loss of local contrast/dynamic range (0=textured, 1=flat)
      - texture_disruption: Loss of fine grain texture (0=detailed, 1=smooth)
      - blue_spatial_var: Blue channel contamination pattern (0=clean, 1=contaminated)
    """
    images = load_yellow_images(scan_dir)
    if images is None or len(images) == 0:
        return None

    # Normalize each flash image first to account for intentional fluorescence variation.
    flashes = np.array([f for f, _ in images], dtype=float)
    norm_imgs = [normalize_fluorescence_intensity(img) for _, img in images]
    sub_all = [compute_submetrics(img) for img in norm_imgs]

    low_idx = 0
    high_idx = len(norm_imgs) - 1
    flash_span = max(flashes[high_idx] - flashes[low_idx], 1e-3)

    high = sub_all[high_idx]
    low = sub_all[low_idx]
    exp_high = compute_experimental_distributions(norm_imgs[high_idx])
    exp_low = compute_experimental_distributions(norm_imgs[low_idx])

    # Weighted composite score (state at high flash).
    weights = {
        'color_fidelity_loss': 0.25,
        'saturation_loss': 0.20,
        'contrast_loss': 0.15,
        'texture_disruption': 0.20,
        'blue_spatial_var': 0.20,
    }

    raw_state = float(sum(weights[k] * high[k] for k in weights))

    # Flash-response term: disruption growth per flash unit.
    delta_term = float(sum(weights[k] * max(high[k] - low[k], 0.0) for k in weights))
    flash_sensitivity = float(np.clip(delta_term / flash_span, 0, 1.5))

    # Composite raw FDI before cross-sample calibration.
    fdi_raw = 0.75 * raw_state + 0.25 * flash_sensitivity

    out = {
        'fdi_raw': fdi_raw,
        'fdi': fdi_raw,
        'flash_low': float(flashes[low_idx]),
        'flash_high': float(flashes[high_idx]),
        'flash_span': float(flash_span),
        'flash_sensitivity': flash_sensitivity,
        'color_fidelity_loss': high['color_fidelity_loss'],
        'saturation_loss': high['saturation_loss'],
        'contrast_loss': high['contrast_loss'],
        'texture_disruption': high['texture_disruption'],
        'blue_spatial_var': high['blue_spatial_var'],
        'color_fidelity_delta': max(high['color_fidelity_loss'] - low['color_fidelity_loss'], 0.0),
        'saturation_delta': max(high['saturation_loss'] - low['saturation_loss'], 0.0),
        'contrast_delta': max(high['contrast_loss'] - low['contrast_loss'], 0.0),
        'texture_delta': max(high['texture_disruption'] - low['texture_disruption'], 0.0),
        'blue_spatial_delta': max(high['blue_spatial_var'] - low['blue_spatial_var'], 0.0),
    }
    for k, v in exp_high.items():
        out[f'exp_{k}'] = float(v)
        out[f'exp_delta_{k}'] = float(v - exp_low[k])
    return out


# ============================================================
# DATA MATCHING
# ============================================================

def match_positions_to_scans(json_path, scan_base_dir):
    """Match JSON positions to extracted scan directories by timestamp."""
    json_path = Path(json_path)
    scan_base_dir = Path(scan_base_dir)

    with open(json_path) as f:
        raw = json.load(f)

    # Extract tag ID
    stem = json_path.stem
    match = re.search(r'_(\d{4}-\d{2}-\d{2})_(\d{2}-\d{2}-\d{2})$', stem)
    tag_id = stem[:match.start()] if match else stem

    # Get available scan directories
    scan_dirs = {}
    for d in scan_base_dir.iterdir():
        if d.is_dir() and d.name[0:4].isdigit():
            try:
                parts = d.name.split('_')
                h, m, s, ms = parts[1].split('-')
                dt = datetime.strptime(f'{parts[0]} {h}:{m}:{s}', '%Y-%m-%d %H:%M:%S')
                scan_dirs[d.name] = dt
            except (ValueError, IndexError):
                continue

    matches = []
    for pos in raw['positions']:
        if 'speckle_area_percent' not in pos:
            continue
        pt = datetime.strptime(pos['timestamp'], '%Y-%m-%dT%H:%M:%SZ')
        best_folder = None
        best_diff = 999999
        for fname, dt in scan_dirs.items():
            diff = abs((dt - pt).total_seconds())
            if diff < best_diff:
                best_diff = diff
                best_folder = fname
        if best_diff < 10 and best_folder:
            matches.append({
                'folder': best_folder,
                'scan_dir': scan_base_dir / best_folder,
                'speckle_area_pct': pos['speckle_area_percent'],
                'height': pos['height_mm'],
                'pitch': pos['pitch_degrees'],
                'roll': pos['roll_degrees'],
                'yaw': pos['yaw_degrees'],
            })

    return tag_id, matches


# ============================================================
# ANALYSIS & PLOTTING
# ============================================================

def run_fdi_analysis(json_path, scan_base_dir):
    """Run FDI analysis for a single tag and return results."""
    tag_id, matches = match_positions_to_scans(json_path, scan_base_dir)
    print(f"  {tag_id}: {len(matches)} matched positions")

    results = []
    for i, m in enumerate(matches):
        fdi_result = compute_fdi(m['scan_dir'])
        if fdi_result is not None:
            results.append({**m, **fdi_result, 'tag_id': tag_id})
        if (i + 1) % 50 == 0:
            print(f"    Processed {i+1}/{len(matches)}")

    print(f"    FDI computed for {len(results)}/{len(matches)} positions")
    return tag_id, results


def calibrate_fdi(all_results, ref_hints=REFERENCE_TAG_HINTS):
    """Calibrate raw FDI so best-performing reference sample maps near zero."""
    tag_ids = list(all_results.keys())
    nonempty_tags = [t for t in tag_ids if len(all_results.get(t, [])) > 0]
    ref_tag = None

    # Prefer explicit reference-like tags (e.g., F53S2 / FEDS2).
    for t in nonempty_tags:
        t_up = t.upper().replace('_', '')
        if any(h.upper().replace('_', '') in t_up for h in ref_hints):
            ref_tag = t
            break
    if ref_tag is None and nonempty_tags:
        ref_tag = nonempty_tags[0]

    ref_vals = np.array([r['fdi_raw'] for r in all_results.get(ref_tag, [])], dtype=float)
    all_vals = np.array(
        [r['fdi_raw'] for tid in tag_ids for r in all_results[tid]],
        dtype=float
    )

    if len(ref_vals) == 0 or len(all_vals) == 0:
        return {'reference_tag': ref_tag, 'baseline': 0.0, 'scale': 1.0}

    # Baseline from the low-FDI tail of reference sample.
    baseline = float(np.percentile(ref_vals, 20))
    scale = float(max(np.percentile(all_vals, 95) - baseline, 1e-3))

    for tid in tag_ids:
        for r in all_results[tid]:
            r['fdi'] = float(np.clip((r['fdi_raw'] - baseline) / scale, 0.0, 1.0))

    return {'reference_tag': ref_tag, 'baseline': baseline, 'scale': scale}


def _rankdata_average(a):
    order = np.argsort(a)
    ranks = np.empty(len(a), dtype=float)
    i = 0
    while i < len(a):
        j = i
        while j + 1 < len(a) and a[order[j + 1]] == a[order[i]]:
            j += 1
        rank = (i + j) / 2.0 + 1.0
        ranks[order[i:j + 1]] = rank
        i = j + 1
    return ranks


def _binary_auc(y_true, scores):
    y_true = np.asarray(y_true).astype(bool)
    scores = np.asarray(scores, dtype=float)
    n_pos = np.sum(y_true)
    n_neg = np.sum(~y_true)
    if n_pos == 0 or n_neg == 0:
        return np.nan
    ranks = _rankdata_average(scores)
    rank_sum_pos = np.sum(ranks[y_true])
    u = rank_sum_pos - n_pos * (n_pos + 1) / 2.0
    return float(u / (n_pos * n_neg))


def _find_bimodal_split(vals, bins=36):
    vals = np.asarray(vals, dtype=float)
    vals = vals[np.isfinite(vals)]
    if len(vals) < bins:
        return None

    vmin, vmax = np.percentile(vals, [1, 99])
    if vmax <= vmin:
        return None
    clipped = np.clip(vals, vmin, vmax)

    hist, _ = np.histogram(clipped, bins=bins, range=(vmin, vmax), density=True)
    if np.all(hist <= 0):
        return None

    peak_idx = [i for i in range(1, bins - 1) if hist[i] >= hist[i - 1] and hist[i] >= hist[i + 1]]
    if len(peak_idx) < 2:
        return {
            'score': 0.0,
            'valley_depth': 0.0,
            'peak_distance': 0.0,
            'threshold': float(np.median(clipped)),
            'vmin': float(vmin),
            'vmax': float(vmax),
        }

    best = None
    for i in range(len(peak_idx)):
        for j in range(i + 1, len(peak_idx)):
            a = peak_idx[i]
            b = peak_idx[j]
            if b - a < 3:
                continue
            valley_rel = int(np.argmin(hist[a:b + 1]))
            valley_idx = a + valley_rel
            valley = hist[valley_idx]
            peak_h = max(hist[a], hist[b], 1e-8)
            valley_depth = float(np.clip((min(hist[a], hist[b]) - valley) / peak_h, 0, 1))
            peak_dist = float((b - a) / (bins - 1))
            objective = valley_depth * peak_dist
            if best is None or objective > best['objective']:
                best = {
                    'valley_depth': valley_depth,
                    'peak_distance': peak_dist,
                    'objective': objective,
                    'valley_idx': valley_idx,
                }

    if best is None:
        return {
            'score': 0.0,
            'valley_depth': 0.0,
            'peak_distance': 0.0,
            'threshold': float(np.median(clipped)),
            'vmin': float(vmin),
            'vmax': float(vmax),
        }

    valley_depth = best['valley_depth']
    peak_dist = best['peak_distance']
    score = float(np.clip(0.6 * valley_depth + 0.4 * peak_dist, 0, 1))
    edges = np.linspace(vmin, vmax, bins + 1)
    thr = float((edges[best['valley_idx']] + edges[best['valley_idx'] + 1]) / 2.0)
    return {
        'score': score,
        'valley_depth': valley_depth,
        'peak_distance': peak_dist,
        'threshold': thr,
        'vmin': float(vmin),
        'vmax': float(vmax),
    }


def analyze_experimental_distributions(all_results, output_dir):
    """Rank experimental metrics by intrinsic bimodality and visualize metric-defined clusters."""
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)

    all_rows = [r for rows in all_results.values() for r in rows]
    if not all_rows:
        return []

    sap = np.array([r['speckle_area_pct'] for r in all_rows], dtype=float)  # diagnostic only

    exp_keys = sorted([
        k for k in all_rows[0].keys()
        if k.startswith('exp_') and isinstance(all_rows[0][k], (int, float))
    ])

    ranking = []
    for key in exp_keys:
        vals = np.array([r.get(key, np.nan) for r in all_rows], dtype=float)
        valid = np.isfinite(vals) & np.isfinite(sap)
        if np.sum(valid) < 20:
            continue

        x = vals[valid]
        split = _find_bimodal_split(x)
        if split is None:
            continue

        thr = split['threshold']
        low = x <= thr
        high = x > thr
        if np.sum(low) == 0 or np.sum(high) == 0:
            continue

        n_low = int(np.sum(low))
        n_high = int(np.sum(high))
        n_tot = n_low + n_high
        balance = float(min(n_low, n_high) / max(n_low, n_high))
        min_frac = float(min(n_low, n_high) / max(n_tot, 1))
        sep = float(
            np.clip(
                abs(np.mean(x[high]) - np.mean(x[low])) / (np.std(x) + 1e-8),
                0, 3
            ) / 3.0
        )
        balance_gate = float(np.clip(min_frac / 0.20, 0, 1))  # full score only if minority >=20%
        combined_raw = 0.50 * split['score'] + 0.30 * sep + 0.20 * balance
        combined = float(np.clip(combined_raw * (0.35 + 0.65 * balance_gate), 0, 1))

        # Diagnostic only: does the metric split align with known speckle values?
        diag_sap = sap[valid]
        auc_diag = _binary_auc(diag_sap > 0, x)

        ranking.append({
            'metric': key,
            'combined_score': combined,
            'bimodality_score': split['score'],
            'valley_depth': split['valley_depth'],
            'peak_distance': split['peak_distance'],
            'threshold': thr,
            'cluster_balance': balance,
            'min_cluster_fraction': min_frac,
            'cluster_separation': sep,
            'diagnostic_auc_nonzero': float(np.nan if np.isnan(auc_diag) else auc_diag),
            'n': int(np.sum(valid)),
        })

    ranking.sort(key=lambda r: (-r['combined_score'], -r['bimodality_score'], r['metric']))
    if not ranking:
        return []

    # Save ranking table
    csv_path = output_dir / 'experimental_bimodality_ranking.csv'
    with open(csv_path, 'w') as f:
        header = [
            'metric', 'combined_score', 'bimodality_score', 'valley_depth',
            'peak_distance', 'threshold', 'cluster_balance', 'min_cluster_fraction', 'cluster_separation',
            'diagnostic_auc_nonzero', 'n'
        ]
        f.write(','.join(header) + '\n')
        for r in ranking:
            f.write(
                f"{r['metric']},{r['combined_score']:.6f},{r['bimodality_score']:.6f},"
                f"{r['valley_depth']:.6f},{r['peak_distance']:.6f},"
                f"{r['threshold']:.6f},{r['cluster_balance']:.6f},{r['min_cluster_fraction']:.6f},{r['cluster_separation']:.6f},"
                f"{r['diagnostic_auc_nonzero']:.6f},{r['n']}\n"
            )
    print(f"  Saved: {csv_path.name}")

    # Plot top metrics as pooled histograms split by metric-defined low/high clusters.
    top = ranking[:8]
    n_top = len(top)
    rows = 2
    cols = 4
    fig, axes = plt.subplots(rows, cols, figsize=(20, 9))
    fig.suptitle('Experimental Distribution Sweep: Top Bimodal Candidates', fontsize=15, fontweight='bold')

    for i, item in enumerate(top):
        ax = axes.flat[i]
        key = item['metric']
        vals = np.array([r.get(key, np.nan) for r in all_rows], dtype=float)
        valid = np.isfinite(vals)
        x = vals[valid]
        high = x > item['threshold']
        if np.sum(valid) > 0:
            if np.sum(~high) > 0:
                ax.hist(x[~high], bins=30, alpha=0.55, color='#1976D2', density=True, label='Low cluster')
            if np.sum(high) > 0:
                ax.hist(x[high], bins=30, alpha=0.50, color='#d32f2f', density=True, label='High cluster')
        ax.axvline(item['threshold'], color='black', linestyle='--', linewidth=1)
        ax.set_title(
            f"{key}\nscore={item['combined_score']:.3f}, bi={item['bimodality_score']:.3f}, thr={item['threshold']:.3f}",
            fontsize=9
        )
        ax.grid(True, axis='y', alpha=0.3)
        if i == 0:
            ax.legend(fontsize=8)

    for j in range(n_top, rows * cols):
        axes.flat[j].axis('off')

    plt.tight_layout(rect=[0, 0, 1, 0.93])
    save_path = output_dir / 'experimental_bimodal_distributions.png'
    plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  Saved: {save_path.name}")

    # Visual validation for top metric candidates (metric-defined low/high).
    def load_display_image(scan_dir):
        master = scan_dir / 'server_data' / 'frame_master.yellow.png'
        if master.exists():
            return np.array(Image.open(master))[:, :, :3]
        yellow_imgs = load_yellow_images(scan_dir)
        if yellow_imgs:
            return (yellow_imgs[-1][1] * 255).astype(np.uint8)
        return None

    for item in ranking[:3]:
        key = item['metric']
        vals = np.array([r.get(key, np.nan) for r in all_rows], dtype=float)
        valid_idx = np.where(np.isfinite(vals))[0]
        if len(valid_idx) < 10:
            continue
        metric_rows = [all_rows[i] for i in valid_idx]
        metric_vals = vals[valid_idx]
        order = np.argsort(metric_vals)
        picks_idx = [order[0], order[len(order)//4], order[len(order)//2], order[3*len(order)//4], order[-1]]
        picks = [metric_rows[i] for i in picks_idx]

        fig, axes = plt.subplots(2, 5, figsize=(22, 8))
        fig.suptitle(f'Metric Visual Validation: {key} (threshold={item["threshold"]:.3f})',
                     fontsize=14, fontweight='bold')

        for col, r in enumerate(picks):
            img = load_display_image(r['scan_dir'])
            if img is None:
                continue
            mval = r.get(key, np.nan)
            cluster = 'HIGH' if mval > item['threshold'] else 'LOW'

            axes[0, col].imshow(img)
            axes[0, col].set_title(
                f"{r['tag_id']}\n{key}={mval:.3f} ({cluster})\nFDI={r['fdi']:.3f}, Speckle={r['speckle_area_pct']:.3f}%",
                fontsize=8, fontweight='bold'
            )
            axes[0, col].set_xticks([])
            axes[0, col].set_yticks([])

            ax = axes[1, col]
            yellow_imgs = load_yellow_images(r['scan_dir'])
            if yellow_imgs:
                _, hi = yellow_imgs[-1]
                g = np.maximum(hi[:, :, 1], 1e-4)
                ratio = (hi[:, :, 2] / g).ravel()
                ax.hist(ratio, bins=30, color='#1976D2', alpha=0.75, edgecolor='black', linewidth=0.3)
                ax.axvline(np.median(ratio), color='red', linestyle='--', linewidth=1)
            ax.set_xlabel('Blue/Green pixel ratio')
            ax.set_ylabel('Count' if col == 0 else '')
            ax.grid(True, axis='y', alpha=0.3)

        axes[0, 0].set_ylabel('Image', fontsize=10)
        axes[1, 0].set_ylabel('Histogram', fontsize=10)
        plt.tight_layout(rect=[0, 0, 1, 0.93])
        save_path = output_dir / f'experimental_visual_{key}.png'
        plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
        plt.close()
        print(f"  Saved: {save_path.name}")

    return ranking


def plot_fdi_focus(all_results, output_dir, calibration_info):
    """Generate focused outputs: histogram trends + visual comparison."""
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)

    tag_ids = list(all_results.keys())
    tag_colors = plt.cm.tab10(np.linspace(0, 1, len(tag_ids)))
    all_rows = [r for tid in tag_ids for r in all_results[tid]]
    all_sap = np.array([r['speckle_area_pct'] for r in all_rows], dtype=float)
    all_fdi = np.array([r['fdi'] for r in all_rows], dtype=float)

    # ---- PLOT 1: Histogram trends (core diagnostic) ----
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('FDI Histogram Trends and Calibration Diagnostics', fontsize=14, fontweight='bold')

    ax = axes[0, 0]
    for tag_id, color in zip(tag_ids, tag_colors):
        vals = np.array([r['fdi'] for r in all_results[tag_id]], dtype=float)
        if len(vals) > 0:
            ax.hist(vals, bins=25, alpha=0.35, color=color, label=tag_id, density=True)
    ax.set_xlabel('Calibrated FDI')
    ax.set_ylabel('Density')
    ax.set_title('FDI Distribution by Tag')
    ax.legend(fontsize=8)
    ax.grid(True, axis='y', alpha=0.3)

    ax = axes[0, 1]
    for tag_id, color in zip(tag_ids, tag_colors):
        vals = np.array([r['speckle_area_pct'] for r in all_results[tag_id]], dtype=float)
        if len(vals) > 0:
            ax.hist(vals, bins=25, alpha=0.35, color=color, label=tag_id, density=True)
    ax.set_xlabel('Speckle Area (%)')
    ax.set_ylabel('Density')
    ax.set_title('Speckle Distribution by Tag')
    ax.grid(True, axis='y', alpha=0.3)

    ax = axes[1, 0]
    h = ax.hist2d(all_sap, all_fdi, bins=(30, 30), cmap='YlOrRd')
    plt.colorbar(h[3], ax=ax, label='Count')
    if np.std(all_sap) > 0 and np.std(all_fdi) > 0:
        corr = np.corrcoef(all_sap, all_fdi)[0, 1]
    else:
        corr = np.nan
    ax.set_xlabel('Speckle Area (%)')
    ax.set_ylabel('Calibrated FDI')
    ax.set_title(f'Joint Histogram (r={corr:.3f})')

    ax = axes[1, 1]
    ref_tag = calibration_info['reference_tag']
    ref_vals = np.array([r['fdi'] for r in all_results.get(ref_tag, [])], dtype=float)
    other_vals = np.array([r['fdi'] for tid in tag_ids if tid != ref_tag for r in all_results[tid]], dtype=float)
    if len(ref_vals) > 0:
        ax.hist(ref_vals, bins=25, alpha=0.6, color='#2E7D32', label=f'{ref_tag} (reference)', density=True)
    if len(other_vals) > 0:
        ax.hist(other_vals, bins=25, alpha=0.5, color='#424242', label='All other tags', density=True)
    ax.axvline(0.0, color='black', linestyle='--', linewidth=1)
    ax.set_xlabel('Calibrated FDI')
    ax.set_ylabel('Density')
    ax.set_title('Reference-vs-Other Separation')
    ax.legend(fontsize=8)
    ax.grid(True, axis='y', alpha=0.3)

    plt.tight_layout(rect=[0, 0, 1, 0.93])
    save_path = output_dir / 'fdi_histogram_trends.png'
    plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  Saved: {save_path.name}")

    def load_display_image(scan_dir):
        master = scan_dir / 'server_data' / 'frame_master.yellow.png'
        if master.exists():
            return np.array(Image.open(master))[:, :, :3]
        yellow_imgs = load_yellow_images(scan_dir)
        if yellow_imgs:
            return (yellow_imgs[-1][1] * 255).astype(np.uint8)
        return None

    def representative_rows(rows, ref_tag=None):
        rows = sorted(rows, key=lambda r: r['fdi'])
        n = len(rows)
        if n == 0:
            return []
        if ref_tag:
            ref_rows = [r for r in rows if r.get('tag_id') == ref_tag]
            ref_pick = ref_rows[0] if ref_rows else rows[0]
            return [ref_pick, rows[n//5], rows[2*n//5], rows[3*n//5], rows[-1]]
        return [rows[0], rows[n//4], rows[n//2], rows[3*n//4], rows[-1]]

    def plot_visual_comparison(rows, save_name, title, ref_tag=None):
        picks = representative_rows(rows, ref_tag=ref_tag)
        if not picks:
            return
        fig, axes = plt.subplots(2, 5, figsize=(22, 8))
        fig.suptitle(title, fontsize=14, fontweight='bold')

        for col, r in enumerate(picks):
            scan_dir = r['scan_dir']
            img = load_display_image(scan_dir)
            if img is None:
                continue

            axes[0, col].imshow(img)
            axes[0, col].set_title(
                f"{r['tag_id']}\nFDI={r['fdi']:.3f} (raw {r['fdi_raw']:.3f})\nSpeckle={r['speckle_area_pct']:.3f}%",
                fontsize=8, fontweight='bold')
            axes[0, col].set_xticks([])
            axes[0, col].set_yticks([])
            axes[0, col].text(0.02, 0.02, f"P={r['pitch']}° R={r['roll']}°",
                              transform=axes[0, col].transAxes, fontsize=7,
                              color='white', va='bottom',
                              bbox=dict(boxstyle='round', facecolor='black', alpha=0.6))

            ax = axes[1, col]
            yellow_imgs = load_yellow_images(scan_dir)
            if yellow_imgs:
                _, hi = yellow_imgs[-1]
                g = np.maximum(hi[:, :, 1], 1e-4)
                ratio = (hi[:, :, 2] / g).ravel()
                ax.hist(ratio, bins=30, color='#1976D2', alpha=0.75, edgecolor='black', linewidth=0.3)
                ax.axvline(np.median(ratio), color='red', linestyle='--', linewidth=1)
            ax.set_xlabel('Blue/Green pixel ratio')
            ax.set_ylabel('Count' if col == 0 else '')
            ax.set_title(f"Flash span={r.get('flash_span', 0):.2f}\nSens={r.get('flash_sensitivity', 0):.3f}",
                         fontsize=9)
            ax.grid(True, axis='y', alpha=0.3)

        axes[0, 0].set_ylabel('Image', fontsize=10)
        axes[1, 0].set_ylabel('Histogram', fontsize=10)
        plt.tight_layout(rect=[0, 0, 1, 0.93])
        save_path = output_dir / save_name
        plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
        plt.close()
        print(f"  Saved: {save_path.name}")

    # ---- PLOT 2: Visual comparison (pooled + per-tag) ----
    plot_visual_comparison(
        all_rows,
        'fdi_visual_comparison.png',
        'Visual Validation: Calibrated FDI vs Speckle Area (Pooled)',
        ref_tag=calibration_info['reference_tag']
    )

    for tag_id in tag_ids:
        tag_rows = all_results[tag_id]
        plot_visual_comparison(
            tag_rows,
            f'fdi_visual_comparison_{tag_id}.png',
            f'Visual Validation: {tag_id}',
            ref_tag=None
        )

    # ---- Print summary statistics ----
    print(f"\n{'='*60}")
    print("  FDI FOCUSED SUMMARY")
    print(f"{'='*60}")
    print(f"  Calibration reference: {calibration_info['reference_tag']}")
    print(f"  Baseline (raw FDI P20 of reference): {calibration_info['baseline']:.4f}")
    print(f"  Scale (global raw FDI P95 - baseline): {calibration_info['scale']:.4f}")
    for tag_id in tag_ids:
        results = all_results[tag_id]
        sap = np.array([r['speckle_area_pct'] for r in results])
        fdi = np.array([r['fdi'] for r in results])
        raw_fdi = np.array([r['fdi_raw'] for r in results])
        print(f"\n  {tag_id} (n={len(results)}):")
        print(f"    Speckle Area: mean={sap.mean():.4f}%, max={sap.max():.4f}%, nonzero={np.sum(sap>0)}")
        print(f"    FDI raw:      mean={raw_fdi.mean():.4f}, max={raw_fdi.max():.4f}")
        print(f"    FDI cal:      mean={fdi.mean():.4f}, max={fdi.max():.4f}, >0.1={np.sum(fdi>0.1)}")
        if np.std(sap) > 0 and np.std(fdi) > 0:
            r = np.corrcoef(sap, fdi)[0, 1]
            print(f"    Correlation:  r={r:.4f}")
        # Cases FDI catches but speckle misses
        fdi_high = fdi > np.percentile(fdi, 75)
        sap_zero = sap == 0
        print(f"    FDI>P75 but speckle=0: {np.sum(fdi_high & sap_zero)} positions")


# ============================================================
# MAIN
# ============================================================

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='FDI analysis with reference calibration and histogram-focused outputs.'
    )
    parser.add_argument(
        '--image-dir',
        type=Path,
        default=IMAGE_DIR,
        help='Directory containing extracted scan folders (default: /tmp/speckle_images/DiscoMotion)'
    )
    parser.add_argument(
        '--json',
        nargs='*',
        type=Path,
        default=None,
        help='Optional explicit JSON files to analyze'
    )
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(exist_ok=True, parents=True)

    if args.json:
        tags = {p.stem: p for p in args.json}
    else:
        tags = {
            'F53_HER1A': SCRIPT_DIR / 'F53_HER1A_2026-03-02_14-06-53.json',
            'F53_FED2B': SCRIPT_DIR / 'F53_FED2B_2026-03-02_11-26-09.json',
            'F53_FED2C': SCRIPT_DIR / 'F53_FED2C_2026-03-02_11-26-09.json',
            'F53_FEDS2A': SCRIPT_DIR / 'F53_FEDS2A_2026-02-26_10-26-33.json',
        }

    if not args.image_dir.exists():
        print(f"Image directory not found: {args.image_dir}")
        raise SystemExit(1)

    all_results = {}
    for tag_name, json_path in tags.items():
        print(f"\nAnalyzing {tag_name}...")
        tag_id, results = run_fdi_analysis(json_path, args.image_dir)
        if results:
            all_results[tag_id] = results
        else:
            print(f"  Warning: no matched scan folders for {tag_id}.")

    if all_results:
        calibration_info = calibrate_fdi(all_results)
        print(f"\nGenerating focused visual outputs...")
        plot_fdi_focus(all_results, OUTPUT_DIR, calibration_info)
        print(f"\nRunning experimental distribution sweep...")
        ranking = analyze_experimental_distributions(all_results, OUTPUT_DIR)
        if ranking:
            print("\nTop experimental distributions by bimodality/separation:")
            for i, row in enumerate(ranking[:8], start=1):
                print(
                    f"  {i}. {row['metric']}: score={row['combined_score']:.3f}, "
                    f"bimodal={row['bimodality_score']:.3f}, "
                    f"balance={row['cluster_balance']:.3f}, min_frac={row['min_cluster_fraction']:.3f}, "
                    f"sep={row['cluster_separation']:.3f}, "
                    f"diagnostic AUC(nonzero)={row['diagnostic_auc_nonzero']:.3f}"
                )

    print(f"\nAll figures saved to: {OUTPUT_DIR}")
    print("Done!")
