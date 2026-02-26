"""
Speckle Analysis - Fluorescent security tag speckling under smartphone flash

Analyzes 4D parameter sweeps (height, yaw, pitch, roll) from the automated
testing rig to characterize reflective speckling across tag orientations.

Usage:
    python SpeckleAnalysis.py                          # Analyze all JSON files in directory
    python SpeckleAnalysis.py file1.json               # Analyze specific file
    python SpeckleAnalysis.py file1.json file2.json    # Compare datasets
"""

import json
import re
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys

# ============================================================
# CONFIGURATION
# ============================================================

SCRIPT_DIR = Path(__file__).parent
OUTPUT_DIR = SCRIPT_DIR / 'figures'

plt.rcParams.update({
    'figure.facecolor': 'white',
    'axes.grid': False,
    'font.size': 10,
})

# ============================================================
# DATA LOADING
# ============================================================

def extract_tag_id(json_path):
    """Extract tag ID from filename pattern: {TAG_ID}_{YYYY-MM-DD}_{HH-MM-SS}.json"""
    stem = json_path.stem
    match = re.search(r'_(\d{4}-\d{2}-\d{2})_(\d{2}-\d{2}-\d{2})$', stem)
    if match:
        return stem[:match.start()]
    return stem


def build_data_arrays(positions):
    """Build numpy arrays from positions list (speckle entries only)."""
    # Filter to positions that have speckle data
    speckle_positions = [p for p in positions if 'speckle_area_percent' in p]
    n = len(speckle_positions)

    heights = np.zeros(n, dtype=int)
    yaws = np.zeros(n, dtype=float)
    pitches = np.zeros(n, dtype=float)
    rolls = np.zeros(n, dtype=float)
    speckle_area = np.zeros(n, dtype=float)

    for i, pos in enumerate(speckle_positions):
        heights[i] = pos['height_mm']
        yaws[i] = pos['yaw_degrees']
        pitches[i] = pos['pitch_degrees']
        rolls[i] = pos['roll_degrees']
        speckle_area[i] = pos['speckle_area_percent']

    return {
        'heights': heights,
        'yaws': yaws,
        'pitches': pitches,
        'rolls': rolls,
        'speckle_area': speckle_area,
        'total_positions': len(positions),
        'speckle_positions': n,
    }


def get_axis_values(sweep_config):
    """Extract unique axis values from sweep config."""
    axes = {}
    for axis_name in ['height', 'yaw', 'pitch', 'roll']:
        cfg = sweep_config[axis_name]
        vals = np.arange(cfg['min'], cfg['max'] + cfg['step'], cfg['step'])
        axes[axis_name] = vals.astype(int) if axis_name == 'height' else vals
    return axes


def load_sweep(json_path):
    """Load a sweep JSON file and return structured data."""
    json_path = Path(json_path)
    with open(json_path, 'r') as f:
        raw = json.load(f)

    tag_id = extract_tag_id(json_path)
    positions = raw['positions']
    sweep_config = raw['sweep']['config']
    sweep_results = raw['sweep']
    data = build_data_arrays(positions)
    axis_values = get_axis_values(sweep_config)

    return {
        'tag_id': tag_id,
        'data': data,
        'axis_values': axis_values,
        'sweep_config': sweep_config,
        'sweep_results': sweep_results,
        'json_path': json_path,
    }


# ============================================================
# METRIC COMPUTATION
# ============================================================

def compute_speckle_by_param(data, param_name, param_values):
    """Compute mean speckle grouped by a parameter."""
    params = data[param_name]
    means = []
    stds = []
    for val in param_values:
        mask = np.isclose(params, val)
        speckles = data['speckle_area'][mask]
        if len(speckles) > 0:
            means.append(np.mean(speckles))
            stds.append(np.std(speckles))
        else:
            means.append(np.nan)
            stds.append(np.nan)
    return np.array(means), np.array(stds)


def compute_2d_grid(data, row_param, row_values, col_param, col_values,
                    fixed_params=None, metric='speckle_mean'):
    """Compute a 2D grid of speckle metric for heatmap display."""
    grid = np.full((len(row_values), len(col_values)), np.nan)
    rows = data[row_param]
    cols = data[col_param]

    for i, rv in enumerate(row_values):
        for j, cv in enumerate(col_values):
            mask = np.isclose(rows, rv) & np.isclose(cols, cv)

            if fixed_params:
                for fp_name, fp_val in fixed_params.items():
                    mask &= np.isclose(data[fp_name], fp_val)

            speckles = data['speckle_area'][mask]
            if len(speckles) > 0:
                if metric == 'speckle_mean':
                    grid[i, j] = np.mean(speckles)
                elif metric == 'speckle_max':
                    grid[i, j] = np.max(speckles)

    return grid


# ============================================================
# PLOT 1: SPECKLE AREA HEATMAP GRID
# ============================================================

def plot_speckle_heatmap_grid(data, axis_values, tag_id):
    """Speckle area heatmaps faceted by height x yaw."""
    height_vals = axis_values['height']
    # Only show heights that have speckle data
    height_vals = np.array([h for h in height_vals
                            if np.any(np.isclose(data['heights'], h))])
    yaw_vals = axis_values['yaw']
    pitch_vals = axis_values['pitch']
    roll_vals = axis_values['roll']

    if len(height_vals) == 0:
        print("  Skipped speckle grid: no speckle data")
        return

    fig, axes = plt.subplots(len(height_vals), len(yaw_vals),
                              figsize=(14, 4 * len(height_vals)), squeeze=False)
    fig.suptitle(f'Speckle Area (%)\nTag: {tag_id}',
                 fontsize=16, fontweight='bold', y=0.98)

    vmax = np.max(data['speckle_area']) if len(data['speckle_area']) > 0 else 0.1

    for i, h in enumerate(height_vals):
        for j, y in enumerate(yaw_vals):
            ax = axes[i, j]
            grid = compute_2d_grid(data, 'pitches', pitch_vals, 'rolls', roll_vals,
                                   fixed_params={'heights': h, 'yaws': y},
                                   metric='speckle_mean')

            ax_im = ax.imshow(grid, cmap='YlOrRd', aspect='equal',
                             origin='lower', interpolation='nearest',
                             vmin=0, vmax=vmax)

            ax.set_xticks(range(len(roll_vals)))
            ax.set_xticklabels([f'{int(v)}' for v in roll_vals], fontsize=7)
            ax.set_yticks(range(len(pitch_vals)))
            ax.set_yticklabels([f'{int(v)}' for v in pitch_vals], fontsize=7)

            if i == len(height_vals) - 1:
                ax.set_xlabel('Roll (deg)', fontsize=8)
            if j == 0:
                ax.set_ylabel('Pitch (deg)', fontsize=8)

            ax.set_title(f'H={h}mm, Yaw={y}\u00b0', fontsize=9, pad=4)

            # Annotate cells with values
            for ri in range(len(pitch_vals)):
                for ci in range(len(roll_vals)):
                    val = grid[ri, ci]
                    if not np.isnan(val):
                        color = 'white' if val > vmax * 0.6 else 'black'
                        ax.text(ci, ri, f'{val:.3f}', ha='center', va='center',
                               fontsize=5, color=color)

    # Shared colorbar
    cbar_ax = fig.add_axes([0.92, 0.15, 0.015, 0.7])
    cbar = fig.colorbar(ax_im, cax=cbar_ax)
    cbar.set_label('Speckle Area (%)', fontsize=10)

    plt.subplots_adjust(left=0.05, right=0.88, top=0.90, bottom=0.08, wspace=0.3, hspace=0.35)
    save_path = OUTPUT_DIR / f'{tag_id}_speckle_grid.png'
    plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  Saved: {save_path.name}")


# ============================================================
# PLOT 2: MARGINAL EFFECTS (SPECKLE ONLY)
# ============================================================

def plot_marginal_effects(data, axis_values, tag_id):
    """Mean speckle as a function of each parameter independently."""
    param_map = {
        'Height (mm)': ('heights', axis_values['height']),
        'Yaw (deg)': ('yaws', axis_values['yaw']),
        'Pitch (deg)': ('pitches', axis_values['pitch']),
        'Roll (deg)': ('rolls', axis_values['roll']),
    }

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f'Mean Speckle Area by Parameter\nTag: {tag_id}',
                 fontsize=16, fontweight='bold')

    color = '#d32f2f'

    for ax, (label, (param_name, param_vals)) in zip(axes.flat, param_map.items()):
        # Only show params that have data
        has_data = [np.any(np.isclose(data[param_name], v)) for v in param_vals]
        param_vals_filtered = param_vals[has_data]

        means, stds = compute_speckle_by_param(data, param_name, param_vals_filtered)
        x = np.arange(len(param_vals_filtered))
        valid = ~np.isnan(means)

        if np.any(valid):
            ax.bar(x[valid], means[valid], 0.6, color=color, alpha=0.7, edgecolor='black',
                   linewidth=0.5)
            ax.errorbar(x[valid], means[valid], yerr=stds[valid],
                       fmt='none', color='black', capsize=4, linewidth=1.5)

        ax.set_xticks(x)
        ax.set_xticklabels([f'{v:.0f}' if v == int(v) else f'{v}' for v in param_vals_filtered])
        ax.set_xlabel(label, fontsize=11)
        ax.set_ylabel('Mean Speckle Area (%)', fontsize=10)
        ax.set_title(label, fontsize=12, fontweight='bold')
        ax.set_ylim(bottom=0)
        ax.grid(True, axis='y', alpha=0.3)

    plt.tight_layout(rect=[0, 0, 1, 0.93])
    save_path = OUTPUT_DIR / f'{tag_id}_marginal_effects.png'
    plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  Saved: {save_path.name}")


# ============================================================
# PLOT 3: ANGULAR SENSITIVITY
# ============================================================

def plot_angular_sensitivity(data, axis_values, tag_id):
    """Angular sensitivity: pitch-roll plane contour + radial plot."""
    pitch_vals = axis_values['pitch']
    roll_vals = axis_values['roll']

    # Heights with speckle data
    height_vals = np.array([h for h in axis_values['height']
                            if np.any(np.isclose(data['heights'], h))])

    if len(height_vals) == 0:
        print("  Skipped angular sensitivity: no speckle data")
        return

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle(f'Angular Sensitivity Analysis\nTag: {tag_id}',
                 fontsize=16, fontweight='bold')

    # Left: Pitch-roll plane contour at height with most speckle
    best_h = None
    best_mean = 0
    for h in height_vals:
        mask = np.isclose(data['heights'], h)
        sp = data['speckle_area'][mask]
        if len(sp) > 0 and np.mean(sp) > best_mean:
            best_mean = np.mean(sp)
            best_h = h

    if best_h is not None:
        grid = compute_2d_grid(data, 'pitches', pitch_vals, 'rolls', roll_vals,
                               fixed_params={'heights': best_h},
                               metric='speckle_mean')

        P, R = np.meshgrid(roll_vals, pitch_vals)
        contour = ax1.contourf(P, R, grid, levels=15, cmap='YlOrRd')
        ax1.contour(P, R, grid, levels=15, colors='gray', linewidths=0.3, alpha=0.5)
        plt.colorbar(contour, ax=ax1, label='Speckle Area (%)')
        ax1.set_xlabel('Roll (deg)', fontsize=11)
        ax1.set_ylabel('Pitch (deg)', fontsize=11)
        ax1.set_title(f'Speckle at H={best_h}mm\n(averaged over yaw)', fontsize=12)
        ax1.set_aspect('equal')
        ax1.axhline(0, color='gray', linewidth=0.5, linestyle='--')
        ax1.axvline(0, color='gray', linewidth=0.5, linestyle='--')

    # Right: Mean speckle vs angular deviation, per height
    colors = plt.cm.tab10(np.linspace(0, 1, max(len(height_vals), 1)))
    for h, clr in zip(height_vals, colors):
        mask = np.isclose(data['heights'], h)
        p = data['pitches'][mask]
        r = data['rolls'][mask]
        sp = data['speckle_area'][mask]

        ang_dist = np.sqrt(p**2 + r**2)
        unique_dists = np.sort(np.unique(np.round(ang_dist, 1)))

        mean_speckles = []
        for d in unique_dists:
            d_mask = np.isclose(np.round(ang_dist, 1), d)
            vals = sp[d_mask]
            mean_speckles.append(np.mean(vals) if len(vals) > 0 else np.nan)

        ax2.plot(unique_dists, mean_speckles, 'o-', color=clr,
                label=f'H={h}mm', linewidth=2, markersize=5)

    ax2.set_xlabel('Angular Deviation from Normal (deg)', fontsize=11)
    ax2.set_ylabel('Mean Speckle Area (%)', fontsize=11)
    ax2.set_title('Speckle vs Angular Deviation', fontsize=12)
    ax2.legend(fontsize=10)
    ax2.set_ylim(bottom=0)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout(rect=[0, 0, 1, 0.92])
    save_path = OUTPUT_DIR / f'{tag_id}_angular_sensitivity.png'
    plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  Saved: {save_path.name}")


# ============================================================
# PLOT 4: SUMMARY STATISTICS
# ============================================================

def plot_summary_statistics(data, axis_values, sweep_config, sweep_results, tag_id):
    """Text-based summary figure focused on speckle."""
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.axis('off')

    sp = data['speckle_area']
    n_speckle = data['speckle_positions']
    n_total = data['total_positions']

    # Per-height speckle stats
    height_lines = []
    for h in axis_values['height']:
        mask = np.isclose(data['heights'], h)
        h_sp = sp[mask]
        if len(h_sp) > 0:
            height_lines.append(
                f"    {h}mm: mean={np.mean(h_sp):.4f}%, max={np.max(h_sp):.4f}%, "
                f"non-zero={np.sum(h_sp > 0)}/{len(h_sp)}"
            )

    # Best/worst angular positions
    worst_idx = np.argmax(sp)
    best_nonzero = sp[sp > 0]
    best_idx = np.argmin(sp) if np.min(sp) > 0 else None

    duration = sweep_results.get('duration_seconds', 0)
    duration_min = duration / 60

    text = (
        f"SPECKLE ANALYSIS SUMMARY\n"
        f"{'='*50}\n\n"
        f"Tag ID:          {tag_id}\n"
        f"Sweep Duration:  {duration_min:.1f} minutes\n"
        f"Total Positions: {n_total}\n"
        f"With Speckle:    {n_speckle}\n\n"
        f"SPECKLE AREA\n"
        f"{'-'*50}\n"
        f"  Mean:       {np.mean(sp):.4f}%\n"
        f"  Std:        {np.std(sp):.4f}%\n"
        f"  Median:     {np.median(sp):.4f}%\n"
        f"  Max:        {np.max(sp):.4f}%\n"
        f"  Non-zero:   {np.sum(sp > 0)}/{len(sp)}\n\n"
        f"  Per Height:\n"
        + '\n'.join(height_lines) +
        f"\n\n"
        f"  Worst: {sp[worst_idx]:.4f}% at H={data['heights'][worst_idx]}mm, "
        f"P={data['pitches'][worst_idx]:.0f}, R={data['rolls'][worst_idx]:.0f}\n\n"
        f"SWEEP CONFIGURATION\n"
        f"{'-'*50}\n"
        f"  Heights:  {list(axis_values['height'])} mm\n"
        f"  Yaw:      {list(axis_values['yaw'])} deg\n"
        f"  Pitch:    {list(axis_values['pitch'])} deg\n"
        f"  Roll:     {list(axis_values['roll'])} deg\n"
        f"  Settle:   {sweep_config.get('settle_time_seconds', '?')}s\n"
    )

    ax.text(0.05, 0.95, text, transform=ax.transAxes,
            fontsize=11, fontfamily='monospace',
            verticalalignment='top', horizontalalignment='left',
            bbox=dict(boxstyle='round,pad=0.8', facecolor='#f5f5f5',
                     edgecolor='#333333', linewidth=1.5))

    fig.suptitle(f'Summary Report - Tag: {tag_id}',
                 fontsize=16, fontweight='bold', y=0.98)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    save_path = OUTPUT_DIR / f'{tag_id}_summary.png'
    plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  Saved: {save_path.name}")


# ============================================================
# PLOT 5: MULTI-DATASET COMPARISON
# ============================================================

def plot_comparison(datasets):
    """Compare speckle across multiple datasets."""
    n = len(datasets)
    tag_ids = [ds['tag_id'] for ds in datasets]

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle(f'Speckle Comparison: {" vs ".join(tag_ids)}',
                 fontsize=16, fontweight='bold')

    colors = plt.cm.Set2(np.linspace(0, 1, n))

    # Panel 1: Mean speckle per height
    all_heights_set = set()
    for ds in datasets:
        all_heights_set.update(ds['data']['heights'])
    all_heights = np.array(sorted(all_heights_set))
    x = np.arange(len(all_heights))
    width = 0.8 / n

    for k, ds in enumerate(datasets):
        means = []
        for h in all_heights:
            mask = np.isclose(ds['data']['heights'], h)
            sp = ds['data']['speckle_area'][mask]
            means.append(np.mean(sp) if len(sp) > 0 else np.nan)
        means = np.array(means)
        valid = ~np.isnan(means)
        bar_x = x[valid] + k * width - 0.4 + width / 2
        ax1.bar(bar_x, means[valid], width,
                color=colors[k], label=ds['tag_id'], edgecolor='black', linewidth=0.5)

    ax1.set_xticks(x)
    ax1.set_xticklabels([f'{int(h)}mm' for h in all_heights])
    ax1.set_ylabel('Mean Speckle Area (%)')
    ax1.set_title('Mean Speckle by Height')
    ax1.legend()
    ax1.set_ylim(bottom=0)

    # Panel 2: Speckle distribution box plots
    speckle_data = []
    labels = []
    for ds in datasets:
        speckle_data.append(ds['data']['speckle_area'])
        labels.append(ds['tag_id'])

    bp = ax2.boxplot(speckle_data, tick_labels=labels, patch_artist=True)
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
    ax2.set_ylabel('Speckle Area (%)')
    ax2.set_title('Speckle Distribution')

    # Panel 3: Speckle vs pitch per tag
    for k, ds in enumerate(datasets):
        pitch_vals = ds['axis_values']['pitch']
        means, stds = compute_speckle_by_param(ds['data'], 'pitches', pitch_vals)
        valid = ~np.isnan(means)
        if np.any(valid):
            ax3.errorbar(pitch_vals[valid], means[valid], yerr=stds[valid],
                        color=colors[k], marker='o', linewidth=2,
                        capsize=3, label=ds['tag_id'])

    ax3.set_xlabel('Pitch (deg)')
    ax3.set_ylabel('Mean Speckle Area (%)')
    ax3.set_title('Speckle vs Pitch')
    ax3.legend()
    ax3.set_ylim(bottom=0)
    ax3.grid(True, alpha=0.3)

    plt.tight_layout(rect=[0, 0, 1, 0.92])
    tag_str = '_vs_'.join(tag_ids)
    save_path = OUTPUT_DIR / f'comparison_{tag_str}.png'
    plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  Saved: {save_path.name}")


# ============================================================
# CONSOLE SUMMARY
# ============================================================

def print_summary(ds):
    """Print text summary to console."""
    data = ds['data']
    tag_id = ds['tag_id']
    axis_values = ds['axis_values']
    sp = data['speckle_area']

    print(f"\n{'='*60}")
    print(f"  Tag: {tag_id}")
    print(f"{'='*60}")
    print(f"  Positions: {data['total_positions']} total, {data['speckle_positions']} with speckle")

    # Per-height speckle
    print(f"\n  Speckle Area per Height:")
    for h in axis_values['height']:
        mask = np.isclose(data['heights'], h)
        h_sp = sp[mask]
        if len(h_sp) > 0:
            print(f"    {h}mm: mean={np.mean(h_sp):.4f}%, max={np.max(h_sp):.4f}%, "
                  f"non-zero={np.sum(h_sp > 0)}/{len(h_sp)}")

    print(f"\n  Overall: mean={np.mean(sp):.4f}%, std={np.std(sp):.4f}%, max={np.max(sp):.4f}%")
    print(f"  Non-zero: {np.sum(sp > 0)}/{len(sp)}")


# ============================================================
# MAIN
# ============================================================

if __name__ == '__main__':
    # Parse arguments
    if len(sys.argv) > 1:
        json_paths = [Path(p) for p in sys.argv[1:]]
    else:
        json_paths = sorted(SCRIPT_DIR.glob('*_*_*.json'))

    if not json_paths:
        print("No sweep JSON files found.")
        print("Place JSON files in the script directory or pass them as arguments.")
        sys.exit(1)

    OUTPUT_DIR.mkdir(exist_ok=True, parents=True)

    # Load all datasets
    datasets = []
    for jp in json_paths:
        print(f"Loading: {jp.name}")
        ds = load_sweep(jp)
        datasets.append(ds)

    # Single-dataset analysis
    for ds in datasets:
        tag_id = ds['tag_id']
        data = ds['data']
        axis_values = ds['axis_values']

        print_summary(ds)

        print(f"\n  Generating plots...")
        plot_speckle_heatmap_grid(data, axis_values, tag_id)
        plot_marginal_effects(data, axis_values, tag_id)
        plot_angular_sensitivity(data, axis_values, tag_id)
        plot_summary_statistics(data, axis_values, ds['sweep_config'],
                                ds['sweep_results'], tag_id)

    # Multi-dataset comparison
    if len(datasets) > 1:
        print(f"\n{'='*60}")
        print("Generating comparison plots...")
        print(f"{'='*60}")
        plot_comparison(datasets)

    print(f"\nAll figures saved to: {OUTPUT_DIR}")
    print("Done!")
