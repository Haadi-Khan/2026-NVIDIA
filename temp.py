import marimo

__generated_with = "0.19.7"
app = marimo.App()


@app.cell
def _():
    import pandas as pd
    import matplotlib.pyplot as plt
    import numpy as np
    return np, pd, plt


@app.cell
def _(np, pd, plt):
    df_graph = pd.read_csv('new_results.csv')
    df_graph['N'] = df_graph['N'].astype(int)
    x_vals = sorted(df_graph['N'].unique())

    # Clean style
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(10, 6))

    # Define styling for each column
    style_config = {
        'MF_Exhaustive': {
            'label': 'Benchmark (Exhaustive)',
            'color': '#2563eb',  # nicer blue
            'plot_type': 'line',
        },
        'MF_MTS': {
            'label': 'MTS (median ± min/max)',
            'color': '#ea580c',  # orange
            'plot_type': 'scatter_errorbar',
        },
        'MF_PCE': {
            'label': 'PCE (median ± min/max)',
            'color': '#16a34a',  # green
            'plot_type': 'scatter_errorbar',
        },
    }

    columns = [col for col in df_graph.columns if col != "N"]

    for col in columns:
        if col not in style_config:
            continue

        medians, min_vals, max_vals = [], [], []
        for n in x_vals:
            y_data = df_graph.loc[df_graph['N'] == n, col].astype(float)
            y_data = y_data[~np.isnan(y_data)]
            if len(y_data) == 0:
                medians.append(np.nan)
                min_vals.append(np.nan)
                max_vals.append(np.nan)
            else:
                medians.append(np.median(y_data))
                min_vals.append(np.min(y_data))
                max_vals.append(np.max(y_data))

        medians_np = np.array(medians)
        min_vals_np = np.array(min_vals)
        max_vals_np = np.array(max_vals)

        if np.all(np.isnan(medians_np)):
            continue

        cfg = style_config[col]

        if cfg['plot_type'] == 'line':
            ax.plot(x_vals, medians_np, 
                    color=cfg['color'], 
                    linewidth=2.2, 
                    label=cfg['label'],
                    zorder=2)
        else:
            yerr = [medians_np - min_vals_np, max_vals_np - medians_np]
            ax.errorbar(x_vals, medians_np, yerr=yerr,
                        fmt='o',
                        color=cfg['color'],
                        markerfacecolor='white',
                        markeredgecolor=cfg['color'],
                        markeredgewidth=1.8,
                        markersize=8,
                        ecolor=cfg['color'],
                        elinewidth=1.5,
                        capsize=4,
                        capthick=1.5,
                        label=cfg['label'],
                        zorder=3)

    ax.axvline(x=27, color='#7c3aed', linestyle='--', linewidth=1.5,
               alpha=0.7, label='Reference Paper Lower Limit', zorder=1)
    # Vertical reference line at N = 37
    ax.axvline(x=37, color='#7c3aed', linestyle='--', linewidth=1.5,
               alpha=0.7, label='Reference Paper Upper Limit', zorder=1)

    # Styling
    ax.set_xlabel("N", fontsize=13, fontweight='medium')
    ax.set_ylabel("Merit Factor (higher is better)", fontsize=13, fontweight='medium')
    ax.set_title("Merit Factor vs N", fontsize=15, fontweight='bold', pad=15)

    # Legend in top right corner
    ax.legend(loc='upper right', frameon=True, fancybox=True,
              framealpha=0.95, fontsize=10, title='Method', title_fontsize=11)

    ax.tick_params(axis='both', labelsize=11)
    ax.set_xlim(0, max(x_vals) + 2)

    # Text box in bottom right corner
    textstr = "Loss function calls:\n  MTS: 250k\n  PCE: 25k"
    props = dict(boxstyle='round,pad=0.4', facecolor='white', 
                 edgecolor='gray', alpha=0.9)
    ax.text(0.98, 0.05, textstr, transform=ax.transAxes, fontsize=9,
            verticalalignment='bottom', horizontalalignment='right', bbox=props)

    plt.tight_layout()
    plt.show()
    return


if __name__ == "__main__":
    app.run()
