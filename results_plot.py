import matplotlib.pyplot as plt
import numpy as np

# --- IEEE Paper Style Configuration ---
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman'],  # Force Times New Roman for IEEE
    'font.size': 12,                    # Readable font size
    'axes.labelsize': 12,
    'legend.fontsize': 11,
    'xtick.labelsize': 11,
    'ytick.labelsize': 11,
    'axes.grid': True,
    'grid.alpha': 0.3,
    'grid.linestyle': '--',             # Dashed grid is less intrusive
    'axes.axisbelow': True,
    'figure.autolayout': True           # Auto-adjusts margins perfectly
})


def plot_hardware_grouped():
    # --- Data Setup ---
    # Models
    models = ['Green-Mamba', 'MobileNetV3', 'ResNet18']

    # Data values (ResNet18 values used here)
    # Metric 1: FPS (Speed) - Higher is better
    fps = [394.03, 116.02, 42.50]

    # Metric 2: Memory (MB) - Lower is better
    memory = [1.58, 5.80, 44.80]

    # Metric 3: Energy (Joules) - Lower is better
    energy = [0.0127, 0.0431, 0.1250]

    # Organize data for plotting
    # Rows: Green-Mamba, MobileNet, ResNet18
    # Cols: FPS, Memory, Energy
    metrics_labels = ['Inference Speed\n(FPS) $\\uparrow$',
                      'Memory Usage\n(MB) $\\downarrow$',
                      'Energy/Inference\n(Joules) $\\downarrow$']

    # Create groups for plotting
    data_gm = [fps[0], memory[0], energy[0]]
    data_mn = [fps[1], memory[1], energy[1]]
    data_rn = [fps[2], memory[2], energy[2]]

    x = np.arange(len(metrics_labels))
    width = 0.25

    # --- Plotting ---
    fig, ax = plt.subplots(figsize=(8, 5))  # Standard width for papers

    # Define Bars with Hatching for B&W print compatibility
    # Hatching patterns: '/' = diagonal, '.' = dots, 'x' = cross
    rects1 = ax.bar(x - width, data_gm, width, label='Green-Mamba (Ours)',
                    color='#2ca02c', edgecolor='black', alpha=0.9, hatch='//')

    rects2 = ax.bar(x, data_mn, width, label='MobileNetV3',
                    color='#1f77b4', edgecolor='black', alpha=0.9, hatch='..')

    rects3 = ax.bar(x + width, data_rn, width, label='ResNet18',
                    color='#ff7f0e', edgecolor='black', alpha=0.9, hatch='xx')

    # Log scale to handle the wide range of values (0.01 to 400)
    ax.set_yscale('log')

    # Labels and Ticks
    ax.set_ylabel('Metric Value (Log Scale)', fontweight='bold')
    ax.set_title('Hardware Efficiency Benchmark (Raspberry Pi 4)',
                 pad=15, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(metrics_labels)

    # Refined Legend
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15),
              ncol=3, frameon=False, columnspacing=1.5)

    # --- Value Annotation ---
    def autolabel(rects):
        for rect in rects:
            height = rect.get_height()
            # Conditional formatting: 3 decimals for small numbers, Int for large
            if height < 1:
                val_str = f'{height:.3f}'
            else:
                val_str = f'{int(height)}'

            ax.annotate(val_str,
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 4),  # Offset slightly higher
                        textcoords="offset points",
                        ha='center', va='bottom',
                        fontsize=10, fontweight='bold',
                        color='black')  # Ensure text is black

    autolabel(rects1)
    autolabel(rects2)
    autolabel(rects3)

    # --- Saving ---
    # Save as PDF for the paper (Vector format - Infinite Resolution)
    plt.savefig('benchmark_resnet18_paper.pdf',
                format='pdf', bbox_inches='tight')

    # Save as High-Res PNG for previews/presentations
    plt.savefig('benchmark_resnet18_paper.png', dpi=600, bbox_inches='tight')

    print("Figures saved as 'benchmark_resnet18_paper.pdf' and 'benchmark_resnet18_paper.png'")
    plt.show()


if __name__ == "__main__":
    plot_hardware_grouped()
