import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os

def generate_dashboard():
    figures = {
        "Ising (Magnetic)": "simulations/phase_transitions/ising_results.png",
        "Potts (Symmetry)": "simulations/phase_transitions/potts_results.png",
        "Percolation (Geometric)": "simulations/phase_transitions/percolation_results.png",
        "Brain (Cognitive)": "figures/critical_brain_size_distribution.png"
    }

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    plt.suptitle("Universality Across Systems", fontsize=20)

    for i, (name, path) in enumerate(figures.items()):
        ax = axes[i//2, i%2]
        if os.path.exists(path):
            img = mpimg.imread(path)
            ax.imshow(img)
            ax.set_title(name)
        else:
            ax.text(0.5, 0.5, f"File Not Found:\n{path}", ha='center', va='center')
        ax.axis('off')

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig("research/observations/universality_comparison.png")
    print("Dashboard saved to research/observations/universality_comparison.png")

if __name__ == "__main__":
    generate_dashboard()