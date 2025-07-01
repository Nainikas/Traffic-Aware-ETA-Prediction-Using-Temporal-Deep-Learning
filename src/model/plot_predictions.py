import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Load saved predictions
data = np.load("data/predictions.npz")
y_true = data["targets"]
y_pred = data["predictions"]

# Create output folder if it doesn't exist
output_dir = Path("outputs")
output_dir.mkdir(exist_ok=True)

# Plot
plt.figure(figsize=(10, 5))
plt.plot(y_true[:200], label="True ETA", linewidth=2)
plt.plot(y_pred[:200], label="Predicted ETA", linestyle="--")
plt.title("Predicted ETA vs True ETA (First 200 Samples)")
plt.xlabel("Sample")
plt.ylabel("ETA (minutes)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(output_dir / "eta_vs_truth.png")
print("Plot saved to outputs/eta_vs_truth.png")
