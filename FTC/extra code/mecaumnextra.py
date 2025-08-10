import numpy as np
import matplotlib.pyplot as plt

def mecanum_speed(theta_deg):
    """Closed-form approximation of mecanum speed profile."""
    theta = np.radians(theta_deg)
    friction_loss = 0.1 * abs(np.sin(theta))
    mecanum_efficiency = 0.7 + 0.3 * abs(np.cos(theta))
    return 1.88 * (1 - friction_loss) * mecanum_efficiency

# Generate data
angles = np.linspace(-180, 180, 361)
speeds = [mecanum_speed(angle) for angle in angles]

# Plot
plt.figure(figsize=(10, 5))
plt.plot(angles, speeds)
plt.title("Mecanum Speed Profile Function")
plt.xlabel("Movement Angle (degrees)")
plt.ylabel("Speed (m/s)")
plt.xticks(np.arange(-180, 181, 45))
plt.grid(True)
plt.show()