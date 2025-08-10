import numpy as np
import matplotlib.pyplot as plt
import math

def mecanum_wheel_speeds(xDist, yDist, rotation_deg=0):
    """Compute wheel speeds (FL, FR, BL, BR) in RPS."""
    lx, ly = 0.3, 0.4  # Robot dimensions (meters)
    wheel_radius = 0.03  # 60mm diameter → 30mm radius (meters)
    omega_z = math.radians(rotation_deg)
    
    ik_matrix = np.array([
        [1, -1, -(lx + ly)],
        [1, 1, (lx + ly)],
        [1, 1, -(lx + ly)],
        [1, -1, (lx + ly)]
    ])
    
    wheel_speeds_rad = (1 / wheel_radius) * np.dot(ik_matrix, [xDist, yDist, omega_z])
    wheel_speeds_rps = wheel_speeds_rad / (2 * math.pi)
    return wheel_speeds_rps

# Parameters
max_rps = 2.0
angles = np.linspace(-np.pi, np.pi, 361)  # -180° to 180°
speeds = []
friction_losses = []

for theta in angles:
    v_x = np.cos(theta)
    v_y = np.sin(theta)
    
    # Compute friction (0% at 0°/180°, 10% at ±90°)
    friction = 0.1 * abs(np.sin(theta))
    effective_speed_scale = 1 - friction
    friction_losses.append(friction * 100)
    
    # Compute required wheel speeds without scaling
    fl, fr, bl, br = mecanum_wheel_speeds(v_x, v_y)
    max_wheel_speed = max(abs(fl), abs(fr), abs(bl), abs(br))
    
    # The true limiting factor is the wheel speed constraint (2 RPS)
    if max_wheel_speed > max_rps:
        # Scale entire movement to keep wheel speeds ≤ 2 RPS
        scale_factor = max_rps / max_wheel_speed
        v_x_scaled = v_x * scale_factor
        v_y_scaled = v_y * scale_factor
        # Recompute with scaled velocities
        fl, fr, bl, br = mecanum_wheel_speeds(v_x_scaled, v_y_scaled)
    else:
        v_x_scaled = v_x
        v_y_scaled = v_y
    
    # Now apply friction to the achievable speed
    v_x_final = v_x_scaled * effective_speed_scale
    v_y_final = v_y_scaled * effective_speed_scale
    
    # Final robot speed (magnitude)
    robot_speed = np.sqrt(v_x_final**2 + v_y_final**2)
    speeds.append(robot_speed)

# Convert angles to degrees
angles_deg = np.degrees(angles)

# Create figure
plt.figure(figsize=(14, 6))

# Speed vs. Angle
plt.subplot(1, 2, 1)
plt.plot(angles_deg, speeds, label="Achievable Speed", color='blue')
plt.title("Robot Speed vs. Movement Direction")
plt.xlabel("Movement Angle (degrees)")
plt.ylabel("Speed (m/s)")
plt.xticks(np.arange(-180, 181, 45))
plt.xlim(-180, 180)
plt.grid(True)

# Mark key points
plt.scatter(0, speeds[len(angles)//2], color='red', label="Forward (0°)")
plt.scatter(90, speeds[int(3*len(angles)/4)], color='green', label="Left (90°)")
plt.scatter(-90, speeds[int(len(angles)/4)], color='purple', label="Right (-90°)")
plt.legend()

# Friction Loss vs. Angle
plt.subplot(1, 2, 2)
plt.plot(angles_deg, friction_losses, color='red')
plt.title("Friction Loss vs. Movement Direction")
plt.xlabel("Movement Angle (degrees)")
plt.ylabel("Friction Loss (%)")
plt.xticks(np.arange(-180, 181, 45))
plt.xlim(-180, 180)
plt.grid(True)

plt.tight_layout()
plt.savefig("mecanum_corrected_plot.png")
plt.close()

print("Corrected graph saved as 'mecanum_corrected_plot.png'")

# Print theoretical values
print("\nTheoretical Speed Values:")
print(f"Forward (0°): {speeds[len(angles)//2]:.2f} m/s")
print(f"Left (90°): {speeds[int(3*len(angles)/4)]:.2f} m/s")
print(f"Right (-90°): {speeds[int(len(angles)/4)]:.2f} m/s")
print(f"Backward (180°): {speeds[-1]:.2f} m/s")