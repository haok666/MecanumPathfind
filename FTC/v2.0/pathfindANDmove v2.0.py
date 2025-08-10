"""
Mecanum Wheel Robot Path Planner
================================
This program plans the fastest path for a 40cm x 40cm mecanum-wheeled robot
through a 3.6666667m square field with rectangular obstacles. It uses:
- A* algorithm with time-based cost
- Proper mecanum wheel inverse kinematics
- 16-direction movement for optimal paths
- Visualization and wheel speed logging
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from heapq import heappop, heappush
import csv

# ======================
# PHYSICAL CONSTANTS
# ======================
FIELD_SIZE = 3.6666667  # meters (exact competition field size)
MARGIN_ERROR = 0.005    # 5mm maximum position error
ROBOT_SIZE = 0.4        # 40cm x 40cm robot dimensions
WHEEL_RADIUS = 0.04     # 80mm diameter wheels
MAX_RPS = 2.5           # Maximum motor rotational speed
MAX_LINEAR_SPEED = MAX_RPS * 2 * np.pi * WHEEL_RADIUS  # m/s
LX = LY = 0.2           # Distance from robot center to wheels (40cm/2)

# ======================
# OBSTACLE CONFIGURATION
# ======================
# Format: [center_x, center_y, width, height] for each rectangular obstacle
OBSTACLES = [
    [1.0, 1.0, 0.5, 0.5],   # Central obstacle
    [2.5, 2.0, 0.3, 0.7],   # Vertical rectangle
    [0.8, 2.5, 0.6, 0.4]    # Horizontal rectangle
]

# ======================
# CORE FUNCTIONS
# ======================

def is_collision(x, y):
    """
    Check if robot at (x,y) collides with any obstacle.
    Args:
        x, y: Robot center coordinates
    Returns:
        bool: True if collision detected
    """
    for (cx, cy, w, h) in OBSTACLES:
        if (abs(x - cx) < (w/2 + ROBOT_SIZE/2)) and (abs(y - cy) < (h/2 + ROBOT_SIZE/2)):
            return True
    return False

def mecanum_wheel_speeds(vx, vy, wz=0):
    """
    Compute wheel speeds using inverse kinematics matrix.
    Args:
        vx: Linear velocity in x-direction (forward) [m/s]
        vy: Linear velocity in y-direction (strafing) [m/s]
        wz: Angular velocity around z-axis [rad/s]
    Returns:
        tuple: (w_fl, w_fr, w_rl, w_rr) wheel speeds in RPS
    """
    # Inverse kinematics matrix (from theory)
    inv_kin_matrix = np.array([
        [1, -1, -(LX + LY)],  # Front left wheel
        [1,  1,  (LX + LY)],  # Front right wheel
        [1,  1, -(LX + LY)],  # Rear left wheel
        [1, -1,  (LX + LY)]   # Rear right wheel
    ]) / WHEEL_RADIUS
    
    # Matrix multiplication to get wheel speeds
    wheel_speeds = inv_kin_matrix @ np.array([vx, vy, wz])
    return tuple(wheel_speeds)

def normalize_speeds(w_fl, w_fr, w_rl, w_rr):
    """
    Scale wheel speeds to stay within motor limits.
    Args:
        w_fl, w_fr, w_rl, w_rr: Raw wheel speeds (RPS)
    Returns:
        tuple: Normalized wheel speeds
    """
    max_speed = max(abs(w_fl), abs(w_fr), abs(w_rl), abs(w_rr))
    if max_speed > MAX_RPS:
        scale = MAX_RPS / max_speed
        return (w_fl*scale, w_fr*scale, w_rl*scale, w_rr*scale)
    return (w_fl, w_fr, w_rl, w_rr)

# ======================
# PATHFINDING ALGORITHM
# ======================

def a_star(start, goal):
    """
    A* pathfinding algorithm with time-based optimization.
    Args:
        start: (x,y) starting position
        goal: (x,y) target position
    Returns:
        list: Path as sequence of (x,y) coordinates
    """
    open_set = []
    heappush(open_set, (0, start))
    came_from = {}
    g_score = {start: 0}
    f_score = {start: heuristic(start, goal)}
    
    while open_set:
        _, current = heappop(open_set)
        
        # Check if reached goal (within 10cm tolerance)
        if np.linalg.norm(np.array(current) - np.array(goal)) < 0.1:
            path = [current]
            while current in came_from:
                current = came_from[current]
                path.append(current)
            return path[::-1]  # Return reversed path
        
        for neighbor, time_cost in get_neighbors(current):
            tentative_g = g_score[current] + time_cost
            if neighbor not in g_score or tentative_g < g_score[neighbor]:
                came_from[neighbor] = current
                g_score[neighbor] = tentative_g
                f_score[neighbor] = tentative_g + heuristic(neighbor, goal)
                heappush(open_set, (f_score[neighbor], neighbor))
    
    return None  # No path found

def heuristic(a, b):
    """
    Time-based heuristic for A* algorithm.
    Args:
        a, b: (x,y) positions
    Returns:
        float: Minimum estimated time between points
    """
    dx, dy = b[0] - a[0], b[1] - a[1]
    return np.sqrt(dx**2 + dy**2) / MAX_LINEAR_SPEED

def get_neighbors(node):
    """
    Generate reachable neighboring positions with time costs.
    Args:
        node: Current (x,y) position
    Returns:
        list: [(neighbor, time_cost)] pairs
    """
    x, y = node
    neighbors = []
    # Check 16 directions (22.5° increments) for smooth movement
    for angle in np.linspace(0, 2*np.pi, 16, endpoint=False):
        dx, dy = 0.1 * np.cos(angle), 0.1 * np.sin(angle)  # 10cm step
        nx, ny = x + dx, y + dy
        if (0 <= nx <= FIELD_SIZE and 0 <= ny <= FIELD_SIZE 
            and not is_collision(nx, ny)):
            # Time cost = distance / max_speed (optimistic)
            neighbors.append(((nx, ny), 0.1 / MAX_LINEAR_SPEED))
    return neighbors

# ======================
# VISUALIZATION & LOGGING
# ======================

def visualize_path(path):
    """
    Generate and save path visualization image.
    Args:
        path: List of (x,y) coordinates
    """
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.set_xlim(0, FIELD_SIZE)
    ax.set_ylim(0, FIELD_SIZE)
    ax.set_aspect('equal')
    ax.set_title('Robot Path Planning')
    
    # Draw obstacles
    for (cx, cy, w, h) in OBSTACLES:
        ax.add_patch(patches.Rectangle(
            (cx - w/2, cy - h/2), w, h, 
            color='red', alpha=0.5, label='Obstacle'
        ))
    
    # Draw path
    if path:
        xs, ys = zip(*path)
        ax.plot(xs, ys, 'b-', linewidth=2, label='Planned Path')
        ax.plot(xs[0], ys[0], 'go', markersize=10, label='Start')  # Start
        ax.plot(xs[-1], ys[-1], 'yo', markersize=10, label='Goal')  # Goal
    
    ax.legend()
    plt.savefig('path_visualization.png', dpi=300)
    plt.close()

def log_movement(path):
    """
    Generate wheel speed log in CSV format.
    Args:
        path: List of (x,y) coordinates
    """
    with open('wheel_speeds.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([
            'Time (s)', 'FL (RPS)', 'FR (RPS)', 
            'RL (RPS)', 'RR (RPS)'
        ])
        
        time_step = 0.1  # 100ms intervals
        for i in range(len(path) - 1):
            # Calculate required velocity between points
            x1, y1 = path[i]
            x2, y2 = path[i+1]
            vx = (x2 - x1) / time_step
            vy = (y2 - y1) / time_step
            
            # Compute and normalize wheel speeds
            w_fl, w_fr, w_rl, w_rr = normalize_speeds(
                *mecanum_wheel_speeds(vx, vy, 0)  # wz=0 for straight movement
            )
            
            writer.writerow([
                f"{i*time_step:.2f}",
                f"{w_fl:.2f}", f"{w_fr:.2f}",
                f"{w_rl:.2f}", f"{w_rr:.2f}"
            ])

# ======================
# MAIN EXECUTION
# ======================

if __name__ == "__main__":
    print("Mecanum Robot Path Planner")
    print("=========================")
    
    # Define start and goal positions (avoid edges)
    START = (0.1, 0.1)
    GOAL = (3.5, 3.5)
    
    # Run pathfinding
    print("Planning path...")
    path = a_star(START, GOAL)
    
    if path:
        print(f"Path found with {len(path)} waypoints!")
        print("Generating visualization...")
        visualize_path(path)
        print("Generating wheel speed logs...")
        log_movement(path)
        print("Results saved to:")
        print("- path_visualization.png")
        print("- wheel_speeds.csv")
    else:
        print("Error: No valid path found!")