import numpy as np
import matplotlib.pyplot as plt
from heapq import heappop, heappush
from scipy.ndimage import binary_dilation
from matplotlib.patches import Rectangle, Polygon, Circle
import time
import math

# ==============================
# CONSTANTS AND CONFIGURATION
# ==============================
fieldSize = 3.6666667  # meters
robotWidth = 0.4  # meters
robotHeight = 0.4  # meters
clearance = 0.01  # meters (1cm)
wheelDiameter = 0.08  # meters (80mm)
wheelRadius = wheelDiameter / 2
maxRPS = 2.5  # Max motor RPS
maxAngVel = maxRPS * 2 * np.pi  # rad/s

# Kinematic parameters (distances from robot center to wheels)
lx = robotWidth / 2
ly = robotHeight / 2

# Time settings
dt = 0.05  # seconds (50ms intervals)

# Grid configuration
gridResolution = 0.005  # meters (5mm)
gridSize = int(fieldSize / gridResolution) + 1

# Start and goal positions (bottom-left to top-right corner)
startPos = (0.2, 0.2)  # Center of robot at start
goalPos = (fieldSize - 0.2, fieldSize - 0.2)  # Center of robot at goal

# Define obstacles as rectangles [min_x, min_y, max_x, max_y]
obstacles = [
    [1.0, 1.0, 1.5, 2.0],
    [2.0, 0.5, 2.5, 1.5],
    [0.5, 2.5, 1.5, 3.0],
    [2.0, 2.0, 3.0, 2.5]
]

# ==============================
# PATH PLANNING WITH THETA*
# ==============================
def preprocess_obstacles():
    """Create obstacle grid with robot clearance and expansion"""
    # Calculate expansion radius (robot circumradius + clearance)
    expansionRadius = (np.sqrt(robotWidth**2 + robotHeight**2) / 2 + clearance)
    expansionCells = int(np.ceil(expansionRadius / gridResolution))
    
    # Initialize grid
    grid = np.ones((gridSize, gridSize), dtype=bool)
    
    # Mark obstacles in grid
    for obs in obstacles:
        min_i = max(0, int(obs[0] / gridResolution))
        min_j = max(0, int(obs[1] / gridResolution))
        max_i = min(gridSize, int(obs[2] / gridResolution) + 1)
        max_j = min(gridSize, int(obs[3] / gridResolution) + 1)
        grid[min_i:max_i, min_j:max_j] = False
    
    # Create circular kernel for dilation
    y, x = np.ogrid[-expansionCells:expansionCells+1, 
                    -expansionCells:expansionCells+1]
    kernel = x**2 + y**2 <= expansionCells**2
    
    # Expand obstacles
    expandedGrid = binary_dilation(~grid, structure=kernel)
    return expandedGrid

def heuristic(a, b):
    """Euclidean distance heuristic for A*"""
    return np.sqrt((a[0]-b[0])**2 + (a[1]-b[1])**2)

def line_of_sight(grid, start, end):
    """Check if straight line path is obstacle-free using Bresenham"""
    x0, y0 = start
    x1, y1 = end
    dx = abs(x1 - x0)
    dy = abs(y1 - y0)
    sx = 1 if x0 < x1 else -1
    sy = 1 if y0 < y1 else -1
    err = dx - dy
    
    while (x0, y0) != (x1, y1):
        if not grid[x0, y0]:
            return False
            
        e2 = 2 * err
        if e2 > -dy:
            err -= dy
            x0 += sx
        if e2 < dx:
            err += dx
            y0 += sy
            
    return True

def theta_star_search(grid, start, goal):
    """Theta* pathfinding algorithm"""
    # Convert positions to grid coordinates
    start_grid = (int(start[0]/gridResolution), int(start[1]/gridResolution))
    goal_grid = (int(goal[0]/gridResolution), int(goal[1]/gridResolution))
    
    # Initialize data structures
    openSet = []
    closedSet = set()
    gScore = np.full((gridSize, gridSize), np.inf)
    fScore = np.full((gridSize, gridSize), np.inf)
    parent = np.full((gridSize, gridSize, 2), -1, dtype=int)
    
    # Initialize starting node
    gScore[start_grid] = 0
    fScore[start_grid] = heuristic(start_grid, goal_grid)
    heappush(openSet, (fScore[start_grid], start_grid))
    parent[start_grid] = start_grid
    
    # Define 8-connected neighborhood
    neighbors = [(dx, dy) for dx in (-1,0,1) for dy in (-1,0,1) if (dx,dy) != (0,0)]
    
    while openSet:
        _, current = heappop(openSet)
        
        if current == goal_grid:
            break
            
        closedSet.add(current)
        
        for dx, dy in neighbors:
            neighbor = (current[0] + dx, current[1] + dy)
            
            # Check bounds and obstacles
            if (neighbor[0] < 0 or neighbor[0] >= gridSize or 
                neighbor[1] < 0 or neighbor[1] >= gridSize):
                continue
                
            if not grid[neighbor] or neighbor in closedSet:
                continue
                
            # Check line of sight to parent
            par = tuple(parent[current])
            if line_of_sight(grid, par, neighbor):
                # Path 2: Direct from parent
                new_g = gScore[par] + heuristic(par, neighbor)
                new_parent = par
            else:
                # Path 1: Through current node
                new_g = gScore[current] + heuristic(current, neighbor)
                new_parent = current
                
            # Update if better path found
            if new_g < gScore[neighbor]:
                gScore[neighbor] = new_g
                fScore[neighbor] = new_g + heuristic(neighbor, goal_grid)
                parent[neighbor] = new_parent
                heappush(openSet, (fScore[neighbor], neighbor))
    
    # Reconstruct path
    path = []
    current = goal_grid
    while current != start_grid:
        path.append((current[0]*gridResolution, current[1]*gridResolution))
        current = tuple(parent[current])
    path.append(start)
    return path[::-1]  # Return reversed path

# ==============================
# PATH PROCESSING AND KINEMATICS
# ==============================
def smooth_path(path):
    """Simplify path while maintaining obstacle clearance"""
    if len(path) < 3:
        return path
        
    smoothed = [path[0]]
    i = 0
    
    while i < len(path)-1:
        j = len(path) - 1
        while j > i+1:
            if line_of_sight(obstacleGrid, 
                            (int(path[i][0]/gridResolution), int(path[i][1]/gridResolution)),
                            (int(path[j][0]/gridResolution), int(path[j][1]/gridResolution))):
                smoothed.append(path[j])
                i = j
                break
            j -= 1
        else:
            smoothed.append(path[i+1])
            i += 1
            
    return smoothed

def calculate_max_velocity(direction):
    """Calculate max velocity in given direction considering wheel constraints"""
    # Normalize direction vector
    magnitude = np.hypot(direction[0], direction[1])
    if magnitude < 1e-5:
        return 0, (0, 0)
        
    ux = direction[0] / magnitude
    uy = direction[1] / magnitude
    
    # Calculate constraints
    s1 = maxAngVel * wheelRadius / abs(ux - uy) if abs(ux - uy) > 1e-5 else np.inf
    s2 = maxAngVel * wheelRadius / abs(ux + uy) if abs(ux + uy) > 1e-5 else np.inf
    s_max = min(s1, s2)
    
    return s_max, (s_max * ux, s_max * uy)

def generate_trajectory(path):
    """Generate velocity profile and wheel commands for path"""
    if len(path) < 2:
        return [], []
        
    segments = []
    wheel_commands = []
    
    # Process each path segment
    for i in range(len(path)-1):
        start = path[i]
        end = path[i+1]
        direction = (end[0]-start[0], end[1]-start[1])
        length = np.hypot(direction[0], direction[1])
        
        # Calculate optimal velocity
        s_max, velocity = calculate_max_velocity(direction)
        if s_max < 1e-5:
            continue
            
        # Segment time and steps
        seg_time = length / s_max
        steps = max(1, int(np.ceil(seg_time / dt)))
        actual_vel = (velocity[0]*dt*steps/length, velocity[1]*dt*steps/length)
        
        # Calculate wheel velocities
        vx, vy = actual_vel
        w_fl = (vx - vy - (lx + ly)*0) / wheelRadius
        w_fr = (vx + vy + (lx + ly)*0) / wheelRadius
        w_rl = (vx + vy - (lx + ly)*0) / wheelRadius
        w_rr = (vx - vy + (lx + ly)*0) / wheelRadius
        
        # Scale if exceeds max
        max_w = max(abs(w_fl), abs(w_fr), abs(w_rl), abs(w_rr))
        if max_w > maxAngVel:
            scale = maxAngVel / max_w
            w_fl *= scale
            w_fr *= scale
            w_rl *= scale
            w_rr *= scale
            
        # Add commands for each time step
        for _ in range(steps):
            segments.append((start, end))
            wheel_commands.append((w_fl, w_fr, w_rl, w_rr))
    
    return segments, wheel_commands

# ==============================
# VISUALIZATION
# ==============================
def visualize_path(path, segments, wheel_commands):
    """Create visualization of path and robot motion"""
    fig, ax = plt.subplots(figsize=(10, 10))
    
    # Draw field
    ax.set_xlim(0, fieldSize)
    ax.set_ylim(0, fieldSize)
    ax.add_patch(Rectangle((0, 0), fieldSize, fieldSize, 
                         fill=False, edgecolor='black', linewidth=2))
    
    # Draw obstacles
    for obs in obstacles:
        ax.add_patch(Rectangle((obs[0], obs[1]), obs[2]-obs[0], obs[3]-obs[1],
                             fill=True, color='gray', alpha=0.7))
    
    # Draw path
    path_x = [p[0] for p in path]
    path_y = [p[1] for p in path]
    ax.plot(path_x, path_y, 'b-', linewidth=1.5, label='Planned Path')
    ax.plot(path_x, path_y, 'ro', markersize=2, alpha=0.5)
    
    # Draw robot at key positions
    for i in range(0, len(path), max(1, len(path)//5)):
        x, y = path[i]
        draw_robot(ax, x, y, 0)
    
    # Draw start and goal
    ax.plot(startPos[0], startPos[1], 'go', markersize=10, label='Start')
    ax.plot(goalPos[0], goalPos[1], 'yo', markersize=10, label='Goal')
    
    # Add labels and legend
    ax.set_title('Robot Path Planning')
    ax.set_xlabel('X Position (m)')
    ax.set_ylabel('Y Position (m)')
    ax.grid(True)
    ax.legend()
    ax.set_aspect('equal')
    
    # Save visualization
    plt.savefig('robot_path_visualization.png', dpi=300)
    plt.close()
    
    # Create wheel command visualization
    if wheel_commands:
        times = [i*dt for i in range(len(wheel_commands))]
        w_fl = [cmd[0] for cmd in wheel_commands]
        w_fr = [cmd[1] for cmd in wheel_commands]
        w_rl = [cmd[2] for cmd in wheel_commands]
        w_rr = [cmd[3] for cmd in wheel_commands]
        
        plt.figure(figsize=(12, 8))
        plt.plot(times, w_fl, label='Front Left')
        plt.plot(times, w_fr, label='Front Right')
        plt.plot(times, w_rl, label='Rear Left')
        plt.plot(times, w_rr, label='Rear Right')
        
        plt.title('Wheel Angular Velocities')
        plt.xlabel('Time (s)')
        plt.ylabel('Angular Velocity (rad/s)')
        plt.legend()
        plt.grid(True)
        plt.savefig('wheel_velocities.png', dpi=200)
        plt.close()

def draw_robot(ax, x, y, theta):
    """Draw robot representation at given position and orientation"""
    # Create robot body
    robot = Rectangle((x - robotWidth/2, y - robotHeight/2), 
                     robotWidth, robotHeight, 
                     fill=False, edgecolor='blue', linewidth=1.5)
    ax.add_patch(robot)
    
    # Draw front indicator
    front_x = x + np.cos(theta) * robotWidth/2
    front_y = y + np.sin(theta) * robotHeight/2
    ax.plot([x, front_x], [y, front_y], 'r-', linewidth=2)
    
    # Draw wheels
    wheel_positions = [
        (x - lx, y + ly),  # Front left
        (x + lx, y + ly),  # Front right
        (x - lx, y - ly),  # Rear left
        (x + lx, y - ly)   # Rear right
    ]
    
    for wx, wy in wheel_positions:
        wheel = Circle((wx, wy), wheelDiameter/4, 
                      fill=True, color='green', alpha=0.6)
        ax.add_patch(wheel)

# ==============================
# MAIN EXECUTION
# ==============================
if __name__ == "__main__":
    # Precompute obstacle grid
    obstacleGrid = preprocess_obstacles()
    
    # Plan path
    print("Planning path with Theta*...")
    start_time = time.time()
    path = theta_star_search(obstacleGrid, startPos, goalPos)
    planning_time = time.time() - start_time
    print(f"Path planning completed in {planning_time:.2f} seconds")
    print(f"Path length: {len(path)} points")
    
    # Simplify path
    if path:
        path = smooth_path(path)
        print(f"Simplified path: {len(path)} points")
    
    # Generate trajectory and wheel commands
    segments, wheel_commands = generate_trajectory(path)
    total_time = len(wheel_commands) * dt
    print(f"Total motion time: {total_time:.2f} seconds")
    
    # Output wheel commands
    print("\nTime Interval | FR (rad/s) | FL (rad/s) | BR (rad/s) | BL (rad/s)")
    for i, (w_fl, w_fr, w_rl, w_rr) in enumerate(wheel_commands):
        start_t = i * dt
        end_t = (i+1) * dt
        print(f"{start_t:05.2f}-{end_t:05.2f} | "
              f"{w_fr:9.4f} | {w_fl:9.4f} | {w_rr:9.4f} | {w_rl:9.4f}")
    
    # Save commands to file
    with open('wheel_commands.csv', 'w') as f:
        f.write("TimeStart,TimeEnd,FR,FL,BR,BL\n")
        for i, (w_fl, w_fr, w_rl, w_rr) in enumerate(wheel_commands):
            start_t = i * dt
            end_t = (i+1) * dt
            f.write(f"{start_t:.3f},{end_t:.3f},{w_fr},{w_fl},{w_rr},{w_rl}\n")
    
    # Generate visualizations
    if path:
        visualize_path(path, segments, wheel_commands)
        print("Visualizations saved to robot_path_visualization.png and wheel_velocities.png")
    else:
        print("No valid path found!")