import numpy as np
import matplotlib.pyplot as plt
from heapq import heappush, heappop
from math import sqrt, atan2, radians, degrees, pi
from scipy.signal import convolve2d
import time
from tqdm import tqdm
import os

# Constants
FIELD_SIZE = 3.6667  # meters
MAX_RPS = 4.0        # revolutions per second
ROBOT_LENGTH = 0.4 # meters
ROBOT_WIDTH = 0.4  # meters
RESOLUTION = 0.005   # meters per grid cell
WHEEL_RADIUS = 0.03  # meters
ROTATION_SPEED = 1 * pi  # rad/s

class RobotController:
    def __init__(self):
        self.command_history = []
        self.current_time = 0.0
    
    def move_motor(self, motor_name, rpm, duration):
        """Log motor movement command"""
        command = {
            'motor': motor_name,
            'rpm': rpm,
            'start': self.current_time,
            'end': self.current_time + duration
        }
        self.command_history.append(command)
        # In real implementation, send to physical motor here
        time.sleep(duration * 0.1)  # Simulate movement (10% speed for testing)

def calculate_mecanum_speed(theta_deg):
    """Calculate speed based on movement direction"""
    theta = radians(theta_deg)
    friction_loss = 0.1 * abs(np.sin(theta))
    efficiency = 0.7 + 0.3 * abs(np.cos(theta))
    return 3.77 * (1 - friction_loss) * efficiency  # Max 3.77 m/s at 4 RPS

def create_obstacle_map(obstacle_list):
    """Convert obstacle list to grid map with safety margins"""
    grid_size = int(FIELD_SIZE / RESOLUTION)
    obstacle_grid = np.zeros((grid_size, grid_size), dtype=bool)
    
    for x, y, side in obstacle_list:
        min_x = max(0, int((x-side/2)/RESOLUTION))
        max_x = min(grid_size-1, int((x+side/2)/RESOLUTION))
        min_y = max(0, int((y-side/2)/RESOLUTION))
        max_y = min(grid_size-1, int((y+side/2)/RESOLUTION))
        obstacle_grid[min_x:max_x+1, min_y:max_y+1] = True
    
    # Add safety margin around obstacles
    clearance = int(0.297 / RESOLUTION)  # 29.7cm diagonal clearance
    kernel = np.ones((2*clearance+1, 2*clearance+1))
    return convolve2d(obstacle_grid, kernel, mode='same') > 0

def is_position_valid(x, y, angle, obstacle_map):
    """Check if robot fits in position without collisions"""
    half_len = ROBOT_LENGTH/2
    half_wid = ROBOT_WIDTH/2
    grid_size = obstacle_map.shape[0]
    
    # Calculate all four corners
    corners = [
        (x + half_len*np.cos(angle) - half_wid*np.sin(angle),
         y + half_len*np.sin(angle) + half_wid*np.cos(angle)),
        (x + half_len*np.cos(angle) + half_wid*np.sin(angle),
         y + half_len*np.sin(angle) - half_wid*np.cos(angle)),
        (x - half_len*np.cos(angle) + half_wid*np.sin(angle),
         y - half_len*np.sin(angle) - half_wid*np.cos(angle)),
        (x - half_len*np.cos(angle) - half_wid*np.sin(angle),
         y - half_len*np.sin(angle) + half_wid*np.cos(angle))
    ]
    
    # Check field boundaries first
    for cx, cy in corners:
        if not (0 <= cx <= FIELD_SIZE and 0 <= cy <= FIELD_SIZE):
            return False
    
    # Check obstacles (with safe grid access)
    for cx, cy in corners:
        grid_x = min(grid_size-1, max(0, int(cx/RESOLUTION)))
        grid_y = min(grid_size-1, max(0, int(cy/RESOLUTION)))
        if obstacle_map[grid_x, grid_y]:
            return False
            
    return True

def find_optimal_path(start_pos, goal_pos, obstacle_map, start_angle=0):
    """A* pathfinding with mecanum constraints"""
    grid_size = obstacle_map.shape[0]
    start_idx = (int(start_pos[0]/RESOLUTION), int(start_pos[1]/RESOLUTION))
    goal_idx = (int(goal_pos[0]/RESOLUTION), int(goal_pos[1]/RESOLUTION))
    
    priority_queue = []
    heappush(priority_queue, (0, 0, *start_idx, start_angle, None))
    visited = {}
    
    # 8-direction movement options
    directions = [
        (1, 0, 0), (-1, 0, pi), (0, 1, pi/2), (0, -1, 3*pi/2),
        (1, 1, pi/4), (1, -1, 7*pi/4), (-1, 1, 3*pi/4), (-1, -1, 5*pi/4)
    ]
    
    with tqdm(total=grid_size*grid_size*8, desc="Finding path") as progress:
        while priority_queue:
            _, cost, x, y, angle, parent = heappop(priority_queue)
            progress.update(1)
            
            if (x, y) == goal_idx:
                path = []
                while parent:
                    path.append((x, y, angle))
                    x, y, angle, parent = parent
                return path[::-1]
            
            if (x, y, angle) in visited:
                continue
            visited[(x, y, angle)] = True
            
            for dx, dy, move_angle in directions:
                new_x, new_y = x + dx, y + dy
                world_x = new_x * RESOLUTION
                world_y = new_y * RESOLUTION
                
                if not is_position_valid(world_x, world_y, move_angle, obstacle_map):
                    continue
                    
                move_dist = RESOLUTION * sqrt(dx**2 + dy**2)
                move_time = move_dist / calculate_mecanum_speed(degrees(move_angle))
                rot_time = abs((move_angle - angle + pi) % (2*pi) - pi) / ROTATION_SPEED
                
                # Properly formatted heappush with line continuation
                heappush(priority_queue,
                        (cost + move_time + rot_time,
                        cost + move_time + rot_time,
                        new_x, new_y, move_angle,
                    (x, y, angle, parent)))
    
    return None  # No path found

def execute_path(robot, path_coords):
    """Execute path with proper motor coordination"""
    if len(path_coords) <= 1:
        return
    
    prev_x, prev_y, prev_angle = path_coords[0]
    
    with tqdm(total=len(path_coords)-1, desc="Executing path") as progress:
        for x, y, angle in path_coords[1:]:
            dx = (x - prev_x) * RESOLUTION
            dy = (y - prev_y) * RESOLUTION
            distance = sqrt(dx**2 + dy**2)
            move_angle = atan2(dy, dx)
            
            # Calculate movement parameters
            move_time = distance / calculate_mecanum_speed(degrees(move_angle))
            rot_time = abs((move_angle - prev_angle + pi) % (2*pi) - pi) / ROTATION_SPEED
            
            # Execute rotation if needed
            if rot_time > 0.001:  # Significant rotation
                if move_angle > prev_angle:
                    # Clockwise rotation
                    robot.move_motor('FR', -MAX_RPS*60, rot_time)
                    robot.move_motor('FL', MAX_RPS*60, rot_time)
                    robot.move_motor('BR', -MAX_RPS*60, rot_time)
                    robot.move_motor('BL', MAX_RPS*60, rot_time)
                else:
                    # Counter-clockwise
                    robot.move_motor('FR', MAX_RPS*60, rot_time)
                    robot.move_motor('FL', -MAX_RPS*60, rot_time)
                    robot.move_motor('BR', MAX_RPS*60, rot_time)
                    robot.move_motor('BL', -MAX_RPS*60, rot_time)
            
            # Execute movement
            vx, vy = np.cos(move_angle), np.sin(move_angle)
            robot.move_motor('FL', (vx-vy)*MAX_RPS*60, move_time)
            robot.move_motor('FR', (vx+vy)*MAX_RPS*60, move_time)
            robot.move_motor('BL', (vx+vy)*MAX_RPS*60, move_time)
            robot.move_motor('BR', (vx-vy)*MAX_RPS*60, move_time)
            
            prev_x, prev_y, prev_angle = x, y, angle
            progress.update(1)

def generate_command_report(robot):
    """Print formatted motor command log"""
    print("\nMotor Command Execution Log:")
    print("{:<20} {:<12} {:<12} {:<12} {:<12}".format(
        "Time Interval", "Front Right", "Front Left", "Back Right", "Back Left"))
    
    # Collect all time points
    time_points = set()
    for cmd in robot.command_history:
        time_points.add(cmd['start'])
        time_points.add(cmd['end'])
    sorted_times = sorted(time_points)
    
    for i in range(len(sorted_times)-1):
        start, end = sorted_times[i], sorted_times[i+1]
        if end - start < 0.001:  # Skip negligible intervals
            continue
            
        # Format time string
        time_str = f"{int(start//60):02d}:{start%60:06.3f} - {int(end//60):02d}:{end%60:06.3f}"
        
        # Find active commands
        motors = {
            'FR': 0.0, 'FL': 0.0, 
            'BR': 0.0, 'BL': 0.0
        }
        for cmd in robot.command_history:
            if cmd['start'] <= start and cmd['end'] >= end:
                motors[cmd['motor']] = cmd['rpm']
        
        print("{:<20} {:<12.1f} {:<12.1f} {:<12.1f} {:<12.1f}".format(
            time_str, motors['FR'], motors['FL'], motors['BR'], motors['BL']))

def visualize_path(path_coords, obstacle_list, filename="path_visualization.png"):
    """Generate and save path visualization"""
    plt.figure(figsize=(10, 10))
    
    # Draw field boundary
    plt.plot([0, FIELD_SIZE, FIELD_SIZE, 0, 0], 
             [0, 0, FIELD_SIZE, FIELD_SIZE, 0], 'k-', linewidth=2)
    
    # Draw obstacles
    for x, y, side in obstacle_list:
        plt.gca().add_patch(plt.Rectangle((x-side/2, y-side/2), side, side,
                                        color='red', alpha=0.5))
    
    if path_coords:
        # Draw center path
        x_vals = [p[0]*RESOLUTION for p in path_coords]
        y_vals = [p[1]*RESOLUTION for p in path_coords]
        plt.plot(x_vals, y_vals, 'b-', linewidth=1.5, label='Path Center')
        
        # Draw robot outlines
        half_len = ROBOT_LENGTH/2
        half_wid = ROBOT_WIDTH/2
        
        for i in range(0, len(path_coords), max(1, len(path_coords)//20)):
            x, y, angle = path_coords[i]
            world_x = x * RESOLUTION
            world_y = y * RESOLUTION
            
            # Calculate corners
            corners = [
                (world_x + half_len*np.cos(angle) - half_wid*np.sin(angle),
                 world_y + half_len*np.sin(angle) + half_wid*np.cos(angle)),
                (world_x + half_len*np.cos(angle) + half_wid*np.sin(angle),
                 world_y + half_len*np.sin(angle) - half_wid*np.cos(angle)),
                (world_x - half_len*np.cos(angle) + half_wid*np.sin(angle),
                 world_y - half_len*np.sin(angle) - half_wid*np.cos(angle)),
                (world_x - half_len*np.cos(angle) - half_wid*np.sin(angle),
                 world_y - half_len*np.sin(angle) + half_wid*np.cos(angle))
            ]
            corners.append(corners[0])  # Close rectangle
            
            plt.plot([c[0] for c in corners], [c[1] for c in corners], 
                    'g-', linewidth=0.8, alpha=0.7)
            
            # Draw front indicator
            front_x = world_x + half_len * np.cos(angle)
            front_y = world_y + half_len * np.sin(angle)
            plt.plot([world_x, front_x], [world_y, front_y], 'r-', linewidth=1)
    
    plt.xlabel("X Position (meters)")
    plt.ylabel("Y Position (meters)")
    plt.title("Robot Navigation Path")
    plt.grid(True)
    plt.axis('equal')
    plt.legend()
    saveDir = "C:/Users/Kenny/Desktop/Coding/FTC"
    filepath = os.path.join(saveDir, filename)
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"\nVisualization saved to {filename}")

if __name__ == "__main__":
    # Configuration
    obstacles = [(2.0, 1.8, 0.3), (1.2, 3.4, 0.4), (0.7, 0.9, 0.2), (3.3, 0.5, 0.3), (1.9, 0.25, 0.2)]
    start_pos = (0.25, 0.25)
    goal_pos = (3.2, 2.8)
    initial_angle = 0  # radians (0 = facing positive x-axis)
    
    # Path planning
    print("Generating obstacle map...")
    obstacle_map = create_obstacle_map(obstacles)
    
    print("Finding optimal path...")
    path = find_optimal_path(start_pos, goal_pos, obstacle_map, initial_angle)
    
    if path:
        print(f"Path found with {len(path)} waypoints")
        
        # Visualization
        visualize_path(path, obstacles)
        
        # Execution
        print("Executing path...")
        robot = RobotController()
        execute_path(robot, path)
        
        # Reporting
        generate_command_report(robot)
    else:
        print("No valid path found!")