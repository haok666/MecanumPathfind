import numpy as np
import matplotlib.pyplot as plt
from heapq import heappush, heappop
from math import sqrt, atan2, radians, degrees, pi
from scipy.signal import convolve2d
import time

# Constants
FIELD_SIZE = 3.6667  # meters
MAX_RPS = 4.0
ROBOT_MASS = 5.0  # kg
ROBOT_LENGTH = 0.423  # meters
ROBOT_WIDTH = 0.415  # meters
RESOLUTION = 0.005  # 0.5cm grid
WHEEL_RADIUS = 0.03  # meters
ROTATION_SPEED = 2 * pi  # rad/s (1 full rotation per second at max speed)

# Mock robot interface
class Robot:
    def movement(self, motor, rpm, duration):
        print(f"Moving {motor} at {rpm} RPM for {duration:.2f}s")
        time.sleep(duration * 0.1)  # Simulate movement (10% of real time)

# Speed profile function
def mecanum_speed(theta_deg):
    theta = radians(theta_deg)
    friction_loss = 0.1 * abs(np.sin(theta))
    mecanum_efficiency = 0.7 + 0.3 * abs(np.cos(theta))
    max_speed = 3.77  # Theoretical max at 4 RPS
    return max_speed * (1 - friction_loss) * mecanum_efficiency

def rotation_time(current_angle, target_angle):
    angle_diff = abs((target_angle - current_angle + pi) % (2 * pi) - pi)
    return angle_diff / ROTATION_SPEED

def heuristic(a, b, current_angle):
    dx = (b[0] - a[0]) * RESOLUTION
    dy = (b[1] - a[1]) * RESOLUTION
    distance = sqrt(dx**2 + dy**2)
    target_angle = atan2(dy, dx)
    rot_time = rotation_time(current_angle, target_angle)
    return (distance / 3.77) + rot_time

def create_obstacle_map(obstacles):
    grid_dim = int(FIELD_SIZE / RESOLUTION)
    obstacle_map = np.zeros((grid_dim, grid_dim), dtype=bool)
    
    for (x, y, side) in obstacles:
        min_x = max(0, int((x - side/2) / RESOLUTION))
        max_x = min(grid_dim-1, int((x + side/2) / RESOLUTION))
        min_y = max(0, int((y - side/2) / RESOLUTION))
        max_y = min(grid_dim-1, int((y + side/2) / RESOLUTION))
        obstacle_map[min_x:max_x+1, min_y:max_y+1] = True
    
    clearance = int(0.297 / RESOLUTION)  # 29.7cm diagonal clearance
    kernel = np.ones((2*clearance+1, 2*clearance+1))
    return np.clip(convolve2d(obstacle_map, kernel, mode='same'), 0, 1)

def find_optimal_path(start, goal, obstacle_map, start_angle=0):
    grid_dim = obstacle_map.shape[0]
    grid_size = FIELD_SIZE / grid_dim
    
    start_idx = (int(start[0]/grid_size), int(start[1]/grid_size))
    goal_idx = (int(goal[0]/grid_size), int(goal[1]/grid_size))
    
    heap = []
    heappush(heap, (heuristic(start_idx, goal_idx, start_angle), 
                   0, *start_idx, start_angle, None))
    
    visited = {}
    
    directions = [(1,0,0), (-1,0,pi), (0,1,pi/2), (0,-1,3*pi/2),
                 (1,1,pi/4), (1,-1,7*pi/4), (-1,1,3*pi/4), (-1,-1,5*pi/4)]
    
    while heap:
        _, current_cost, x, y, current_angle, parent = heappop(heap)
        
        if (x,y) == goal_idx:
            return reconstruct_path(visited, (x, y, current_angle))
        
        if (x,y,current_angle) in visited and visited[(x,y,current_angle)][0] <= current_cost:
            continue
            
        visited[(x,y,current_angle)] = (current_cost, parent)
        
        for dx, dy, move_angle in directions:
            nx, ny = x + dx, y + dy
            
            if 0 <= nx < grid_dim and 0 <= ny < grid_dim:
                if obstacle_map[nx, ny]:
                    continue
                
                move_dist = grid_size * sqrt(dx**2 + dy**2)
                move_speed = mecanum_speed(degrees(move_angle))
                move_time = move_dist / move_speed
                rot_time = rotation_time(current_angle, move_angle)
                new_cost = current_cost + move_time + rot_time
                
                if (nx, ny, move_angle) not in visited or new_cost < visited.get((nx, ny, move_angle), (float('inf'),))[0]:
                    heappush(heap, (new_cost + heuristic((nx,ny), goal_idx, move_angle), 
                                  new_cost, nx, ny, move_angle, (x, y, current_angle)))
    
    return None

def reconstruct_path(visited, goal_node):
    path = []
    current = goal_node
    while current is not None:
        path.append(current)
        current = visited[current][1] if current in visited else None
    return path[::-1]

def execute_path(robot, path):
    if not path:
        print("No path to execute!")
        return
    
    prev_x, prev_y, prev_angle = path[0]
    
    for i in range(1, len(path)):
        x, y, angle = path[i]
        dx = x - prev_x
        dy = y - prev_y
        distance = RESOLUTION * sqrt(dx**2 + dy**2)
        
        # Calculate movement parameters
        move_angle = atan2(dy, dx)
        rot_time = rotation_time(prev_angle, move_angle)
        move_time = distance / mecanum_speed(degrees(move_angle))
        
        # Execute rotation if needed
        if rot_time > 0.01:  # Only rotate if significant
            print(f"Rotating from {degrees(prev_angle):.1f}° to {degrees(move_angle):.1f}°")
            rotate_robot(robot, prev_angle, move_angle)
        
        # Execute movement
        print(f"Moving {distance*100:.1f}cm at {degrees(move_angle):.1f}°")
        move_robot(robot, move_angle, distance)
        
        prev_x, prev_y, prev_angle = x, y, angle

def rotate_robot(robot, current_angle, target_angle):
    angle_diff = (target_angle - current_angle + pi) % (2 * pi) - pi
    duration = abs(angle_diff) / ROTATION_SPEED
    
    # Determine rotation direction (clockwise/counter-clockwise)
    if angle_diff > 0:
        # Rotate clockwise (right motors backward, left motors forward)
        robot.movement('FR', -MAX_RPS*60, duration)
        robot.movement('FL', MAX_RPS*60, duration)
        robot.movement('BR', -MAX_RPS*60, duration)
        robot.movement('BL', MAX_RPS*60, duration)
    else:
        # Rotate counter-clockwise
        robot.movement('FR', MAX_RPS*60, duration)
        robot.movement('FL', -MAX_RPS*60, duration)
        robot.movement('BR', MAX_RPS*60, duration)
        robot.movement('BL', -MAX_RPS*60, duration)

def move_robot(robot, angle, distance):
    speed_rps = mecanum_speed(degrees(angle))
    duration = distance / speed_rps
    
    # Convert angle to wheel speeds
    vx = np.cos(angle)
    vy = np.sin(angle)
    
    # Wheel speeds in RPS
    fl = (vx - vy) * MAX_RPS
    fr = (vx + vy) * MAX_RPS
    bl = (vx + vy) * MAX_RPS
    br = (vx - vy) * MAX_RPS
    
    # Execute movement
    robot.movement('FL', fl*60, duration)
    robot.movement('FR', fr*60, duration)
    robot.movement('BL', bl*60, duration)
    robot.movement('BR', br*60, duration)

def visualize_path(path, obstacles):
    if not path:
        print("No path to visualize")
        return
    
    plt.figure(figsize=(10, 10))
    
    # Draw obstacles
    for (x, y, side) in obstacles:
        plt.gca().add_patch(plt.Rectangle((x-side/2, y-side/2), side, side, color='red', alpha=0.5))
    
    # Draw path
    x_coords = [p[0]*RESOLUTION for p in path]
    y_coords = [p[1]*RESOLUTION for p in path]
    plt.plot(x_coords, y_coords, 'b-', linewidth=2)
    
    # Draw orientation arrows every 10 steps
    for i in range(0, len(path), 10):
        x, y, angle = path[i]
        dx = 0.1 * np.cos(angle)
        dy = 0.1 * np.sin(angle)
        plt.arrow(x*RESOLUTION, y*RESOLUTION, dx, dy, head_width=0.05, color='green')
    
    plt.xlim(0, FIELD_SIZE)
    plt.ylim(0, FIELD_SIZE)
    plt.title("Robot Path with Orientation")
    plt.xlabel("X (m)")
    plt.ylabel("Y (m)")
    plt.grid(True)
    plt.show()

# Main execution
if __name__ == "__main__":
    # Define obstacles (x,y,side_length)
    obstacles = [
        (1.0, 1.0, 0.5),
        (2.5, 2.0, 0.3),
        (1.8, 1.2, 0.4)
    ]
    
    start = (0.1, 0.1)
    goal = (3.5, 3.5)
    start_angle = 0  # Facing positive x-axis
    
    # Create obstacle map
    obstacle_map = create_obstacle_map(obstacles)
    
    # Find path
    path = find_optimal_path(start, goal, obstacle_map, start_angle)
    
    if path:
        print(f"Found path with {len(path)} steps")
        
        # Visualize
        visualize_path(path, obstacles)
        
        # Execute path
        robot = Robot()
        execute_path(robot, path)
    else:
        print("No valid path found")