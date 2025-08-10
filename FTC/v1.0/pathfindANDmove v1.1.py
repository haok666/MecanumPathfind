import numpy as np
import matplotlib.pyplot as plt
from heapq import heappush, heappop
from math import sqrt, atan2, radians, degrees, pi
from scipy.signal import convolve2d
import time

# Constants
FIELD_SIZE = 3.6667
MAX_RPS = 4.0
ROBOT_DIMS = (0.423, 0.415)
RESOLUTION = 0.005
WHEEL_RADIUS = 0.03
ROTATION_SPEED = 2 * pi

class Robot:
    def movement(self, motor, rpm, duration):
        pass  # Interface for hardware control

def mecanum_speed(theta_deg):
    theta = radians(theta_deg)
    return 3.77 * (1 - 0.1*abs(np.sin(theta))) * (0.7 + 0.3*abs(np.cos(theta)))

def create_obstacle_map(obstacles):
    grid_dim = int(FIELD_SIZE / RESOLUTION)
    obstacle_map = np.zeros((grid_dim, grid_dim), dtype=bool)
    
    for x, y, side in obstacles:
        min_x = max(0, int((x-side/2)/RESOLUTION))
        max_x = min(grid_dim-1, int((x+side/2)/RESOLUTION))
        min_y = max(0, int((y-side/2)/RESOLUTION))
        max_y = min(grid_dim-1, int((y+side/2)/RESOLUTION))
        obstacle_map[min_x:max_x+1, min_y:max_y+1] = True
    
    clearance = int(0.297 / RESOLUTION)
    kernel = np.ones((2*clearance+1, 2*clearance+1))
    return convolve2d(obstacle_map, kernel, mode='same') > 0

def find_path(start, goal, obstacle_map, start_angle=0):
    grid_dim = obstacle_map.shape[0]
    start_idx = (int(start[0]/RESOLUTION), int(start[1]/RESOLUTION))
    goal_idx = (int(goal[0]/RESOLUTION), int(goal[1]/RESOLUTION))
    
    heap = []
    heappush(heap, (0, 0, *start_idx, start_angle, None))
    visited = {}
    
    directions = [(1,0,0), (-1,0,pi), (0,1,pi/2), (0,-1,3*pi/2),
                 (1,1,pi/4), (1,-1,7*pi/4), (-1,1,3*pi/4), (-1,-1,5*pi/4)]
    
    while heap:
        _, cost, x, y, angle, parent = heappop(heap)
        
        if (x,y) == goal_idx:
            path = []
            while parent:
                path.append((x, y, angle))
                x, y, angle, parent = parent
            return path[::-1]
        
        if (x,y,angle) in visited:
            continue
        visited[(x,y,angle)] = True
        
        for dx, dy, move_angle in directions:
            nx, ny = x+dx, y+dy
            if 0<=nx<grid_dim and 0<=ny<grid_dim and not obstacle_map[nx,ny]:
                move_dist = RESOLUTION * sqrt(dx**2 + dy**2)
                move_time = move_dist / mecanum_speed(degrees(move_angle))
                rot_time = abs((move_angle - angle + pi) % (2*pi) - pi) / ROTATION_SPEED
                heappush(heap, (cost + move_time + rot_time, 
                              cost + move_time + rot_time,
                              nx, ny, move_angle, (x, y, angle, parent)))

def execute_path(robot, path):
    if len(path) <= 1:
        return
    
    prev_x, prev_y, prev_angle = path[0]
    for x, y, angle in path[1:]:
        dx = (x - prev_x) * RESOLUTION
        dy = (y - prev_y) * RESOLUTION
        distance = sqrt(dx**2 + dy**2)
        move_angle = atan2(dy, dx)
        
        if abs(prev_angle - move_angle) > 0.1:
            rot_time = abs(prev_angle - move_angle) / ROTATION_SPEED
            if move_angle > prev_angle:
                robot.movement('FR', -MAX_RPS*60, rot_time)
                robot.movement('FL', MAX_RPS*60, rot_time)
            else:
                robot.movement('FR', MAX_RPS*60, rot_time)
                robot.movement('FL', -MAX_RPS*60, rot_time)
            time.sleep(rot_time)
        
        move_time = distance / mecanum_speed(degrees(move_angle))
        vx, vy = np.cos(move_angle), np.sin(move_angle)
        robot.movement('FL', (vx-vy)*MAX_RPS*60, move_time)
        robot.movement('FR', (vx+vy)*MAX_RPS*60, move_time)
        robot.movement('BL', (vx+vy)*MAX_RPS*60, move_time)
        robot.movement('BR', (vx-vy)*MAX_RPS*60, move_time)
        time.sleep(move_time)
        
        prev_x, prev_y, prev_angle = x, y, angle

def visualize_path(path, obstacles):
    plt.figure(figsize=(10, 10))
    
    # Draw field boundary
    plt.plot([0, FIELD_SIZE, FIELD_SIZE, 0, 0], [0, 0, FIELD_SIZE, FIELD_SIZE, 0], 'k-', linewidth=5)
    
    # Draw obstacles
    for x, y, side in obstacles:
        plt.gca().add_patch(plt.Rectangle((x-side/2, y-side/2), side, side, color='red', alpha=0.5, label='Obstacles'))
    
    # Draw path and robot outlines
    if path:
        # Center path line
        x_coords = [p[0]*RESOLUTION for p in path]
        y_coords = [p[1]*RESOLUTION for p in path]
        plt.plot(x_coords, y_coords, 'b-', linewidth=1.5, label='Path (center)')
        
        # Robot dimensions
        half_length = ROBOT_DIMS[0]/2
        half_width = ROBOT_DIMS[1]/2
        
        # Draw robot outlines at intervals
        for i in range(0, len(path), max(1, len(path)//20)):  # Show every 5% of path
            x, y, angle = path[i]
            x_world = x * RESOLUTION
            y_world = y * RESOLUTION
            
            # Calculate corner positions
            corners = [
                (x_world + half_length*np.cos(angle) - half_width*np.sin(angle),
                 y_world + half_length*np.sin(angle) + half_width*np.cos(angle)),
                
                (x_world + half_length*np.cos(angle) + half_width*np.sin(angle),
                 y_world + half_length*np.sin(angle) - half_width*np.cos(angle)),
                
                (x_world - half_length*np.cos(angle) + half_width*np.sin(angle),
                 y_world - half_length*np.sin(angle) - half_width*np.cos(angle)),
                
                (x_world - half_length*np.cos(angle) - half_width*np.sin(angle),
                 y_world - half_length*np.sin(angle) + half_width*np.cos(angle))
            ]
            
            # Close the rectangle
            corners.append(corners[0])
            
            # Plot robot outline
            plt.plot([c[0] for c in corners], [c[1] for c in corners], 
                    'g-', linewidth=0.8, alpha=0.7)
            
            # Draw front indicator
            front_x = x_world + half_length * np.cos(angle)
            front_y = y_world + half_length * np.sin(angle)
            plt.plot([x_world, front_x], [y_world, front_y], 'r-', linewidth=1)

    plt.xlabel("X (m)")
    plt.ylabel("Y (m)")
    plt.title("Robot Path with Outline Visualization")
    plt.legend()
    plt.grid(True)
    plt.axis('equal')
    plt.show()

if __name__ == "__main__":
    obstacles = [(1.85, 1.85, 1.0)]
    start = (0.25, 0.25)
    goal = (2.5, 3.35)
    
    obstacle_map = create_obstacle_map(obstacles)
    path = find_path(start, goal, obstacle_map)
    
    if path:
        visualize_path(path, obstacles)
        robot = Robot()
        execute_path(robot, path)
    else:
        print("No valid path found")