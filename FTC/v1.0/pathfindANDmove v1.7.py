import numpy as np
import matplotlib.pyplot as plt
from heapq import heappush, heappop
from math import sqrt, atan2, radians, degrees, pi, cos, sin
from scipy.signal import convolve2d
import time
import os

# Constants
FIELD_SIZE = 3.6667  # meters
MAX_RPS = 4.0        # revolutions per second
ROBOT_LENGTH = 0.423 # meters
ROBOT_WIDTH = 0.415  # meters
RESOLUTION = 0.005   # meters per grid cell
WHEEL_RADIUS = 0.03  # meters
ROTATION_SPEED = 90  # degrees per second

class RobotController:
    def __init__(self):
        self.command_history = []
        self.current_time = 0.0
        self.current_angle = 0  # degrees
        self.x_pos = 0.0
        self.y_pos = 0.0

    def move_to_position(self, target_x, target_y, speed=0.5):
        """Move to target coordinates while avoiding obstacles"""
        angle_to_target = degrees(atan2(target_y - self.y_pos, target_x - self.x_pos))
        distance = sqrt((target_x - self.x_pos)**2 + (target_y - self.y_pos)**2)
        
        self.turn_to_angle(angle_to_target)
        self.move_in_direction(angle_to_target, distance, speed)

    def move_in_direction(self, direction_deg, distance, speed):
        """Execute movement in specified direction"""
        duration = distance / speed
        vx = cos(radians(direction_deg))
        vy = sin(radians(direction_deg))
        
        # Calculate wheel speeds
        fr_rpm = (vx - vy) * MAX_RPS * 60
        fl_rpm = (vx + vy) * MAX_RPS * 60
        br_rpm = (vx + vy) * MAX_RPS * 60
        bl_rpm = (vx - vy) * MAX_RPS * 60
        
        self._log_command('FR', fr_rpm, duration)
        self._log_command('FL', fl_rpm, duration)
        self._log_command('BR', br_rpm, duration)
        self._log_command('BL', bl_rpm, duration)
        
        # Update position
        self.x_pos += distance * cos(radians(direction_deg))
        self.y_pos += distance * sin(radians(direction_deg))
        self.current_time += duration

    def turn_to_angle(self, target_angle_deg):
        """Rotate to specified heading"""
        angle_diff = (target_angle_deg - self.current_angle + 180) % 360 - 180
        duration = abs(angle_diff) / ROTATION_SPEED
        
        if angle_diff > 0:
            # Clockwise turn
            self._log_command('FR', -MAX_RPS*60, duration)
            self._log_command('FL', MAX_RPS*60, duration)
            self._log_command('BR', -MAX_RPS*60, duration)
            self._log_command('BL', MAX_RPS*60, duration)
        else:
            # Counter-clockwise turn
            self._log_command('FR', MAX_RPS*60, duration)
            self._log_command('FL', -MAX_RPS*60, duration)
            self._log_command('BR', MAX_RPS*60, duration)
            self._log_command('BL', -MAX_RPS*60, duration)
        
        self.current_angle = target_angle_deg
        self.current_time += duration

    def _log_command(self, motor, rpm, duration):
        """Record motor command"""
        self.command_history.append({
            'motor': motor,
            'rpm': rpm,
            'start': self.current_time,
            'end': self.current_time + duration,
            'x': self.x_pos,
            'y': self.y_pos,
            'angle': self.current_angle
        })

def create_obstacle_map(obstacle_list):
    """Generate grid map with obstacles"""
    grid_size = int(FIELD_SIZE / RESOLUTION)
    obstacle_grid = np.zeros((grid_size, grid_size), dtype=bool)
    
    for x, y, side in obstacle_list:
        min_x = max(0, int((x-side/2)/RESOLUTION))
        max_x = min(grid_size-1, int((x+side/2)/RESOLUTION))
        min_y = max(0, int((y-side/2)/RESOLUTION))
        max_y = min(grid_size-1, int((y+side/2)/RESOLUTION))
        obstacle_grid[min_x:max_x+1, min_y:max_y+1] = True
    
    # Add safety margin
    clearance = int(0.297 / RESOLUTION)
    kernel = np.ones((2*clearance+1, 2*clearance+1))
    return convolve2d(obstacle_grid, kernel, mode='same') > 0

def find_optimal_path(start_pos, goal_pos, obstacle_map):
    """A* pathfinding implementation"""
    grid_size = obstacle_map.shape[0]
    start_x = int(start_pos[0]/RESOLUTION)
    start_y = int(start_pos[1]/RESOLUTION)
    goal_x = int(goal_pos[0]/RESOLUTION)
    goal_y = int(goal_pos[1]/RESOLUTION)
    
    open_set = []
    heappush(open_set, (0, start_x, start_y))
    
    came_from = {}
    g_score = {(start_x, start_y): 0}
    
    directions = [(0,1), (1,1), (1,0), (1,-1), (0,-1), (-1,-1), (-1,0), (-1,1)]
    
    while open_set:
        _, current_x, current_y = heappop(open_set)
        
        if (current_x, current_y) == (goal_x, goal_y):
            path = []
            while (current_x, current_y) in came_from:
                path.append((current_x, current_y))
                current_x, current_y = came_from[(current_x, current_y)]
            return [(p[0]*RESOLUTION, p[1]*RESOLUTION) for p in reversed(path)]
        
        for dx, dy in directions:
            neighbor_x = current_x + dx
            neighbor_y = current_y + dy
            
            if (0 <= neighbor_x < grid_size and 
                0 <= neighbor_y < grid_size and 
                not obstacle_map[neighbor_x, neighbor_y]):
                
                move_cost = sqrt(dx**2 + dy**2)
                tentative_g = g_score[(current_x, current_y)] + move_cost
                
                if ((neighbor_x, neighbor_y) not in g_score or 
                    tentative_g < g_score[(neighbor_x, neighbor_y)]):
                    came_from[(neighbor_x, neighbor_y)] = (current_x, current_y)
                    g_score[(neighbor_x, neighbor_y)] = tentative_g
                    heuristic = sqrt((neighbor_x-goal_x)**2 + (neighbor_y-goal_y)**2)
                    heappush(open_set, (tentative_g + heuristic, neighbor_x, neighbor_y))
    
    return None

def visualize_path(robot, obstacle_list, output_dir="."):
    """Generate clean path visualization with robot outline"""
    plt.figure(figsize=(10, 10))
    
    # Setup plot
    plt.xlim(0, FIELD_SIZE)
    plt.ylim(0, FIELD_SIZE)
    plt.gca().set_aspect('equal')
    plt.grid(True, alpha=0.3)
    
    # Draw obstacles
    for x, y, side in obstacle_list:
        plt.gca().add_patch(
            plt.Rectangle(
                (x-side/2, y-side/2), side, side,
                color='#FF6B6B', alpha=0.3, 
                edgecolor='#FF2626', linewidth=1
            )
        )
    
    # Draw path
    path_x = [cmd['x'] for cmd in robot.command_history]
    path_y = [cmd['y'] for cmd in robot.command_history]
    plt.plot(path_x, path_y, '#4285F4', linewidth=1.5, alpha=0.9, label='Path')
    
    # NEW: Draw robot outline at sample points along path
    sample_rate = max(1, len(robot.command_history) // 20)  # Show ~20 outlines
    for i, cmd in enumerate(robot.command_history):
        if i % sample_rate == 0 or i == len(robot.command_history)-1:
            # Calculate corners of robot rectangle
            angle_rad = radians(cmd['angle'])
            cos_a, sin_a = cos(angle_rad), sin(angle_rad)
            half_l, half_w = ROBOT_LENGTH/2, ROBOT_WIDTH/2
            
            # Calculate corner positions
            corners = [
                (cmd['x'] + half_l*cos_a - half_w*sin_a, cmd['y'] + half_l*sin_a + half_w*cos_a),
                (cmd['x'] + half_l*cos_a + half_w*sin_a, cmd['y'] + half_l*sin_a - half_w*cos_a),
                (cmd['x'] - half_l*cos_a + half_w*sin_a, cmd['y'] - half_l*sin_a - half_w*cos_a),
                (cmd['x'] - half_l*cos_a - half_w*sin_a, cmd['y'] - half_l*sin_a + half_w*cos_a),
                (cmd['x'] + half_l*cos_a - half_w*sin_a, cmd['y'] + half_l*sin_a + half_w*cos_a)  # Close the rectangle
            ]
            
            # Plot outline
            outline_x, outline_y = zip(*corners)
            plt.plot(outline_x, outline_y, '#4285F4', linewidth=0.5, alpha=0.2)
    
    # Mark key positions
    if robot.command_history:
        # Start position
        plt.scatter(
            robot.command_history[0]['x'], 
            robot.command_history[0]['y'],
            color='#0F9D58', s=100, label='Start'
        )
        # End position
        plt.scatter(
            robot.command_history[-1]['x'],
            robot.command_history[-1]['y'],
            color='#DB4437', s=100, label='End'
        )
    
    plt.title("Robot Navigation Path", pad=20)
    plt.xlabel("X Position (meters)")
    plt.ylabel("Y Position (meters)")
    plt.legend()
    
    # Save output
    os.makedirs(output_dir, exist_ok=True)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    filepath = os.path.join(output_dir, f"robot_path_{timestamp}.png")
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Visualization saved to {filepath}")

def print_command_log(robot, output_dir="."):
    """Print organized command log to terminal and save to text file"""
    os.makedirs(output_dir, exist_ok=True)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    log_filepath = os.path.join(output_dir, f"robot_log_{timestamp}.txt")
    
    # Create a string buffer to hold all output
    output_buffer = []
    
    # Header
    header = "="*80 + "\n" + "ROBOT COMMAND LOG".center(80) + "\n" + "="*80
    output_buffer.append(header)
    
    # Group commands into movement segments
    segments = []
    current_segment = None
    
    for cmd in robot.command_history:
        if current_segment is None:
            current_segment = {
                'start': cmd['start'],
                'end': cmd['end'],
                'motors': {
                    'FR': cmd['rpm'],
                    'FL': cmd['rpm'],
                    'BR': cmd['rpm'],
                    'BL': cmd['rpm']
                },
                'type': 'ROTATE' if abs(cmd['rpm']) == MAX_RPS*60 else 'MOVE',
                'x': cmd['x'],
                'y': cmd['y'],
                'angle': cmd['angle']
            }
        else:
            if (abs(cmd['start'] - current_segment['start']) < 0.001 and
                abs(cmd['end'] - current_segment['end']) < 0.001):
                current_segment['motors'][cmd['motor']] = cmd['rpm']
            else:
                segments.append(current_segment)
                current_segment = None
    
    if current_segment:
        segments.append(current_segment)
    
    # Table header
    table_header = (f"\n{'Time':<15} {'Type':<10} {'Position':<20} {'Heading':<10} "
                   f"{'Motor Speeds (RPM)':<30}")
    output_buffer.append(table_header)
    output_buffer.append("-"*80)
    
    # Add each segment to output
    for seg in segments:
        time_str = f"{seg['start']:.2f}-{seg['end']:.2f}s"
        move_type = seg['type']
        pos_str = f"({seg['x']:.2f}, {seg['y']:.2f})"
        angle_str = f"{seg['angle']:.0f}°"
        motors_str = (
            f"FR:{seg['motors']['FR']:>6.1f} "
            f"FL:{seg['motors']['FL']:>6.1f} "
            f"BR:{seg['motors']['BR']:>6.1f} "
            f"BL:{seg['motors']['BL']:>6.1f}"
        )
        
        line = f"{time_str:<15} {move_type:<10} {pos_str:<20} {angle_str:<10} {motors_str:<30}"
        output_buffer.append(line)
    
    # Footer with summary
    footer = "="*80
    summary = (f"\nTotal commands: {len(segments)}\n"
               f"Total duration: {robot.command_history[-1]['end']:.2f}s" 
               if robot.command_history else "0s")
    output_buffer.append(summary)
    output_buffer.append(footer)
    
    # Write to file
    with open(log_filepath, 'w') as f:
        f.write("\n".join(output_buffer))
    
    # Print to terminal
    print("\n".join(output_buffer))
    print(f"\nLog file saved to {log_filepath}")

if __name__ == "__main__":
    # Configuration
    obstacles = [(1.0, 1.0, 0.5), (2.0, 2.0, 0.3), (3.0, 1.5, 0.4)]
    start_pos = (0.1, 0.1)
    goal_pos = (3.5, 3.5)
    
    # Create path
    obstacle_map = create_obstacle_map(obstacles)
    path = find_optimal_path(start_pos, goal_pos, obstacle_map)
    
    if path:
        print(f"Found path with {len(path)} waypoints")
        
        # Execute path
        robot = RobotController()
        for x, y in path:
            robot.move_to_position(x, y)
        
        # Generate outputs
        visualize_path(robot, obstacles)
        print_command_log(robot)  # Now saves to file automatically
    else:
        print("No valid path found!")

if __name__ == "__main__":
    # Configuration
    obstacles = [(1.0, 1.0, 0.5), (2.0, 2.0, 0.3), (3.0, 1.5, 0.4)]
    start_pos = (0.1, 0.1)
    goal_pos = (3.5, 3.5)
    
    # Create path
    obstacle_map = create_obstacle_map(obstacles)
    path = find_optimal_path(start_pos, goal_pos, obstacle_map)
    
    if path:
        print(f"Found path with {len(path)} waypoints")
        
        # Execute path
        robot = RobotController()
        for x, y in path:
            robot.move_to_position(x, y)
        
        # Generate outputs
        visualize_path(robot, obstacles)
        print_command_log(robot)
    else:
        print("No valid path found!")