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
ROTATION_SPEED = 720  # degrees per second
MIN_TURN_ANGLE = 15  # degrees before initiating turn
SPEED_PRECISION = 1  # decimal places for speed rounding
POSITION_PRECISION = 3  # decimal places for position rounding
ANGLE_PRECISION = 0   # decimal places for angle rounding
ZIGZAG_THRESHOLD = 0.1  # meters deviation to consider as zigzag
ZIGZAG_WINDOW = 5      # number of points to analyze
STRAIGHT_LINE_BOOST = 1.2  # speed multiplier when going straight
MIN_DISTANCE_FOR_ZIGZAG_CHECK = 1.0  # meters from target to check for zigzag

class RobotController:
    def __init__(self):
        self.command_history = []
        self.current_time = 0.0
        self.current_angle = 0  # degrees
        self.x_pos = 0.0
        self.y_pos = 0.0
        self.prev_positions = []  # For zigzag detection
        self.last_motor_values = {'FR': 0, 'FL': 0, 'BR': 0, 'BL': 0}
        self.last_command_type = None

    def _round_speed(self, speed):
        return round(speed, SPEED_PRECISION)

    def _round_position(self, pos):
        return round(pos, POSITION_PRECISION)

    def _round_angle(self, angle):
        return round(angle, ANGLE_PRECISION)

    def _detect_zigzag(self):
        """Detect if recent movements form a zigzag pattern"""
        if len(self.prev_positions) < ZIGZAG_WINDOW:
            return False
            
        # Calculate total movement vector
        total_dx = self.prev_positions[-1][0] - self.prev_positions[0][0]
        total_dy = self.prev_positions[-1][1] - self.prev_positions[0][1]
        
        # If total movement is small, don't consider it zigzag
        if sqrt(total_dx**2 + total_dy**2) < 0.05:
            return False
            
        avg_angle = atan2(total_dy, total_dx)
        
        # Check deviations from average direction
        deviations = []
        for p1, p2 in zip(self.prev_positions[:-1], self.prev_positions[1:]):
            segment_angle = atan2(p2[1]-p1[1], p2[0]-p1[0])
            angle_diff = (segment_angle - avg_angle + pi) % (2*pi) - pi
            deviations.append(abs(angle_diff))
        
        # If most segments deviate significantly, it's a zigzag
        significant_deviations = sum(d > radians(15) for d in deviations)
        return significant_deviations / len(deviations) > 0.6

    def _execute_straight_line(self, angle, distance, speed):
        """Execute optimized straight line movement"""
        angle_deg = degrees(angle)
        self.turn_to_angle(angle_deg)
        
        duration = distance / speed
        rpm = self._round_speed(speed * MAX_RPS * 60 / (WHEEL_RADIUS * 2 * pi))
        
        # Only log if different from last command
        current_motors = {'FR': rpm, 'FL': rpm, 'BR': rpm, 'BL': rpm}
        if (self.last_command_type != 'MOVE' or 
            any(self.last_motor_values[m] != rpm for m in ['FR', 'FL', 'BR', 'BL'])):
            
            self._log_command('FR', rpm, duration)
            self._log_command('FL', rpm, duration)
            self._log_command('BR', rpm, duration)
            self._log_command('BL', rpm, duration)
            self.last_motor_values = current_motors
            self.last_command_type = 'MOVE'
        
        # Update position
        self.x_pos = self._round_position(self.x_pos + distance * cos(angle))
        self.y_pos = self._round_position(self.y_pos + distance * sin(angle))
        self.current_time += duration

    def move_to_position(self, target_x, target_y, speed=0.5):
        target_x = self._round_position(target_x)
        target_y = self._round_position(target_y)
        speed = self._round_speed(speed)
        
        # Track position for zigzag detection
        self.prev_positions.append((self.x_pos, self.y_pos))
        if len(self.prev_positions) > ZIGZAG_WINDOW:
            self.prev_positions.pop(0)
        
        # Check for zigzag pattern when close to target
        distance_to_target = sqrt((target_x-self.x_pos)**2 + (target_y-self.y_pos)**2)
        if (distance_to_target < MIN_DISTANCE_FOR_ZIGZAG_CHECK and 
            self._detect_zigzag() and 
            distance_to_target > 0.1):
            
            print(f"Optimizing zigzag path (distance: {distance_to_target:.2f}m)")
            # Calculate direct line to target
            direct_angle = atan2(target_y - self.y_pos, target_x - self.x_pos)
            
            # Boost speed for straight line
            boosted_speed = min(speed * STRAIGHT_LINE_BOOST, MAX_RPS * WHEEL_RADIUS * 2 * pi)
            self._execute_straight_line(direct_angle, distance_to_target, boosted_speed)
            return
            
        # Normal movement
        while True:
            dx = target_x - self._round_position(self.x_pos)
            dy = target_y - self._round_position(self.y_pos)
            distance = sqrt(dx**2 + dy**2)
            
            if distance < 0.001:
                break
                
            step_size = min(0.1, distance)
            step_x = self._round_position(self.x_pos + dx * (step_size/distance))
            step_y = self._round_position(self.y_pos + dy * (step_size/distance))
            
            self._execute_movement_step(step_x, step_y, speed)

    def _execute_movement_step(self, target_x, target_y, speed):
        dx = target_x - self._round_position(self.x_pos)
        dy = target_y - self._round_position(self.y_pos)
        distance = sqrt(dx**2 + dy**2)
        desired_angle = self._round_angle(degrees(atan2(dy, dx)))
        angle_diff = (desired_angle - self.current_angle + 180) % 360 - 180
        
        if abs(angle_diff) > MIN_TURN_ANGLE:
            self.turn_to_angle(desired_angle)
        
        move_angle_rad = radians(desired_angle - self.current_angle)
        vx = cos(move_angle_rad) * speed
        vy = sin(move_angle_rad) * speed
        
        duration = distance / speed
        
        fr_rpm = self._round_speed((vx - vy) * MAX_RPS * 60)
        fl_rpm = self._round_speed((vx + vy) * MAX_RPS * 60)
        br_rpm = self._round_speed((vx + vy) * MAX_RPS * 60)
        bl_rpm = self._round_speed((vx - vy) * MAX_RPS * 60)
        
        max_rpm = max(abs(fr_rpm), abs(fl_rpm), abs(br_rpm), abs(bl_rpm))
        if max_rpm > MAX_RPS * 60:
            scale = (MAX_RPS * 60) / max_rpm
            fr_rpm = self._round_speed(fr_rpm * scale)
            fl_rpm = self._round_speed(fl_rpm * scale)
            br_rpm = self._round_speed(br_rpm * scale)
            bl_rpm = self._round_speed(bl_rpm * scale)
        
        # Only log if different from last command
        current_motors = {'FR': fr_rpm, 'FL': fl_rpm, 'BR': br_rpm, 'BL': bl_rpm}
        if (self.last_command_type != 'MOVE' or 
            any(self.last_motor_values[m] != current_motors[m] for m in ['FR', 'FL', 'BR', 'BL'])):
            
            self._log_command('FR', fr_rpm, duration)
            self._log_command('FL', fl_rpm, duration)
            self._log_command('BR', br_rpm, duration)
            self._log_command('BL', bl_rpm, duration)
            self.last_motor_values = current_motors
            self.last_command_type = 'MOVE'
        
        self.x_pos = target_x
        self.y_pos = target_y
        self.current_angle = desired_angle
        self.current_time += duration

    def turn_to_angle(self, target_angle_deg):
        target_angle_deg = self._round_angle(target_angle_deg)
        angle_diff = (target_angle_deg - self.current_angle + 180) % 360 - 180
        if abs(angle_diff) < 1:
            return
            
        duration = abs(angle_diff) / ROTATION_SPEED
        
        rpm = self._round_speed(MAX_RPS * 60)
        if angle_diff > 0:
            # Clockwise turn
            new_motors = {'FR': -rpm, 'FL': rpm, 'BR': -rpm, 'BL': rpm}
        else:
            # Counter-clockwise turn
            new_motors = {'FR': rpm, 'FL': -rpm, 'BR': rpm, 'BL': -rpm}
        
        # Only log if different from last command
        if (self.last_command_type != 'ROTATE' or 
            any(self.last_motor_values[m] != new_motors[m] for m in ['FR', 'FL', 'BR', 'BL'])):
            
            for motor, rpm_val in new_motors.items():
                self._log_command(motor, rpm_val, duration)
            self.last_motor_values = new_motors
            self.last_command_type = 'ROTATE'
        
        self.current_angle = target_angle_deg
        self.current_time += duration

    def _log_command(self, motor, rpm, duration):
        self.command_history.append({
            'motor': motor,
            'rpm': rpm,
            'start': self.current_time,
            'end': self.current_time + duration,
            'x': self._round_position(self.x_pos),
            'y': self._round_position(self.y_pos),
            'angle': self._round_angle(self.current_angle)
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

def smooth_path(path, obstacle_map, smoothing_aggressiveness=0.5):
    """
    Smooth path while ensuring it reaches the destination
    Args:
        smoothing_aggressiveness: 0.0 (no smoothing) to 1.0 (maximum smoothing)
    """
    if len(path) < 3:
        return path.copy()
    
    # Convert to numpy array for vector operations
    path_array = np.array(path)
    
    # Always keep first and last points exactly the same
    smoothed = path_array.copy()
    
    # Apply simple averaging filter
    for i in range(1, len(path)-1):
        # Weighted average between original and smoothed path
        smoothed[i] = (smoothing_aggressiveness * 0.5 * (smoothed[i-1] + smoothed[i+1]) + (1 - smoothing_aggressiveness) * path_array[i])
        
        # Ensure we don't collide with obstacles
        if is_collision(smoothed[i], obstacle_map):
            smoothed[i] = path_array[i]  # Revert if collision
    
    # Ensure we have enough points for proper following
    min_points = max(10, int(len(path) * 0.3))  # At least 10 points or 30% of original
    if len(smoothed) < min_points:
        # Linear interpolation to add more points
        new_path = []
        for i in range(len(smoothed)-1):
            new_path.append(smoothed[i])
            # Add intermediate points every 10cm
            dist = np.linalg.norm(smoothed[i+1] - smoothed[i])
            num_points = int(dist / 0.1)
            for j in range(1, num_points+1):
                t = j / (num_points+1)
                new_point = smoothed[i] * (1-t) + smoothed[i+1] * t
                new_path.append(new_point)
        new_path.append(smoothed[-1])
        smoothed = np.array(new_path)
    
    return smoothed

def is_collision(point, obstacle_map):
    """Check if point collides with obstacles"""
    x = int(point[0]/RESOLUTION)
    y = int(point[1]/RESOLUTION)
    if 0 <= x < obstacle_map.shape[0] and 0 <= y < obstacle_map.shape[1]:
        return obstacle_map[x, y]
    return True

def visualize_path(robot, obstacle_list, output_dir="."):
    """Generate visualization with robot outline"""
    plt.figure(figsize=(10, 10))
    plt.xlim(0, FIELD_SIZE)
    plt.ylim(0, FIELD_SIZE)
    plt.gca().set_aspect('equal')
    plt.grid(True, alpha=0.3)
    
    # Draw obstacles
    for x, y, side in obstacle_list:
        plt.gca().add_patch(
            plt.Rectangle(
                (x-side/2, y-side/2), side, side,
                color='#2E86C1', alpha=0.3,
                edgecolor='#1B4F72', linewidth=1
            )
        )
    
    # Draw path
    path_x = [cmd['x'] for cmd in robot.command_history]
    path_y = [cmd['y'] for cmd in robot.command_history]
    plt.plot(path_x, path_y, '#E74C3C', linewidth=2.0, alpha=0.9, label='Path Center')
    
    # Draw robot outlines
    sample_rate = max(1, len(robot.command_history) // 20)
    for i, cmd in enumerate(robot.command_history):
        if i % sample_rate == 0 or i == len(robot.command_history)-1:
            angle_rad = radians(cmd['angle'])
            cos_a, sin_a = cos(angle_rad), sin(angle_rad)
            half_l, half_w = ROBOT_LENGTH/2, ROBOT_WIDTH/2
            
            corners = [
                (cmd['x'] + half_l*cos_a - half_w*sin_a, cmd['y'] + half_l*sin_a + half_w*cos_a),
                (cmd['x'] + half_l*cos_a + half_w*sin_a, cmd['y'] + half_l*sin_a - half_w*cos_a),
                (cmd['x'] - half_l*cos_a + half_w*sin_a, cmd['y'] - half_l*sin_a - half_w*cos_a),
                (cmd['x'] - half_l*cos_a - half_w*sin_a, cmd['y'] - half_l*sin_a + half_w*cos_a),
                (cmd['x'] + half_l*cos_a - half_w*sin_a, cmd['y'] + half_l*sin_a + half_w*cos_a)
            ]
            
            outline_x, outline_y = zip(*corners)
            plt.plot(outline_x, outline_y, '#7D3C98', linewidth=0.8, alpha=0.4)
    
    # Mark start/end
    if robot.command_history:
        plt.scatter(
            robot.command_history[0]['x'], 
            robot.command_history[0]['y'],
            color='#27AE60', s=120, label='Start', edgecolors='black', linewidths=0.5
        )
        plt.scatter(
            robot.command_history[-1]['x'],
            robot.command_history[-1]['y'],
            color='#E74C3C', s=120, label='End', edgecolors='black', linewidths=0.5
        )
    
    plt.title("Robot Navigation Path", pad=20, fontsize=14)
    plt.xlabel("X Position (meters)", fontsize=12)
    plt.ylabel("Y Position (meters)", fontsize=12)
    plt.legend(loc='upper right', fontsize=10)
    
    os.makedirs(output_dir, exist_ok=True)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    filepath = os.path.join(output_dir, f"robot_path_{timestamp}.png")
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Visualization saved to {filepath}")

def print_command_log(robot, output_dir="."):
    """Print and save command log with proper consolidation"""
    os.makedirs(output_dir, exist_ok=True)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    log_filepath = os.path.join(output_dir, f"robot_log_{timestamp}.txt")
    
    output_buffer = []
    output_buffer.append("="*80)
    output_buffer.append("ROBOT COMMAND LOG".center(80))
    output_buffer.append("="*80)

    # First organize commands by time period with all four wheels
    time_periods = {}
    for cmd in robot.command_history:
        period_key = (cmd['start'], cmd['end'])
        if period_key not in time_periods:
            time_periods[period_key] = {
                'motors': {'FR': 0, 'FL': 0, 'BR': 0, 'BL': 0},
                'x': round(cmd['x'], POSITION_PRECISION),
                'y': round(cmd['y'], POSITION_PRECISION),
                'angle': round(cmd['angle'], ANGLE_PRECISION)
            }
        time_periods[period_key]['motors'][cmd['motor']] = round(cmd['rpm'], SPEED_PRECISION)

    # Convert to list of segments sorted by time
    segments = []
    for (start, end), data in sorted(time_periods.items(), key=lambda x: x[0][0]):
        segments.append({
            'start': start,
            'end': end,
            'motors': data['motors'],
            'type': 'ROTATE' if any(abs(rpm) == round(MAX_RPS*60, SPEED_PRECISION) for rpm in data['motors'].values()) else 'MOVE',
            'x': data['x'],
            'y': data['y'],
            'angle': data['angle'],
            'original_start': start,
            'original_end': end
        })

    # Now consolidate continuous identical segments
    consolidated = []
    if segments:
        current = segments[0].copy()
        
        for seg in segments[1:]:
            if (seg['motors'] == current['motors'] and
                seg['type'] == current['type'] and
                seg['x'] == current['x'] and
                seg['y'] == current['y'] and
                seg['angle'] == current['angle']):
                
                current['end'] = seg['end']
                current['original_end'] = seg['original_end']
            else:
                consolidated.append(current)
                current = seg.copy()

        consolidated.append(current)

    # Build output
    output_buffer.append(f"\n{'Time':<15} {'Type':<10} {'Position':<20} {'Heading':<10} {'Motor Speeds (RPM)':<30}")
    output_buffer.append("-"*80)

    for seg in consolidated:
        time_str = f"{seg['original_start']:.2f}-{seg['original_end']:.2f}s" if seg['original_start'] != seg['original_end'] else f"{seg['original_start']:.2f}s"
        move_type = seg['type']
        pos_str = f"({seg['x']:.2f}, {seg['y']:.2f})"
        angle_str = f"{seg['angle']:.0f}°"
        motors_str = (
            f"FR:{seg['motors']['FR']:>6.1f} "
            f"FL:{seg['motors']['FL']:>6.1f} "
            f"BR:{seg['motors']['BR']:>6.1f} "
            f"BL:{seg['motors']['BL']:>6.1f}"
        )

        output_buffer.append(f"{time_str:<15} {move_type:<10} {pos_str:<20} {angle_str:<10} {motors_str:<30}")

    output_buffer.append("="*80)
    output_buffer.append(f"\nTotal commands: {len(consolidated)} (reduced from {len(segments)})")
    output_buffer.append(f"Total duration: {robot.command_history[-1]['end']:.2f}s" if robot.command_history else "0s")
    output_buffer.append("="*80)

    with open(log_filepath, 'w') as f:
        f.write("\n".join(output_buffer))
    print("\n".join(output_buffer))
    print(f"\nLog file saved to {log_filepath}")

if __name__ == "__main__":
    # Configuration
    obstacles = [(1.0, 1.0, 0.5), (2.0, 2.0, 0.3), (3.0, 1.5, 0.4)]
    start_pos = (0.1, 0.1)
    goal_pos = (3.5, 3.5)
    smoothing_aggressiveness = 0.5  # Moderate smoothing
    
    # Create path
    obstacle_map = create_obstacle_map(obstacles)
    path = find_optimal_path(start_pos, goal_pos, obstacle_map)
    
    if path:
        print(f"Original path has {len(path)} waypoints")
        
        # Smooth path
        smoothed_path = smooth_path(path, obstacle_map, smoothing_aggressiveness)
        print(f"Smoothed path has {len(smoothed_path)} waypoints")
        
        # Execute path
        robot = RobotController()
        for x, y in smoothed_path:
            robot.move_to_position(x, y)
        
        # Verify final position
        final_error = sqrt((goal_pos[0] - robot.x_pos)**2 + (goal_pos[1] - robot.y_pos)**2)
        print(f"\nFinal position error: {final_error:.4f} meters")
        
        # Generate outputs
        visualize_path(robot, obstacles)
        print_command_log(robot)
    else:
        print("No valid path found!")