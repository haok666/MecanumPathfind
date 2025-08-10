def find_min_distance_node(unvisited_nodes, distances):
    """
    Finds the unvisited node with the smallest known distance
    """
    min_distance = float('infinity')
    min_node = None
    
    for node in unvisited_nodes:
        if distances[node] < min_distance:
            min_distance = distances[node]
            min_node = node
    
    return min_node


def dijkstra_algorithm(graph, start_node):
    """
    Calculates shortest distances from start_node to all other nodes
    using Dijkstra's algorithm without heapq
    """
    
    # Initialize distances dictionary
    distances = {}
    for node in graph:
        distances[node] = float('infinity')
    distances[start_node] = 0
    
    # Track visited and unvisited nodes
    unvisited_nodes = list(graph.keys())
    visited_nodes = []
    
    # Dictionary to store the path
    previous_nodes = {}
    
    # Main algorithm loop
    while len(unvisited_nodes) > 0:
        
        # Find the unvisited node with smallest distance
        current_node = find_min_distance_node(unvisited_nodes, distances)
        
        # If all remaining nodes are unreachable, break
        if distances[current_node] == float('infinity'):
            break
        
        # Visit all neighbors of current node
        for neighbor, edge_weight in graph[current_node].items():
            
            # Calculate tentative distance
            tentative_distance = distances[current_node] + edge_weight
            
            # Update distance if shorter path found
            if tentative_distance < distances[neighbor]:
                distances[neighbor] = tentative_distance
                previous_nodes[neighbor] = current_node
        
        # Mark current node as visited
        unvisited_nodes.remove(current_node)
        visited_nodes.append(current_node)
    
    return distances, previous_nodes


def reconstruct_path(previous_nodes, start_node, target_node):
    """
    Reconstructs the shortest path from start_node to target_node
    using the previous_nodes dictionary
    """
    path = []
    current_node = target_node
    
    # Work backwards from target to start
    while current_node != start_node:
        path.insert(0, current_node)
        current_node = previous_nodes[current_node]
    
    # Add start node
    path.insert(0, start_node)
    
    return path


def print_results(distances, previous_nodes, start_node):
    """
    Prints the algorithm results in a readable format
    """
    print(f"\nShortest distances from node {start_node}:")
    for node, distance in distances.items():
        print(f"To {node}: {distance}")
    
    print("\nPrevious nodes in shortest paths:")
    for node, prev in previous_nodes.items():
        print(f"{node} <- {prev}")


# Example graph (adjacency list format)
example_graph = {
    'A': {'B': 6, 'D': 1},
    'B': {'A': 6, 'D': 2, 'E': 2},
    'C': {'B': 5, 'E': 5},
    'D': {'A': 1, 'B': 2, 'E': 1},
    'E': {'B': 2, 'D': 1, 'C': 5}
}

# Run the algorithm
start_node = 'A'
shortest_distances, previous_nodes = dijkstra_algorithm(example_graph, start_node)

# Display results
print_results(shortest_distances, previous_nodes, start_node)

# Example path reconstruction
target_node = 'C'
shortest_path = reconstruct_path(previous_nodes, start_node, target_node)
print(f"\nShortest path from {start_node} to {target_node}: {shortest_path}")