import torch

from learner.hybrid_zertz_ai import HybridZertzAI


# Assuming the previous classes (C6ConvLayer, CNNFeatureExtractor, GNNFeatureExtractor, HybridZertzAI) are already defined

def label_to_grid_mapping(board_size=48):
    """
    Maps board labels to grid coordinates based on board size.
    Args:
        board_size (str): '37-ring' or '48-ring'.
    Returns:
        label_to_coord (dict): Mapping from label (str) to (row, col) tuple.
        height (int): Number of rows in the grid.
        width (int): Number of columns in the grid.
    """
    label_to_coord = {}

    if board_size == 37:
        # Define the mapping based on the 37-ring board structure
        labels = [
            ['A4', 'B5', 'C6', 'D7', 'E8'],
            ['A3', 'B4', 'C5', 'D6', 'E7'],
            ['A2', 'B3', 'C4', 'D5', 'E6', 'F7'],
            ['A1', 'B2', 'C3', 'D4', 'E5', 'F6', 'G7'],
            ['B1', 'C2', 'D3', 'E4', 'F5', 'G6'],
            ['C1', 'D2', 'E3', 'F4', 'G5'],
            ['D1', 'E2', 'F3', 'G4']
        ]
        height = 7
        width = 9  # Adjust based on actual grid
    elif board_size == 48:
        # Define the mapping based on the 48-ring board structure
        labels = [
            ['A5', 'B6', 'C7', 'D8', 'E9'],
            ['A4', 'B5', 'C6', 'D7', 'E8', 'F9'],
            ['A3', 'B4', 'C5', 'D6', 'E7', 'F8', 'G9'],
            ['A2', 'B3', 'C4', 'D5', 'E6', 'F7', 'G8', 'H9'],
            ['A1', 'B2', 'C3', 'D4', 'E5', 'F6', 'G7', 'H8'],
            ['B1', 'C2', 'D3', 'E4', 'F5', 'G6', 'H7'],
            ['C1', 'D2', 'E3', 'F4', 'G5', 'H6'],
            ['D1', 'E2', 'F3', 'G4', 'H5']
        ]
        height = 8
        width = 9  # Adjust based on actual grid
    else:
        raise ValueError("Unsupported board size. Choose '37' or '48'.")

    for row_idx, row in enumerate(labels):
        for col_idx, label in enumerate(row):
            label_to_coord[label] = (row_idx, col_idx)

    return label_to_coord, height, width


def grid_to_graph_no_mask(height, width, label_to_coord, board_size=48):
    """
    Converts a hexagonal grid to a graph structure without using a mask.
    Inactive rings are treated as removed with no edges.
    Args:
        height (int): Number of rows in the grid.
        width (int): Number of columns in the grid.
        label_to_coord (dict): Mapping from label to (row, col) coordinates.
        board_size (str): '37-ring' or '48-ring'.
    Returns:
        edge_index (torch.LongTensor): Edge connections in COO format.
        active_nodes (set): Set of (row, col) tuples indicating active nodes.
    """
    edge_index = []
    active_nodes = set()

    if board_size == '37-ring':
        # Define active labels for 37-ring board
        active_labels = [
            'A4', 'B5', 'C6', 'D7', 'E8',
            'A3', 'B4', 'C5', 'D6', 'E7',
            'A2', 'B3', 'C4', 'D5', 'E6', 'F7',
            'A1', 'B2', 'C3', 'D4', 'E5', 'F6', 'G7',
            'B1', 'C2', 'D3', 'E4', 'F5', 'G6',
            'C1', 'D2', 'E3', 'F4', 'G5',
            'D1', 'E2', 'F3', 'G4'
        ]
    elif board_size == '48-ring':
        # Define active labels for 48-ring board (all are active)
        active_labels = [
            'A5', 'B6', 'C7', 'D8', 'E9',
            'A4', 'B5', 'C6', 'D7', 'E8', 'F9',
            'A3', 'B4', 'C5', 'D6', 'E7', 'F8', 'G9',
            'A2', 'B3', 'C4', 'D5', 'E6', 'F7', 'G8', 'H9',
            'A1', 'B2', 'C3', 'D4', 'E5', 'F6', 'G7', 'H8',
            'B1', 'C2', 'D3', 'E4', 'F5', 'G6', 'H7',
            'C1', 'D2', 'E3', 'F4', 'G5', 'H6',
            'D1', 'E2', 'F3', 'G4', 'H5'
        ]
    else:
        raise ValueError("Unsupported board size. Choose '37-ring' or '48-ring'.")

    # Collect active nodes
    for label, (row, col) in label_to_coord.items():
        if label in active_labels:
            active_nodes.add((row, col))

    # Construct edge_index by connecting only active nodes
    for row, col in active_nodes:
        node = row * width + col
        neighbors = get_hex_neighbors(row, col, height, width)
        for nr, nc in neighbors:
            if (nr, nc) in active_nodes:
                neighbor_node = nr * width + nc
                edge_index.append([node, neighbor_node])

    # Remove duplicate edges
    edge_index = list(set(tuple(edge) for edge in edge_index))

    if len(edge_index) == 0:
        edge_index = torch.empty((2, 0), dtype=torch.long)
    else:
        edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()

    return edge_index, active_nodes


def get_hex_neighbors(row, col, height, width):
    """
    Retrieves the neighbors of a cell in a hexagonal grid based on row parity.
    Args:
        row (int): Current row.
        col (int): Current column.
        height (int): Total rows.
        width (int): Total columns.
    Returns:
        neighbors (list of tuples): List of (row, col) tuples representing neighbors.
    """
    # Define neighbor offsets for even and odd rows (offset grid)
    if row % 2 == 0:
        # Even row
        directions = [(-1, 0), (-1, 1), (0, -1), (0, 1), (1, 0), (1, 1)]
    else:
        # Odd row
        directions = [(-1, -1), (-1, 0), (0, -1), (0, 1), (1, -1), (1, 0)]

    neighbors = []
    for dr, dc in directions:
        nr, nc = row + dr, col + dc
        if 0 <= nr < height and 0 <= nc < width:
            neighbors.append((nr, nc))
    return neighbors


def create_input_tensor(marbles, ring_presence, label_to_coord, height, width):
    """
    Creates the input tensor based on the current game state.
    Args:
        marbles (list of tuples): List of (marble_type, position_label).
        ring_presence (list of str): List of position labels indicating present rings.
        label_to_coord (dict): Mapping from label to (row, col) coordinates.
        height (int): Number of rows in the grid.
        width (int): Number of columns in the grid.
    Returns:
        x (torch.Tensor): Input tensor of shape [1, 4, height, width].
    """
    num_feature_planes = 4  # ring, w, g, b
    x = torch.zeros(1, num_feature_planes, height, width)

    # Populate ring presence
    for label in ring_presence:
        if label not in label_to_coord:
            continue  # Skip invalid labels
        row, col = label_to_coord[label]
        x[0, 0, row, col] = 1  # Ring plane

    # Populate marble planes
    for marble_type, label in marbles:
        if label not in label_to_coord:
            continue  # Skip invalid labels
        row, col = label_to_coord[label]
        if x[0, 0, row, col] == 0:
            continue  # Cannot place marbles on removed rings
        if marble_type == 'w':
            x[0, 1, row, col] = 1  # White plane
        elif marble_type == 'g':
            x[0, 2, row, col] = 1  # Gray plane
        elif marble_type == 'b':
            x[0, 3, row, col] = 1  # Black plane

    return x


def create_captured_counts(captured_p1, captured_p2):
    """
    Creates a tensor representing the captured marble counts for both players.
    Args:
        captured_p1 (dict): Captured counts for Player 1, e.g., {'w': 5, 'g': 3, 'b': 2}.
        captured_p2 (dict): Captured counts for Player 2, e.g., {'w': 4, 'g': 6, 'b': 1}.
    Returns:
        counts (torch.Tensor): Tensor of shape [1, 6] representing captured counts.
    """
    counts = torch.tensor([
        captured_p1.get('w', 0),
        captured_p1.get('g', 0),
        captured_p1.get('b', 0),
        captured_p2.get('w', 0),
        captured_p2.get('g', 0),
        captured_p2.get('b', 0)
    ], dtype=torch.float32).unsqueeze(0)  # Shape: [1, 6]

    # Normalize counts (optional, depending on training strategy)
    # For example, assuming a maximum of 10 captures per type
    counts = counts / 10.0

    return counts


def example_usage(board_size='37-ring'):
    """
    Example usage of the HybridZertzAI network with dummy data based on the provided board structure.
    Args:
        board_size (str): '37-ring' or '48-ring'.
    """
    # Get board dimensions and label mappings
    label_to_coord, height, width = label_to_grid_mapping(board_size)

    # Convert grid to graph
    edge_index, active_nodes = grid_to_graph_no_mask(height, width, label_to_coord, board_size)

    # Define number of moves (active rings)
    num_moves = len(active_nodes)

    # Instantiate the hybrid network
    model = HybridZertzAI(
        in_channels=4,  # Four channels: ring, w, g, b
        cnn_hidden=32,
        gnn_hidden=64,
        num_moves=num_moves,
        height=height,
        width=width,
        num_captured_features=6  # p1_w, p1_g, p1_b, p2_w, p2_g, p2_b
    )

    # Create dummy input data
    # Example marbles: list of (marble_type, position_label)
    marbles = [
        ('w', 'A4'),
        ('g', 'B5'),
        ('b', 'C6'),
        ('w', 'D7'),
        ('g', 'E8'),
        # ... more marbles as needed
    ]

    # Example ring presence
    ring_presence = ['A4', 'B5', 'C6', 'D7', 'E8', 'A3', 'B4', 'C5', 'D6', 'E7', 'A2', 'B3', 'C4', 'D5', 'E6', 'F7',
                     'A1', 'B2', 'C3', 'D4', 'E5', 'F6', 'G7', 'B1', 'C2', 'D3', 'E4', 'F5', 'G6', 'C1', 'D2',
                     'E3', 'F4', 'G5', 'D1', 'E2', 'F3', 'G4']  # Adjust based on actual game state

    x = create_input_tensor(marbles, ring_presence, label_to_coord, height, width)  # Shape: [1, 4, height, width]

    # Example captured counts
    captured_p1 = {'w': 5, 'g': 3, 'b': 2}
    captured_p2 = {'w': 4, 'g': 6, 'b': 1}
    captured_counts = create_captured_counts(captured_p1, captured_p2)  # Shape: [1, 6]

    # Determine if captures are possible (dummy logic for illustration)
    # In a real scenario, this should be based on game rules
    # For example purposes, let's assume no captures are possible
    captures_available = False  # Change to True to simulate captures

    # Set action_type based on captures availability
    action_type = 'cap' if captures_available else 'put'

    # Forward pass
    policy, value = model(x, edge_index, captured_counts, action_type=action_type)

    print(f"Board Size: {board_size}")
    print(f"Action Type: {action_type.upper()}")
    print(f"Policy shape: {policy.shape}")  # Expected: [1, num_moves]
    print(f"Value shape: {value.shape}")  # Expected: [1, 1]")
    print(f"Policy: {policy}")
    print(f"Value: {value}")