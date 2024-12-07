import json
import random
import numpy as np
import itertools
import os
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

class ItemPlacer:
    def __init__(self, table_width=1.0, table_depth=0.8, table_height=0.7):
        """
        Initialize item placer with table dimensions and tracking.

        :param table_width: Width of the table (x-axis)
        :param table_depth: Depth of the table (y-axis)
        :param table_height: Height of the table surface
        """
        self.table_width = table_width
        self.table_depth = table_depth
        self.table_height = table_height
        self.placed_items = []

        # Item size estimates (approximate bounding box diagonals)
        self.item_sizes = {
            '002_master_chef_can': 0.1,
            '044_flat_screwdriver': 0.2,
            '072-e_toy_airplane': 0.3,
            '048_hammer': 0.4,
            '073-e_lego_duplo': 0.15
        }

    def check_collision(self, x, y, item_model):
        """
        Check if the proposed position collides with already placed items.

        :param x: X position to check
        :param y: Y position to check
        :param item_model: Model ID of the item
        :return: Boolean indicating if placement is clear
        """
        item_size = self.item_sizes.get(item_model, 0.2)

        for placed_x, placed_y, placed_size, _ in self.placed_items:
            # Calculate distance between item centers
            dist = np.sqrt((x - placed_x)**2 + (y - placed_y)**2)

            # If distance is less than sum of their radii, it's a collision
            if dist < (item_size + placed_size) / 2:
                return False
        return True

    def place_item(self, item_model):
        """
        Place an item on the table with no collisions.

        :param item_model: Model ID of the item
        :return: [x, y, z] position
        """
        item_size = self.item_sizes.get(item_model, 0.2)

        # Maximum attempts to place an item
        max_attempts = 100
        for _ in range(max_attempts):
            # Random x and y within table bounds, accounting for item size
            x = random.uniform(
                -self.table_width/2 + item_size/2,
                self.table_width/2 - item_size/2
            )
            y = random.uniform(
                -self.table_depth/2 + item_size/2,
                self.table_depth/2 - item_size/2
            )

            # Check for collisions
            if self.check_collision(x, y, item_model):
                # Store placed item with its position and size
                self.placed_items.append((x, y, item_size, item_model))
                return [x, y, self.table_height]

        raise ValueError(f"Could not place item {item_model} without collision")

def is_kitchen_item(model_id):
    """
    Determine if a model is a kitchen-related item.
    """
    kitchen_keywords = [
        'master_chef_can',
        'mug',
        'cup',
        'plate',
        'bowl',
        'spatula',
        'knife',
        'fork',
        'spoon',
        'measuring'
    ]
    return any(keyword in model_id.lower() for keyword in kitchen_keywords)

def generate_random_rep_points(num_points=16):
    """
    Generate random representative points.
    """
    return [
        [
            random.uniform(-0.3, 0.3),  # x
            random.uniform(-0.3, 0.3),  # y
            random.uniform(-0.1, 0.3)   # z
        ] for _ in range(num_points)
    ]

def visualize_scene(scene, save_path=None):
    """
    Visualize a scene's item placements.

    :param scene: Scene dictionary with actor placements
    :param save_path: Optional path to save the visualization
    """
    # Create a new figure
    plt.figure(figsize=(10, 8))

    # Plot table
    table_width = 1.0
    table_depth = 0.8
    plt.gca().add_patch(Rectangle(
        (-table_width/2, -table_depth/2),
        table_width, table_depth,
        fill=False,
        edgecolor='brown',
        linewidth=2
    ))

    # Color mapping for different items
    color_map = {
        '002_master_chef_can': 'red',
        '044_flat_screwdriver': 'blue',
        '072-e_toy_airplane': 'green',
        '048_hammer': 'purple',
        '073-e_lego_duplo': 'orange'
    }

    # Plot items
    for actor in scene['actors']:
        x, y, _ = actor['pose'][:3]
        model_id = actor['model_id']

        # Get color, default to gray if not in color map
        color = color_map.get(model_id, 'gray')

        # Plot item as a circle
        item_size = ItemPlacer().item_sizes.get(model_id, 0.2)
        plt.scatter(x, y, c=color, s=500, alpha=0.7, edgecolors='black')

        # Add model ID as text
        plt.text(x, y, model_id.split('_')[0],
                 horizontalalignment='center',
                 verticalalignment='center',
                 color='white',
                 fontweight='bold')

    plt.title('Item Placement on Table')
    plt.xlabel('X Position (m)')
    plt.ylabel('Y Position (m)')
    plt.axis('equal')
    plt.grid(True, linestyle='--', alpha=0.7)

    # Adjust plot limits to match table
    plt.xlim(-table_width/2 - 0.1, table_width/2 + 0.1)
    plt.ylim(-table_depth/2 - 0.1, table_depth/2 + 0.1)

    # Save or show the plot
    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()

def generate_kitchen_item_scenes(input_file, max_scenes=10, max_items=3):
    """
    Generate new scenes with kitchen item permutations.
    """
    # Load original data to get reference items
    with open(input_file, 'r') as f:
        original_data = json.load(f)

    # Find kitchen items
    kitchen_items = [
        actor['model_id'] for scene in original_data
        for actor in scene['actors']
        if is_kitchen_item(actor['model_id'])
    ]

    # Remove duplicates
    kitchen_items = list(set(kitchen_items))

    # Generate scenes
    new_scenes = []

    # Generate permutations
    for r in range(1, min(max_items, len(kitchen_items)) + 1):
        for perm in itertools.permutations(kitchen_items, r):
            # Reset item placer for each scene
            item_placer = ItemPlacer()

            # Create a new scene
            scene = {
                "actors": [
                    {
                        "model_id": item,
                        "pose": [
                            *item_placer.place_item(item),  # x, y, z position
                            0, 0, 0, 1  # zero rotation (x, y, z, w)
                        ],
                        "scale": 1.0,  # Default scale
                        "rep_pts": generate_random_rep_points()
                    } for item in perm
                ]
            }
            new_scenes.append(scene)

            # Stop if we've reached max scenes
            if len(new_scenes) >= max_scenes:
                return new_scenes

    return new_scenes

def main(input_file='ycb_train_5k.json', output_file='kitchen_item_permutations.json'):
    """
    Main function to generate kitchen item permutation scenes.
    """
    # Seed for reproducibility
    random.seed(42)
    np.random.seed(42)

    # Generate scenes
    kitchen_scenes = generate_kitchen_item_scenes(input_file)

    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_file) or '.', exist_ok=True)

    # Save to JSON
    with open(output_file, 'w') as f:
        json.dump(kitchen_scenes, f, indent=2)

    # Visualize the first few scenes
    for i, scene in enumerate(kitchen_scenes[:3]):
        visualize_scene(scene, save_path=f'img/scene_{i+1}_placement.png')

    print(f"Generated {len(kitchen_scenes)} scenes with kitchen item permutations in {output_file}")
    print("Visualizations saved as PNG files")

if __name__ == "__main__":
    main()