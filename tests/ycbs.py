import json
import random
import torch
import numpy as np
import itertools
import os
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from transforms3d.quaternions import mat2quat

from mani_skill import PACKAGE_ASSET_DIR, ASSET_DIR
from mani_skill.utils.building.actors.ycb import _load_ycb_dataset, YCB_DATASET
from mani_skill.utils.io_utils import load_json
# euler_to_quat
from mani_skill.utils.geometry.rotation_conversions import euler_angles_to_matrix, axis_angle_to_quaternion
from transforms3d.euler import euler2quat, quat2mat, euler2mat, quat2euler

kitchen_keywords = [
    'pitcher',
    'mug',
    'cup',
    'plate',
    'bowl',
    'spatula',
    'knife',
    'fork',
    'spoon',
]

YCB_ASSETS_DIR = ASSET_DIR / "assets/mani_skill2_ycb/models"

TABLE_POSE = {
    "position": [-0.12, 0, -0.6306],
    "rotation": euler2quat(0, 0, np.pi / 2).tolist()
}

ycb_dataset = {}
class ItemPlacer:
    def __init__(self,
                 items,
                 table_dimensions : tuple[float, float, float] = (1.6579739, 0.829081, 0.6306),
                 ):
        """
        Initialize item placer with table dimensions and tracking.
        items: dict of item model IDs and their data (aabbs, densities, etc.)
        table_dimensions: (width, depth, height) of the table
        """
        self.table_width = table_dimensions[0]
        self.table_depth = table_dimensions[1]
        self.table_height = table_dimensions[2]
        self.placed_items = []

        # Item size estimates (approximate bounding box diagonals)
        self.items = items

    def check_aabb_overlap(self, center1, aabb1, center2, aabb2, rotation1, rotation2):
        #convert to bounding circle
        radius1 = np.linalg.norm(np.array([aabb1['max'][0] - aabb1['min'][0], aabb1['max'][1] - aabb1['min'][1]]) / 2.)
        radius2 = np.linalg.norm(np.array([aabb2['max'][0] - aabb2['min'][0], aabb2['max'][1] - aabb2['min'][1]]) / 2.)
        dist = np.linalg.norm(center1[:2] - center2[:2])
        print(dist <= radius1 + radius2)

        return dist <= radius1 + radius2

        # TODO: check if this function correct
        # # Compute center and half-extents
        # half_extents1 = np.array([aabb1['max'][0] - aabb1['min'][0], aabb1['max'][1] - aabb1['min'][1]]) / 2
        # half_extents2 = np.array([aabb2['max'][0] - aabb2['min'][0], aabb2['max'][1] - aabb2['min'][1]]) / 2
        #
        # # Compute vertices of both AABBs
        # def get_aabb_vertices(center, extents, rot):
        #     # Compute corner points without rotation
        #     corners = np.array([
        #         [-extents[0], -extents[1]],
        #         [extents[0], -extents[1]],
        #         [extents[0], extents[1]],
        #         [-extents[0], extents[1]]
        #     ])
        #
        #     # Rotate corners
        #     rotated_corners = (rot @ corners.T).T
        #
        #     # Translate rotated corners
        #     return rotated_corners + center[:2]
        #
        # # Get vertices
        # vertices1 = get_aabb_vertices(center1, half_extents1, rotation1[:2, :2])
        # vertices2 = get_aabb_vertices(center2, half_extents2, rotation2[:2, :2])
        #
        # # Project vertices onto separating axes
        # def project_vertices(vertices, axis):
        #     return [np.dot(v, axis) for v in vertices]
        #
        # # Axes to test (box normals)
        # axes = [
        #     np.array([1, 0]),  # x-axis
        #     np.array([0, 1]),  # y-axis
        # ]
        #
        # # Add rotated axes
        # def get_rotated_normals(rot):
        #     return [
        #         rot @ np.array([1, 0]),
        #         rot @ np.array([0, 1])
        #     ]
        #
        # axes.extend(get_rotated_normals(rotation1[:2, :2]))
        # axes.extend(get_rotated_normals(rotation2[:2, :2]))
        #
        # # Test each axis
        # for axis in axes:
        #     # Project vertices onto this axis
        #     proj1 = project_vertices(vertices1, axis)
        #     proj2 = project_vertices(vertices2, axis)
        #
        #     # Check for separation
        #     if max(proj1) < min(proj2) or max(proj2) < min(proj1):
        #         return False
        #
        # return True

    def check_collision(self, x, y, yaw, item_model):
        """
        Check if the proposed position collides with already placed items.

        :param x: X position to check
        :param y: Y position to check
        :param item_model: Model ID of the item
        :return: Boolean indicating if placement is clear
        """
        aabb_item = self.items[item_model]['aabb']
        center_item = np.array([x, y, self.table_height + (aabb_item['max'][2] - aabb_item['min'][2])/2])
        rotation_item = euler_angles_to_matrix(torch.tensor([0, 0, yaw]), 'ZYX').cpu().numpy()
        for placed_x, placed_y, placed_yaw, placed_aabb, _ in self.placed_items:
            rotation_placed = euler_angles_to_matrix(torch.tensor([[0, 0, placed_yaw]]), 'ZYX').squeeze(0).cpu().numpy()
            center_placed = np.array([placed_x, placed_y, self.table_height + (placed_aabb['max'][2] - placed_aabb['min'][2])/2])
            if self.check_aabb_overlap(center_item, aabb_item, center_placed, placed_aabb, rotation_item, rotation_placed):
                return True
        #     x_dist = abs(x - placed_x)
        #     y_dist = abs(y - placed_y)
        #     if (x_dist < ((aabb_item['max'][0] - aabb_item['min'][0] + placed_size['max'][0] - placed_size['min'][0]) / 2)) and \
        #         (y_dist < ((aabb_item['max'][1] - aabb_item['min'][1] + placed_size['max'][1] - placed_size['min'][1]) / 2)):
        #         return False
        return False

    def place_item(self, item_model):
        """
        Place an item on the table with no collisions.

        :param item_model: Model ID of the item
        :return: [x, y, z] position
        """
        item_aabb = self.items[item_model]['aabb']

        # Maximum attempts to place an item
        max_attempts = 100
        for _ in range(max_attempts):
            # Random x and y within table bounds, accounting for item size
            # TODO: consider item size and orientation for better placement (e.g., x is not based only on item width,

            x = random.uniform(
                -self.table_width/2 + (item_aabb['max'][0] - item_aabb['min'][0]) + TABLE_POSE["position"][0] + 0.015,
                self.table_width/2 - (item_aabb['max'][0] - item_aabb['min'][0])  + TABLE_POSE["position"][0] - 0.015
            )
            y = random.uniform(
                -self.table_depth/2 + (item_aabb['max'][1] - item_aabb['min'][1]) + TABLE_POSE["position"][1] + 0.015,
                self.table_depth/2 - (item_aabb['max'][1] - item_aabb['min'][1]) - TABLE_POSE["position"][1] - 0.015
            )

            yaw = random.uniform(0, 2*np.pi)

            # Check for collisions
            if not self.check_collision(x, y, yaw, item_model):
                print(f"Placed item {item_model} at ({x:.2f}, {y:.2f})")
                # Store placed item with its position and size
                self.placed_items.append((x, y, yaw, item_aabb, item_model))
                return [x, y, self.table_height + (item_aabb['max'][2] - item_aabb['min'][2])/2, yaw]

        # raise ValueError(f"Could not place item {item_model} without collision")
        return False

def is_kitchen_item(model_id):
    """
    Determine if a model is a kitchen-related item.
    """
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
    table_width = 1.6579739
    table_depth = 0.829081
    plt.gca().add_patch(Rectangle(
        (TABLE_POSE['position'][0] - table_width/2, TABLE_POSE['position'][1] - table_depth/2),
        table_width, table_depth,
        angle=np.rad2deg(np.pi / 2),
        rotation_point='center',
        fill=False,
        edgecolor='brown',
        linewidth=2
    ))

    # Color mapping for different items
    color_options = ['red', 'blue', 'green', 'orange', 'purple', 'cyan', 'magenta', 'yellow']
    color_map = {
        actor['model_id']: color_options[i % len(color_options)]
        for i, actor in enumerate(scene['actors'])
    }

    # Plot items
    for actor in scene['actors']:
        x, y, _ = actor['pose'][:3]
        model_id = actor['model_id']

        # Get color, default to gray if not in color map
        color = color_map.get(model_id, 'gray')

        # Plot item as a circle
        item = ycb_dataset['model_data'][model_id]
        item_size = item['bbox']
        quaternion = actor['pose'][3:]
        angle = quat2euler(quaternion, 'szyx')[0]

        plt.gca().add_patch(plt.Rectangle(
            (x - (item_size['max'][0] - item_size['min'][0])/2, y - (item_size['max'][1] - item_size['min'][1])/2),
            item_size['max'][0] - item_size['min'][0],
            item_size['max'][1] - item_size['min'][1],
            angle=np.rad2deg(angle),
            rotation_point='center',
            fill=True,
            color=color,
            alpha=0.5
        ))
        # Add model ID as text
        plt.text(x, y, model_id,
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
    # plt.xlim(-table_width/2 - 0.1, table_width/2 + 0.1)
    # plt.ylim(-table_depth/2 - 0.1, table_depth/2 + 0.1)

    # Save or show the plot
    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()

def generate_kitchen_item_scenes(max_scenes=40, min_items=10, max_items=15):
    """
    Generate new scenes with kitchen item permutations.
    """
    kitchen_ycb_dirs = {}
    for model_id in ycb_dataset['model_data']:
        if is_kitchen_item(model_id):
            metadata = ycb_dataset['model_data'][model_id]
            model_scales = metadata.get("scales", [1.0])
            scale = model_scales[0]
            aabb = metadata["bbox"]
            density = metadata.get("density", 1000)
            physical_material = None
            model_dir = YCB_ASSETS_DIR / model_id
            kitchen_ycb_dirs[model_id] = {
                'aabb': aabb,
                'scale': scale,
                'density': density,
                'physical_material': physical_material,
                'model_dir': model_dir
            }

    # Generate scenes
    new_scenes = []

    # Generate permutations
    # for r in range(min_items, max_items + 1):
    for r in range(40):
        # get random r items
        # perm = np.random.choice(list(kitchen_ycb_dirs.keys()), r, replace=False)
        perm = np.random.choice(list(kitchen_ycb_dirs.keys()), 3, replace=False)
        perm = perm.tolist()
        perm.append("029_plate")
        # for perm in itertools.permutations(kitchen_ycb_dirs.keys(), r):
            # Reset item placer for each scene
        item_placer = ItemPlacer(kitchen_ycb_dirs)
        actors = []
        for item in perm:
            # Attempt to place item
            placing = item_placer.place_item(item)
            if not placing:
                continue
            yaw = placing[3]
            placing = placing[:3]
            # rotate to table
            placing = np.array(placing)
            placing = quat2mat(TABLE_POSE['rotation']) @ placing + np.array(TABLE_POSE['position'])
            placing = placing.tolist()
            quat = euler2mat(0, 0, yaw)
            quat = quat2mat(TABLE_POSE['rotation']) @ quat
            quat = mat2quat(quat).tolist()
            placing.extend(quat)
            actors.append({
                "model_id": item,
                "pose": [
                    *placing,  # x, y, z, w, x, y, z
                ],
                "scale": 1.0,  # Default scale
                "rep_pts": generate_random_rep_points()
            })
        scene = {
            "actors": actors
        }
        new_scenes.append(scene)

        # Stop if we've reached max scenes
        if len(new_scenes) >= max_scenes:
            return new_scenes

    return new_scenes

def main(output_file='kitchen_item_permutations.json'):
    """
    Main function to generate kitchen item permutation scenes.
    """
    # Seed for reproducibility
    random.seed(29)
    np.random.seed(29)

    # Load YCB dataset
    global ycb_dataset
    ycb_dataset = {
        "model_data": load_json(ASSET_DIR / "assets/mani_skill2_ycb/info_raw.json"),
    }

    # Generate scenes
    kitchen_scenes = generate_kitchen_item_scenes()

    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_file) or '.', exist_ok=True)

    # Save to JSON
    with open(output_file, 'w') as f:
        json.dump(kitchen_scenes, f, indent=2)

    # Visualize the first few scenes
    for i, scene in enumerate(kitchen_scenes):
        visualize_scene(scene, save_path=f'img/scene_{i+1}_placement.png')

    print(f"Generated {len(kitchen_scenes)} scenes with kitchen item permutations in {output_file}")
    print("Visualizations saved as PNG files")

if __name__ == "__main__":
    main()