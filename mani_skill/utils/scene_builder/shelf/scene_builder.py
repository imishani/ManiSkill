from __future__ import annotations

from typing import List

import numpy as np
import sapien
import sapien.render
import torch
from transforms3d.euler import euler2quat

from mani_skill.agents.robots.ridgeback_ur10e import RIDGEBACK_WHEELS_COLLISION_BIT
from mani_skill.utils.building.ground import build_ground
from mani_skill.utils.scene_builder import SceneBuilder


class ShelfSceneBuilder(SceneBuilder):
    """
    Builds a scene with a freestanding shelf unit and a small delivery cart beside it.

    Coordinate convention
    ---------------------
    - Ground at z = 0.
    - Shelf front face at x = SHELF_FRONT_X, extending in +x (depth direction).
    - Shelf is centered at y = 0.
    - Robot approaches from x < SHELF_FRONT_X, facing +x.

    The shelf has a configurable number of intermediate horizontal boards.
    ``target_shelf_surface_z`` gives the z-height of the top surface of the
    board chosen as the "working" level for the retrieval task.

    No back panel is built by default; set ``add_back_panel=True`` to include one.

    Delivery cart
    -------------
    A small table is placed to the right (positive y) of the shelf at a
    reachable height.  Its top-surface z is stored in ``cart_surface_z``.
    """

    # ---------------------------------------------------------------------------
    # Geometry parameters – override in subclasses or pass to build()
    # ---------------------------------------------------------------------------

    # Shelf
    SHELF_FRONT_X: float = 1.0       # x-coord of the shelf's front face
    SHELF_DEPTH: float = 0.45        # x extent of shelf interior
    SHELF_WIDTH: float = 1.2         # y extent (total, including side panels)
    SHELF_TOTAL_HEIGHT: float = 1.3  # z extent (total)
    PANEL_THICKNESS: float = 0.025   # thickness of all structural boards/panels

    # Which intermediate level (0-indexed from bottom) is the "target" level.
    # E.g. with NUM_INTERMEDIATE_LEVELS=1, there is only level 0.
    NUM_INTERMEDIATE_LEVELS: int = 1
    TARGET_LEVEL_IDX: int = 0        # index into the intermediate levels

    # Delivery cart
    CART_WIDTH: float = 0.6          # x extent
    CART_DEPTH: float = 0.5          # y extent
    CART_HEIGHT: float = 0.75        # height of cart surface above ground
    CART_Y_GAP: float = 0.3          # gap between shelf right panel and cart

    def build(
        self,
        add_back_panel: bool = False,
    ):
        """Build all static/kinematic scene geometry.

        Parameters
        ----------
        add_back_panel : bool
            Whether to close the back of the shelf with a solid panel.
        """
        pt = self.PANEL_THICKNESS
        sw = self.SHELF_WIDTH
        sd = self.SHELF_DEPTH
        sh = self.SHELF_TOTAL_HEIGHT
        sfx = self.SHELF_FRONT_X

        # Useful derived quantities
        shelf_cx = sfx + sd / 2           # x-center of shelf body
        inner_half_width = (sw - 2 * pt) / 2  # half-width of the interior opening

        # ------------------------------------------------------------------
        # Ground plane
        # ------------------------------------------------------------------
        floor_width = 500 if self.scene.parallel_in_single_scene else 100
        self.ground = build_ground(self.scene, floor_width=floor_width, altitude=0)

        # ------------------------------------------------------------------
        # Shelf parts (all kinematic)
        # ------------------------------------------------------------------
        wood_mat = sapien.render.RenderMaterial(base_color=[0.65, 0.45, 0.25, 1.0])
        shelf_parts: List = []

        def _build_box(half_size, cx, cy, cz, name):
            b = self.scene.create_actor_builder()
            b.add_box_collision(half_size=half_size)
            b.add_box_visual(half_size=half_size, material=wood_mat)
            b.initial_pose = sapien.Pose(p=[cx, cy, cz])
            shelf_parts.append(b.build_kinematic(name=name))

        # Side panels (left: -y side, right: +y side)
        _build_box(
            half_size=(sd / 2 + pt / 2, pt / 2, sh / 2),
            cx=sfx + sd / 2,
            cy=-(sw / 2 - pt / 2),
            cz=sh / 2,
            name="shelf_left_panel",
        )
        _build_box(
            half_size=(sd / 2 + pt / 2, pt / 2, sh / 2),
            cx=sfx + sd / 2,
            cy=sw / 2 - pt / 2,
            cz=sh / 2,
            name="shelf_right_panel",
        )

        # Bottom board (sits on ground)
        _build_box(
            half_size=(sd / 2, inner_half_width, pt / 2),
            cx=shelf_cx,
            cy=0.0,
            cz=pt / 2,
            name="shelf_bottom_board",
        )

        # Top board
        _build_box(
            half_size=(sd / 2, inner_half_width, pt / 2),
            cx=shelf_cx,
            cy=0.0,
            cz=sh - pt / 2,
            name="shelf_top_board",
        )

        # Intermediate shelf boards
        # Evenly spaced between bottom and top boards.
        n = self.NUM_INTERMEDIATE_LEVELS
        level_surface_zs: List[float] = []
        for i in range(n):
            # bottom of board at evenly spaced interval
            board_bottom_z = (i + 1) * sh / (n + 1)
            board_cz = board_bottom_z + pt / 2
            level_surface_zs.append(board_bottom_z + pt)
            _build_box(
                half_size=(sd / 2, inner_half_width, pt / 2),
                cx=shelf_cx,
                cy=0.0,
                cz=board_cz,
                name=f"shelf_board_level_{i}",
            )

        if add_back_panel:
            _build_box(
                half_size=(pt / 2, inner_half_width, sh / 2),
                cx=sfx + sd + pt / 2,
                cy=0.0,
                cz=sh / 2,
                name="shelf_back_panel",
            )

        self.shelf_parts = shelf_parts
        self.level_surface_zs = level_surface_zs

        # ------------------------------------------------------------------
        # Delivery cart (small table to the right of the shelf)
        # ------------------------------------------------------------------
        metal_mat = sapien.render.RenderMaterial(base_color=[0.5, 0.5, 0.55, 1.0])
        leg_r = 0.02          # leg half-thickness
        leg_hh = (self.CART_HEIGHT - pt) / 2  # leg half-height

        cart_cx = sfx + self.CART_WIDTH / 2
        cart_cy = sw / 2 + self.CART_Y_GAP + self.CART_DEPTH / 2
        cart_top_cz = self.CART_HEIGHT - pt / 2

        b = self.scene.create_actor_builder()
        # Table top
        b.add_box_collision(half_size=(self.CART_WIDTH / 2, self.CART_DEPTH / 2, pt / 2))
        b.add_box_visual(
            half_size=(self.CART_WIDTH / 2, self.CART_DEPTH / 2, pt / 2),
            material=metal_mat,
        )
        # Four legs (relative to the cart actor origin at the table-top center)
        for dx in [self.CART_WIDTH / 2 - leg_r, -(self.CART_WIDTH / 2 - leg_r)]:
            for dy in [self.CART_DEPTH / 2 - leg_r, -(self.CART_DEPTH / 2 - leg_r)]:
                leg_pose = sapien.Pose(p=[dx, dy, -leg_hh - pt / 2])
                b.add_box_collision(pose=leg_pose, half_size=(leg_r, leg_r, leg_hh))
                b.add_box_visual(pose=leg_pose, half_size=(leg_r, leg_r, leg_hh), material=metal_mat)

        b.initial_pose = sapien.Pose(p=[cart_cx, cart_cy, cart_top_cz])
        self.cart = b.build_kinematic(name="delivery_cart")
        self.cart_surface_z = self.CART_HEIGHT

        self.scene_objects = shelf_parts + [self.cart, self.ground]

    # ------------------------------------------------------------------
    # Convenience properties
    # ------------------------------------------------------------------

    @property
    def target_shelf_surface_z(self) -> float:
        """Z-height of the top surface of the chosen target shelf board."""
        return self.level_surface_zs[self.TARGET_LEVEL_IDX]

    @property
    def shelf_inner_half_width(self) -> float:
        return (self.SHELF_WIDTH - 2 * self.PANEL_THICKNESS) / 2

    @property
    def shelf_interior_half_depth(self) -> float:
        return self.SHELF_DEPTH / 2

    @property
    def shelf_center_x(self) -> float:
        return self.SHELF_FRONT_X + self.SHELF_DEPTH / 2

    # ------------------------------------------------------------------
    # initialize
    # ------------------------------------------------------------------

    def initialize(self, env_idx: torch.Tensor):
        """All shelf geometry is kinematic and never moves; nothing to reset."""
        if hasattr(self.env, "robot_uids") and self.env.robot_uids == "ridgebackur10e":
            self.ground.set_collision_group_bit(
                group=2, bit_idx=RIDGEBACK_WHEELS_COLLISION_BIT, bit=1
            )
