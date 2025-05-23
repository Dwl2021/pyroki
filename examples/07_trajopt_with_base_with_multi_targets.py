"""Trajectory Optimization

Basic Trajectory Optimization using PyRoKi.

Robot going over a wall, while avoiding world-collisions.
"""

import time
from typing import Literal

import numpy as np
import pyroki as pk
import trimesh
import tyro
import viser
from viser.extras import ViserUrdf
from robot_descriptions.loaders.yourdfpy import load_robot_description

import pyroki_snippets as pks


def main():
   
    urdf = load_robot_description("yumi_description_mobile")
    target_link_names = ["yumi_link_7_l", "yumi_link_7_r"]
    robot = pk.Robot.from_urdf(urdf)
    down_wxyz = np.array([0, 0, 1, 0])
    robot_coll = pk.collision.RobotCollision.from_urdf(urdf)

    # Define the trajectory problem:
    # - number of timesteps, timestep size
    timesteps, dt = 50, 0.1
    # - the start and end poses.
    end_pos_l = np.array([0.5, -0.15, 0.2])
    end_pos_r = np.array([0.5, -0.45, 0.2])
    start_state = np.array(robot.joint_var_cls.default_factory()[None])[0]
    # Define the obstacles:
    # - Ground
    ground_coll = pk.collision.HalfSpace.from_point_and_normal(
        np.array([0.0, 0.0, 0.0]), np.array([0.0, 0.0, 1.0])
    )
    # - Wall
    wall_height = 0.4
    wall_width = 0.05
    wall_length = 0.4
    wall_intervals = np.arange(start=0.3, stop=wall_length + 0.3, step=0.05)
    translation = np.concatenate(
        [
            wall_intervals.reshape(-1, 1),
            np.full((wall_intervals.shape[0], 1), 0.0),
            np.full((wall_intervals.shape[0], 1), wall_height / 2),
        ],
        axis=1,
    )
    wall_coll = pk.collision.Capsule.from_radius_height(
        position=translation,
        radius=np.full((translation.shape[0], 1), wall_width / 2),
        height=np.full((translation.shape[0], 1), wall_height),
    )
    world_coll = [ground_coll, wall_coll]

    

    # Visualize!
    server = viser.ViserServer()
    base_frame = server.scene.add_frame("/base", show_axes=False)
    urdf_vis = ViserUrdf(server, urdf, root_node_name="/base")
    server.scene.add_grid("/grid", width=2, height=2, cell_size=0.1)
    server.scene.add_mesh_trimesh(
        "wall_box",
        trimesh.creation.box(
            extents=(wall_length, wall_width, wall_height),
            transform=trimesh.transformations.translation_matrix(
                np.array([0.5, 0.0, wall_height / 2])
            ),
        ),
    )
    for name, pos in zip(["end_l", "end_r"], [end_pos_l, end_pos_r]):
        server.scene.add_frame(
            f"/{name}",
            position=pos,
            wxyz=down_wxyz,
            axes_length=0.05,
            axes_radius=0.01,
        )

    slider = server.gui.add_slider(
        "Timestep", min=0, max=timesteps - 1, step=1, initial_value=0
    )
    playing = server.gui.add_checkbox("Playing", initial_value=True)
    
    traj, base_pos, base_wxyz = pks.solve_trajopt_with_base_with_multiple_targets(
        robot=robot,
        robot_coll=robot_coll,
        world_coll=world_coll,
        target_link_names=target_link_names,
        start_joints_state=start_state,
        start_base_position=np.array(base_frame.position),
        start_base_wxyz=np.array(base_frame.wxyz),
        target_positions=np.array([end_pos_l, end_pos_r]),
        target_wxyzs=np.array([down_wxyz, down_wxyz]),
        timesteps=timesteps,
        dt=dt
    )
    traj = np.array(traj)
    base_pos = np.array(base_pos)
    base_wxyz = np.array(base_wxyz)

    while True:
        if playing.value:
            slider.value = (slider.value + 1) % timesteps

        urdf_vis.update_cfg(traj[slider.value])
        base_frame.position = base_pos[slider.value]
        base_frame.wxyz = base_wxyz[slider.value]
        time.sleep(1.0 / 10.0)


if __name__ == "__main__":
    tyro.cli(main)
