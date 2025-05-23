"""Online Planning

Run online planning in collision aware environments.
"""

import time

import numpy as np
import pyroki as pk
import viser
from pyroki.collision import HalfSpace, RobotCollision, Sphere
from robot_descriptions.loaders.yourdfpy import load_robot_description
from viser.extras import ViserUrdf
import yourdfpy
import jaxlie


import pyroki_snippets as pks


def main():
    """Main function for online planning with collision."""
    # urdf = load_robot_description("panda_description")
    # target_link_name = "panda_hand"
    # urdf = load_robot_description("yumi_description_mobile")
    urdf = yourdfpy.URDF.load("../rm65/robot_description.urdf")
    target_link_names = ["left_Link6", "right_Link6"]
    robot = pk.Robot.from_urdf(urdf)

    robot_coll = RobotCollision.from_urdf(urdf)
    plane_coll = HalfSpace.from_point_and_normal(
        np.array([0.0, 0.0, 0.0]), np.array([0.0, 0.0, 1.0])
    )
    sphere_coll = Sphere.from_center_and_radius(
        np.array([0.0, 0.0, 0.0]), np.array([0.05])
    )

    # Define the online planning parameters.
    len_traj, dt = 5, 0.1

    # Set up visualizer.
    server = viser.ViserServer()
    server.scene.add_grid("/ground", width=2, height=2, cell_size=0.1)
    base_frame = server.scene.add_frame("/base", show_axes=False)
    urdf_vis = ViserUrdf(server, urdf, root_node_name="/base")

    # Create interactive controller for IK target.
    ik_target_handle_0 = server.scene.add_transform_controls(
        "/ik_target_0", scale=0.2, position=(2, -0.15, 0.5), wxyz=(0, 0, 1, 0)
    )
    ik_target_handle_1 = server.scene.add_transform_controls(
        "/ik_target_1", scale=0.2, position=(2, -0.45, 0.5), wxyz=(0, 0, 1, 0)
    )

    # Create interactive controller and mesh for the sphere obstacle.
    sphere_handle = server.scene.add_transform_controls(
        "/obstacle", scale=0.2, position=(0.4, 0.3, 0.4)
    )
    server.scene.add_mesh_trimesh("/obstacle/mesh", mesh=sphere_coll.to_trimesh())
    target_frame_handle = server.scene.add_batched_axes(
        "target_frame",
        axes_length=0.05,
        axes_radius=0.005,
        batched_positions=np.zeros((25, 3)),
        batched_wxyzs=np.array([[1.0, 0.0, 0.0, 0.0]] * 25),
    )
    
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

    timing_handle = server.gui.add_number("Elapsed (ms)", 0.001, disabled=True)

    sol_pos, sol_wxyz = None, None
    sol_traj = np.array(
        robot.joint_var_cls.default_factory()[None].repeat(len_traj, axis=0)
    )
    flag = False
    while True:
        start_time = time.time()

        sphere_coll_world_current = sphere_coll.transform_from_wxyz_position(
            wxyz=np.array(sphere_handle.wxyz),
            position=np.array(sphere_handle.position),
        )
        world_coll_list = [plane_coll, sphere_coll_world_current, wall_coll]
        sol_traj, sol_pos, sol_wxyz, base_pos, base_wxyz = pks.solve_online_planning_with_base_with_multiple_targets(
            robot=robot,
            robot_coll=robot_coll,
            world_coll=world_coll_list,
            target_link_names=target_link_names,
            target_positions=np.array([ik_target_handle_0.position, ik_target_handle_1.position]),
            target_wxyzs=np.array([ik_target_handle_0.wxyz, ik_target_handle_1.wxyz]),
            timesteps=len_traj,
            dt=dt,
            start_cfg=sol_traj[0],
            prev_sols=sol_traj,
            fix_base_position=(False, False, True),
            fix_base_orientation=(True, True, False),
            prev_pos=base_frame.position,
            prev_wxyz=base_frame.wxyz
        )
        if not flag:
            curr_time = time.time()
            print(f"Time taken to warm up: {curr_time - start_time}")
            input("Press Enter to continue...")
            flag = True

        # Update timing handle.
        timing_handle.value = (
            0.99 * timing_handle.value + 0.01 * (time.time() - start_time) * 1000
        )

        for i in range(3):  # Execute 3 steps at once
            urdf_vis.update_cfg(sol_traj[i])
            base_frame.position = np.array(base_pos[i])
            base_frame.wxyz = np.array(base_wxyz[i])
            print(f"Step {i} base position:", base_frame.position)

        # Update the planned trajectory visualization.
        if hasattr(target_frame_handle, "batched_positions"):
            # Transform positions and orientations through base frame
            transformed_positions = []
            transformed_wxyzs = []
            for pos, wxyz in zip(sol_pos, sol_wxyz):
                # Create SE3 transform for current pose
                current_pose = jaxlie.SE3.from_rotation_and_translation(
                    jaxlie.SO3(wxyz), pos
                )
                # Create SE3 transform for base frame
                base_pose = jaxlie.SE3.from_rotation_and_translation(
                    jaxlie.SO3(base_frame.wxyz), base_frame.position
                )
                # Transform through base frame
                transformed_pose = base_pose @ current_pose
                transformed_positions.append(transformed_pose.translation())
                transformed_wxyzs.append(transformed_pose.rotation().wxyz)
            
            target_frame_handle.batched_positions = np.array(transformed_positions)  # type: ignore[attr-defined]
            target_frame_handle.batched_wxyzs = np.array(transformed_wxyzs)  # type: ignore[attr-defined]
        else:
            # This is an older version of Viser.
            # Transform positions and orientations through base frame
            transformed_positions = []
            transformed_wxyzs = []
            for pos, wxyz in zip(sol_pos, sol_wxyz):
                # Create SE3 transform for current pose
                current_pose = jaxlie.SE3.from_rotation_and_translation(
                    jaxlie.SO3(wxyz), pos
                )
                # Create SE3 transform for base frame
                base_pose = jaxlie.SE3.from_rotation_and_translation(
                    jaxlie.SO3(base_frame.wxyz), base_frame.position
                )
                # Transform through base frame
                transformed_pose = base_pose @ current_pose
                transformed_positions.append(transformed_pose.translation())
                transformed_wxyzs.append(transformed_pose.rotation().wxyz)
            
            target_frame_handle.positions_batched = np.array(transformed_positions)  # type: ignore[attr-defined]
            target_frame_handle.wxyzs_batched = np.array(transformed_wxyzs)  # type: ignore[attr-defined]


if __name__ == "__main__":
    main()
