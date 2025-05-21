"""Mobile IK

Same as 01_basic_ik.py, but with a mobile base!
"""

import time
import viser
from robot_descriptions.loaders.yourdfpy import load_robot_description
from pyroki.collision import HalfSpace, RobotCollision, Sphere
import numpy as np

import pyroki as pk
from viser.extras import ViserUrdf
import pyroki_snippets as pks


def main():
    """Main function for IK with a mobile base.
    The base is fixed along the xy plane, and is biased towards being at the origin.
    """

    urdf = load_robot_description("yumi_description_mobile")
    target_link_name = ["yumi_link_7_r", "yumi_link_7_l"]

    # Create robot.
    robot = pk.Robot.from_urdf(urdf)
    
    robot_coll = RobotCollision.from_urdf(urdf)
    plane_coll = HalfSpace.from_point_and_normal(
        np.array([0.0, 0.0, 0.0]), np.array([0.0, 0.0, 1.0])
    )
    sphere_coll = Sphere.from_center_and_radius(
        np.array([0.0, 0.0, 0.0]), np.array([0.05])
    )

    # Set up visualizer.
    server = viser.ViserServer()
    server.scene.add_grid("/ground", width=2, height=2)
    base_frame = server.scene.add_frame("/base", show_axes=False)
    urdf_vis = ViserUrdf(server, urdf, root_node_name="/base")

    # Create interactive controller with initial position.
    ik_target_0 = server.scene.add_transform_controls(
        "/ik_target_0", scale=0.2, position=(0.41, -0.3, 0.56), wxyz=(0, 0, 1, 0)
    )
    ik_target_1 = server.scene.add_transform_controls(
        "/ik_target_1", scale=0.2, position=(0.41, 0.3, 0.56), wxyz=(0, 0, 1, 0)
    )
    # Create interactive controller and mesh for the sphere obstacle.
    sphere_handle = server.scene.add_transform_controls(
        "/obstacle", scale=0.2, position=(0.4, 0.0, 0.4)
    )
    server.scene.add_mesh_trimesh("/obstacle/mesh", mesh=sphere_coll.to_trimesh())
    
    timing_handle = server.gui.add_number("Elapsed (ms)", 0.001, disabled=True)

    cfg = np.array(robot.joint_var_cls(0).default_factory())

    while True:
        # Solve IK.
        start_time = time.time()
        
        sphere_coll_world_current = sphere_coll.transform_from_wxyz_position(
            wxyz=np.array(sphere_handle.wxyz),
            position=np.array(sphere_handle.position),
        )
        
        world_coll_list = [plane_coll, sphere_coll_world_current]
        base_pos, base_wxyz, cfg = pks.solve_ik_with_base_with_multiple_targets(
            robot=robot,
            coll=robot_coll,
            world_coll_list=world_coll_list,
            target_link_name=target_link_name,
            target_position=np.array([ik_target_0.position, ik_target_1.position]),
            target_wxyz=np.array([ik_target_0.wxyz, ik_target_1.wxyz]),
            fix_base_position=(False, False, True),  # Only free along xy plane.
            fix_base_orientation=(True, True, False),  # Free along z-axis rotation.
            prev_pos=base_frame.position,
            prev_wxyz=base_frame.wxyz,
            prev_cfg=cfg,
        )

        # Update timing handle.
        elapsed_time = time.time() - start_time
        timing_handle.value = 0.99 * timing_handle.value + 0.01 * (elapsed_time * 1000)

        # Update visualizer.
        urdf_vis.update_cfg(cfg)
        base_frame.position = np.array(base_pos)
        base_frame.wxyz = np.array(base_wxyz)


if __name__ == "__main__":
    main()
