"""
Solves the basic IK problem with multiple targets and collision avoidance.
"""

from typing import Sequence

import jax
import jax.numpy as jnp
import jax_dataclasses as jdc
import jaxlie
import jaxls
import numpy as onp
import pyroki as pk


def solve_ik_with_multiple_targets_with_collision(
    robot: pk.Robot,
    coll: pk.collision.RobotCollision,
    world_coll_list: Sequence[pk.collision.CollGeom],
    target_link_names: Sequence[str],
    target_positions: onp.ndarray,
    target_wxyzs: onp.ndarray,
) -> onp.ndarray:
    """
    Solves the basic IK problem for a robot with multiple targets.

    Args:
        robot: PyRoKi Robot.
        coll: Robot collision model.
        world_coll_list: List of collision geometries in the world.
        target_link_names: Sequence[str]. Length: num_targets.
        target_positions: ArrayLike. Shape: (num_targets, 3).
        target_wxyzs: ArrayLike. Shape: (num_targets, 4).

    Returns:
        cfg: ArrayLike. Shape: (robot.joint.actuated_count,).
    """
    num_targets = len(target_link_names)
    assert target_positions.shape == (num_targets, 3)
    assert target_wxyzs.shape == (num_targets, 4)
    target_link_indices = [robot.links.names.index(name) for name in target_link_names]

    cfg = _solve_ik_with_multiple_targets_with_collision_jax(
        robot,
        coll,
        world_coll_list,
        jnp.array(target_wxyzs),
        jnp.array(target_positions),
        jnp.array(target_link_indices),
    )
    assert cfg.shape == (robot.joints.num_actuated_joints,)

    return onp.array(cfg)


@jdc.jit
def _solve_ik_with_multiple_targets_with_collision_jax(
    robot: pk.Robot,
    coll: pk.collision.RobotCollision,
    world_coll_list: Sequence[pk.collision.CollGeom],
    target_wxyzs: jax.Array,
    target_positions: jax.Array,
    target_link_indices: jax.Array,
) -> jax.Array:
    """Solves the IK problem with multiple targets and collision avoidance. Returns joint configuration."""
    joint_var = robot.joint_var_cls(0)
    vars = [joint_var]

    # Create target poses for all targets
    target_poses = [
        jaxlie.SE3.from_rotation_and_translation(
            jaxlie.SO3(wxyz), position
        )
        for wxyz, position in zip(target_wxyzs, target_positions)
    ]

    # Create pose costs for all targets
    pose_costs = [
        pk.costs.pose_cost(
            robot,
            joint_var,
            target_pose=target_pose,
            target_link_index=link_idx,
            pos_weight=5.0,
            ori_weight=1.0,
        )
        for target_pose, link_idx in zip(target_poses, target_link_indices)
    ]

    costs = pose_costs + [
        pk.costs.limit_cost(
            robot,
            joint_var=joint_var,
            weight=100.0,
        ),
        pk.costs.rest_cost(
            joint_var,
            rest_pose=jnp.array(joint_var.default_factory()),
            weight=0.01,
        ),
        pk.costs.self_collision_cost(
            robot,
            robot_coll=coll,
            joint_var=joint_var,
            margin=0.02,
            weight=5.0,
        ),
    ]
    costs.extend(
        [
            pk.costs.world_collision_cost(
                robot, coll, joint_var, world_coll, 0.05, 10.0
            )
            for world_coll in world_coll_list
        ]
    )

    sol = (
        jaxls.LeastSquaresProblem(costs, vars)
        .analyze()
        .solve(verbose=False, linear_solver="dense_cholesky")
    )
    return sol[joint_var]
