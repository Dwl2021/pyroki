from typing import Sequence

import jax
import jax.numpy as jnp
import jax_dataclasses as jdc
import jaxlie
import jaxls
import numpy as onp
import pyroki as pk
from jax.typing import ArrayLike
from ._solve_ik_with_base_with_multi_targets \
    import _solve_ik_with_base_with_multiple_targets_jax \
        as _solve_ik_with_base_with_multiple_targets_jax


def solve_trajopt_with_base_with_multiple_targets(
    robot: pk.Robot,
    robot_coll: pk.collision.RobotCollision,
    world_coll: Sequence[pk.collision.CollGeom],
    target_link_names: Sequence[str],
    start_joints_state: onp.ndarray,
    start_base_position: onp.ndarray,
    start_base_wxyz: onp.ndarray,
    target_positions: onp.ndarray,
    target_wxyzs: onp.ndarray,
    timesteps: int,
    dt: float,
) -> tuple[onp.ndarray, onp.ndarray, onp.ndarray]:

    # 1. Solve IK for the end poses.
    target_link_indices = [robot.links.names.index(name) for name in target_link_names]
    T_world_targets = [
        jaxlie.SE3(
            jnp.concatenate([jnp.array(target_wxyz), jnp.array(target_position)], axis=-1)
        )
        for target_wxyz, target_position in zip(target_wxyzs, target_positions)
    ]
    start_base_pose = jaxlie.SE3.from_rotation_and_translation(
        jaxlie.SO3(start_base_wxyz),
        start_base_position,
    )
    end_base_pose, end_joints_state = _solve_ik_with_base_with_multiple_targets_jax(
            robot=robot,
            coll=robot_coll,
            world_coll_list=world_coll,
            T_world_targets=T_world_targets,
            target_joint_indices=jnp.array(target_link_indices),
            fix_base=jnp.array([False, False, True] + [True, True, False]),
            base_pos=jnp.array(start_base_position),
            base_wxyz=jnp.array(start_base_wxyz),
            prev_cfg=jnp.array(start_joints_state),
    )

    # 2. Initialize the trajectory through linearly interpolating the start and end poses.
    init_traj = jnp.linspace(jnp.array(start_joints_state), end_joints_state, timesteps)

    # 3. Optimize the trajectory.
    def mm_retract_fn(transform: jaxlie.SE3, delta: jax.Array) -> jaxlie.SE3:
        """Same as jaxls.SE3Var.retract_fn, but removing updates on certain axes."""
        delta = delta * (1 - jnp.array([False, False, True] + [True, True, False]))
        return jaxls.SE3Var.retract_fn(transform, delta)

    class ConstrainedSE3Var(
        jaxls.Var[jaxlie.SE3],
        default_factory=lambda: jaxlie.SE3.from_rotation_and_translation(
            jaxlie.SO3(start_base_wxyz),
            start_base_position,
        ),
        tangent_dim=jaxlie.SE3.tangent_dim,
        retract_fn=mm_retract_fn,
    ): ...
    
    base_pose_var = ConstrainedSE3Var(jnp.arange(0, timesteps))
    base_pose_var_prev = ConstrainedSE3Var(jnp.arange(0, timesteps - 1))
    base_pose_var_next = ConstrainedSE3Var(jnp.arange(1, timesteps))
    
    traj_var = robot.joint_var_cls(jnp.arange(0, timesteps))
    traj_var_prev = robot.joint_var_cls(jnp.arange(0, timesteps - 1))
    traj_var_next = robot.joint_var_cls(jnp.arange(1, timesteps))

    robot = jax.tree.map(lambda x: x[None], robot)  # Add batch dimension.
    robot_coll = jax.tree.map(lambda x: x[None], robot_coll)  # Add batch dimension.

    # Basic regularization / limit costs.
    factors: list[jaxls.Cost] = [
        pk.costs.limit_cost(
            robot,
            traj_var,
            jnp.array([100.0])[None],
        ),
    ]

    # Collision avoidance.
    def compute_world_coll_residual(
        vals: jaxls.VarValues,
        robot: pk.Robot,
        robot_coll: pk.collision.RobotCollision,
        world_coll_obj: pk.collision.CollGeom,
        prev_traj_vars: jaxls.Var[jax.Array],
        curr_traj_vars: jaxls.Var[jax.Array],
        prev_base_pose_var: ConstrainedSE3Var,
        curr_base_pose_var: ConstrainedSE3Var,
    ): 
        coll = robot_coll.get_swept_capsules_with_base(
            robot, vals[prev_traj_vars], vals[curr_traj_vars], vals[prev_base_pose_var], vals[curr_base_pose_var]
        )
        dist = pk.collision.collide(
            coll.reshape((-1, 1)), world_coll_obj.reshape((1, -1))
        )
        colldist = pk.collision.colldist_from_sdf(dist, 0.1)
        return (colldist * 20.0).flatten()

    for world_coll_obj in world_coll:
        factors.append(
            jaxls.Cost(
                compute_world_coll_residual,
                (
                    robot,
                    robot_coll,
                    jax.tree.map(lambda x: x[None], world_coll_obj),
                    traj_var_prev,
                    traj_var_next,
                    base_pose_var_prev,
                    base_pose_var_next,
                ),
                name="World Collision (sweep)",
            )
        )

    # Start / end pose constraints.
    factors.extend(
        [
            jaxls.Cost(
                lambda vals, var: ((vals[var] - jnp.array(start_joints_state)) * 1000.0).flatten(),
                (robot.joint_var_cls(jnp.arange(0, 5)),),
                name="start_pose_constraint",
            ),
            jaxls.Cost(
                lambda vals, var: ((vals[var] - jnp.array(end_joints_state)) * 1000.0).flatten(),
                (robot.joint_var_cls(jnp.arange(timesteps - 5, timesteps)),),
                name="end_pose_constraint",
            )
        ]
    )
    
    @jaxls.Cost.create_factory(name="MatchBoundBaseCost")
    def match_bound_base_cost(
        vals: jaxls.VarValues,
        base_pose_var_start: ConstrainedSE3Var,
        base_pose_var_end: ConstrainedSE3Var,
    ):
        start_poses = vals[base_pose_var_start]
        end_poses = vals[base_pose_var_end]
        start_error = (start_poses.inverse() @ start_base_pose).log()
        end_error = (end_poses.inverse() @ end_base_pose).log()
        return jnp.concatenate([start_error, end_error]).flatten() * 1000.0
    
    factors.append(
        match_bound_base_cost(
            ConstrainedSE3Var(jnp.arange(0, 2)),
            ConstrainedSE3Var(jnp.arange(timesteps - 2, timesteps)),
        )
    )

    # Velocity / acceleration / jerk minimization.
    factors.extend(
        [
            pk.costs.smoothness_cost(
                traj_var_next,
                traj_var_prev,
                weight=50.0,
            ),
            pk.costs.five_point_velocity_cost(
                robot,
                robot.joint_var_cls(jnp.arange(4, timesteps)),
                robot.joint_var_cls(jnp.arange(3, timesteps - 1)),
                robot.joint_var_cls(jnp.arange(1, timesteps - 3)),
                robot.joint_var_cls(jnp.arange(0, timesteps - 4)),
                dt,
                jnp.array([100.0])[None],
            ),
            pk.costs.five_point_acceleration_cost(
                robot.joint_var_cls(jnp.arange(2, timesteps - 2)),
                robot.joint_var_cls(jnp.arange(4, timesteps)),
                robot.joint_var_cls(jnp.arange(3, timesteps - 1)),
                robot.joint_var_cls(jnp.arange(1, timesteps - 3)),
                robot.joint_var_cls(jnp.arange(0, timesteps - 4)),
                dt,
                jnp.array([0.1])[None],
            ),
            pk.costs.five_point_jerk_cost(
                robot.joint_var_cls(jnp.arange(6, timesteps)),
                robot.joint_var_cls(jnp.arange(5, timesteps - 1)),
                robot.joint_var_cls(jnp.arange(4, timesteps - 2)),
                robot.joint_var_cls(jnp.arange(2, timesteps - 4)),
                robot.joint_var_cls(jnp.arange(1, timesteps - 5)),
                robot.joint_var_cls(jnp.arange(0, timesteps - 6)),
                dt,
                jnp.array([0.1])[None],
            ),
        ]
    )
    
    @jaxls.Cost.create_factory(name="BaseVelLimitCost")
    def base_vel_limit_cost(
        vals: jaxls.VarValues,
        base_pose_var: ConstrainedSE3Var,
        base_pose_var_prev: ConstrainedSE3Var,
        weight: float,
        dt: float,
    ):
        base_vel = (vals[base_pose_var].translation() - vals[base_pose_var_prev].translation()) / dt
        residual = jnp.maximum(0.0, jnp.abs(base_vel) - jnp.array([0.3, 0.3, 0.0]))
        return residual.flatten() * weight
    
    factors.append(
        base_vel_limit_cost(
            base_pose_var=base_pose_var_next,
            base_pose_var_prev=base_pose_var_prev,
            weight=1000.0,
            dt=dt,
        )
    )
    @jaxls.Cost.create_factory(name="SE3SmoothnessCost")
    def base_smoothness_cost(
        vals: jaxls.VarValues,
        base_pose_var: ConstrainedSE3Var,
        base_pose_var_prev: ConstrainedSE3Var,
    ):  
        base_smoothness = (vals[base_pose_var].inverse() @ vals[base_pose_var_prev]).log().flatten()
        return base_smoothness * 100.0
    factors.append(
        base_smoothness_cost(
            base_pose_var=base_pose_var_next,
            base_pose_var_prev=base_pose_var_prev,
        )
    )
    
    # 4. Solve the optimization problem.
    solution = (
        jaxls.LeastSquaresProblem(
            factors,
            [traj_var, base_pose_var],
        )
        .analyze()
        .solve(
            initial_vals=jaxls.VarValues.make((traj_var.with_value(init_traj), base_pose_var.with_value(jaxlie.SE3.identity((timesteps,)))))
        )
    )
    return onp.array(solution[traj_var]), onp.array(solution[base_pose_var].translation()), onp.array(solution[base_pose_var].rotation().wxyz)
