import math
import time

import numpy as np
import pybullet as p
import pybullet_data


class DirectPyBulletAntEnv:
    def __init__(self, render=False, max_episode_steps=1000, seed=0, action_scale=30.0):
        self.render = render
        self.max_episode_steps = max_episode_steps
        self.action_scale = action_scale
        self.rng = np.random.default_rng(seed)

        self.client = p.connect(p.GUI if render else p.DIRECT)
        p.setAdditionalSearchPath(pybullet_data.getDataPath(), physicsClientId=self.client)
        p.setGravity(0, 0, -9.8, physicsClientId=self.client)
        p.setTimeStep(1.0 / 240.0, physicsClientId=self.client)

        self.plane_id = None
        self.ant_id = None
        self.joint_ids = []
        self.foot_links = []
        self.step_count = 0

    def close(self):
        if self.client is not None:
            p.disconnect(self.client)
            self.client = None

    def sample_friction(self, low=0.02, high=2.0):
        return float(10 ** self.rng.uniform(math.log10(low), math.log10(high)))

    def reset(self, friction=1.0):
        p.resetSimulation(physicsClientId=self.client)
        p.setGravity(0, 0, -9.8, physicsClientId=self.client)
        p.setTimeStep(1.0 / 240.0, physicsClientId=self.client)

        self.plane_id = p.loadURDF("plane.urdf", physicsClientId=self.client)

        ant_ids = p.loadMJCF("mjcf/ant.xml", physicsClientId=self.client)
        self.ant_id = ant_ids[0] if isinstance(ant_ids, (list, tuple)) else ant_ids

        self.joint_ids = []
        self.foot_links = []

        num_joints = p.getNumJoints(self.ant_id, physicsClientId=self.client)
        for j in range(num_joints):
            info = p.getJointInfo(self.ant_id, j, physicsClientId=self.client)
            joint_type = info[2]
            link_name = info[12].decode("utf-8").lower()

            if joint_type in (p.JOINT_REVOLUTE, p.JOINT_PRISMATIC):
                self.joint_ids.append(j)

            if "foot" in link_name or "ankle" in link_name:
                self.foot_links.append(j)

        if not self.foot_links:
            self.foot_links = list(range(num_joints))

        for joint_id in self.joint_ids:
            p.setJointMotorControl2(
                self.ant_id,
                joint_id,
                controlMode=p.VELOCITY_CONTROL,
                force=0,
                physicsClientId=self.client,
            )

        self.set_friction(friction)
        self.step_count = 0
        return self._get_obs()

    def set_friction(self, friction):
        p.changeDynamics(self.plane_id, -1, lateralFriction=friction, physicsClientId=self.client)
        for link in self.foot_links:
            p.changeDynamics(self.ant_id, link, lateralFriction=friction, physicsClientId=self.client)

    def _get_obs(self):
        base_pos, base_quat = p.getBasePositionAndOrientation(self.ant_id, physicsClientId=self.client)
        base_lin_vel, base_ang_vel = p.getBaseVelocity(self.ant_id, physicsClientId=self.client)
        joint_states = p.getJointStates(self.ant_id, self.joint_ids, physicsClientId=self.client)
        q = np.array([s[0] for s in joint_states], dtype=np.float32)
        qd = np.array([s[1] for s in joint_states], dtype=np.float32)

        return np.concatenate([
            np.array(base_pos, dtype=np.float32),
            np.array(base_quat, dtype=np.float32),
            np.array(base_lin_vel, dtype=np.float32),
            np.array(base_ang_vel, dtype=np.float32),
            q,
            qd,
        ])

    def _contact_cost(self):
        total = 0.0
        for link in self.foot_links:
            pts = p.getContactPoints(bodyA=self.ant_id, linkIndexA=link, physicsClientId=self.client)
            for pt in pts:
                normal_force = pt[9]
                total += min(normal_force, 1.0) ** 2
        return 5e-4 * total

    def step(self, action):
        action = np.asarray(action, dtype=np.float32)
        action = np.clip(action, -1.0, 1.0)
        assert action.shape[0] == len(self.joint_ids)

        torques = self.action_scale * action
        x_before = p.getBasePositionAndOrientation(self.ant_id, physicsClientId=self.client)[0][0]

        for joint_id, torque in zip(self.joint_ids, torques):
            p.setJointMotorControl2(
                self.ant_id,
                joint_id,
                controlMode=p.TORQUE_CONTROL,
                force=float(torque),
                physicsClientId=self.client,
            )

        p.stepSimulation(physicsClientId=self.client)

        base_pos = p.getBasePositionAndOrientation(self.ant_id, physicsClientId=self.client)[0]
        x_after, z_after = base_pos[0], base_pos[2]

        self.step_count += 1

        dt = 1.0 / 240.0
        forward_reward = (x_after - x_before) / dt
        alive_bonus = 1.0
        ctrl_cost = 0.5 * float(np.square(action).sum())
        contact_cost = self._contact_cost()

        reward = forward_reward + alive_bonus - ctrl_cost - contact_cost

        done = not (0.08 <= z_after <= 1.0)
        if self.step_count >= self.max_episode_steps:
            done = True

        obs = self._get_obs()
        info = {
            "forward_reward": forward_reward,
            "alive_bonus": alive_bonus,
            "ctrl_cost": ctrl_cost,
            "contact_cost": contact_cost,
        }
        return obs, reward, done, info

def main():
    render = True
    env = DirectPyBulletAntEnv(render=render, max_episode_steps=1000, seed=0)

    global_steps = 0
    friction = env.sample_friction()
    next_change_step = 2_000_000
    obs = env.reset(friction=friction)

    while global_steps < 20_000_000:
        done = False
        ep_return = 0.0

        while not done:
            if render:
                time.sleep(1.0 / 240.0)
            action = np.zeros(len(env.joint_ids), dtype=np.float32)  # replace with PPO output
            obs, reward, done, info = env.step(action)
            ep_return += reward
            global_steps += 1

        if global_steps >= next_change_step:
            friction = env.sample_friction()
            next_change_step += 2_000_000

        obs = env.reset(friction=friction)
        print("steps:", global_steps, "return:", ep_return, "friction:", friction)

    env.close()


if __name__ == "__main__":
    main()