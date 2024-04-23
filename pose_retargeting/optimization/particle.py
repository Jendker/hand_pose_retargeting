import numpy as np


class Particle:
    def __init__(self, mujoco_env, parameters, contact_pairs, masked_indices):
        self.parameters = parameters

        self.best_position = None
        self.personal_best = float("inf")

        self.position = None
        self.velocity = None
        self.simulator_initial_state = None
        self.global_best_position = None

        self.actuator_count = mujoco_env.getNumberOfJoints()

        global_orient_std = 5 / 180 * np.pi
        finger_std = 0.2  # given as angle velocity
        global_trans_std = 0.03
        init_action_range = [global_trans_std, global_trans_std, global_trans_std, global_orient_std, global_orient_std,
                             global_orient_std]
        self.init_action_range = np.pad(init_action_range, (0, self.actuator_count - len(init_action_range)), 'constant',
                                                            constant_values=finger_std)
        self.actions_upper_bound = mujoco_env.env.model.actuator_ctrlrange[:, 1]
        self.actions_lower_bound = mujoco_env.env.model.actuator_ctrlrange[:, 0]
        self.position_lower_bound = None
        self.position_upper_bound = None
        self.velocity_bound = None
        self.contact_pairs = contact_pairs
        self.masked_indices = masked_indices
        self.original_actions = None

        self.sim_mujoco_worker = None

    def initializePosition(self, position, original_actions, simulator_state):
        self.original_actions = original_actions
        self.simulator_initial_state = simulator_state
        self.position_lower_bound = np.maximum(self.actions_lower_bound, position - self.init_action_range)
        self.position_upper_bound = np.minimum(self.actions_upper_bound, position + self.init_action_range)
        # TODO: Why does this assert fail?
        # if not np.all(self.position_upper_bound >= self.position_lower_bound):
        #     error = 1
        # assert np.all(self.position_upper_bound >= self.position_lower_bound)
        self.velocity_bound = np.abs(self.position_upper_bound - self.position_lower_bound)
        self.position = np.random.uniform(self.position_lower_bound, self.position_upper_bound)
        self.position[self.masked_indices] = original_actions[self.masked_indices]
        self.best_position = self.position
        self.personal_best = float("inf")

    def initializeVelocity(self):
        self.velocity = np.random.uniform(-self.velocity_bound, self.velocity_bound)

    def updatePositionAndVelocity(self):
        self.velocity = self.velocity * self.parameters['omega']\
            + self.parameters['c1'] * np.random.uniform(size=self.actuator_count) * (self.best_position - self.position)\
            + self.parameters['c2'] * np.random.uniform(size=self.actuator_count) * (self.global_best_position - self.position)
        self.velocity = np.clip(self.velocity, -self.velocity_bound, self.velocity_bound)
        self.position = self.position + self.velocity
        self.position = np.clip(self.position, self.position_lower_bound, self.position_upper_bound)
        self.position[self.masked_indices] = self.original_actions[self.masked_indices]

    def updateGlobalBest(self, global_best_position):
        self.global_best_position = global_best_position

    def updatePersonalBest(self, this_fitness):
        if this_fitness < self.personal_best:
            self.personal_best = this_fitness
            self.best_position = self.position

    def simulationStep(self, sim_mujoco_worker):
        self.sim_mujoco_worker = sim_mujoco_worker
        self.sim_mujoco_worker.env.set_env_state(self.simulator_initial_state)
        self.sim_mujoco_worker.env.step(self.position)

    def getActiveContactsDist(self):
        dist = {}
        for geom1 in self.contact_pairs:
            d1 = []
            for coni in range(self.sim_mujoco_worker.env.data.ncon):
                con = self.sim_mujoco_worker.env.data.contact[coni]
                # get distances for all active contacts with geom1
                # d1 = [data.contact[coni].dim for coni in range(data.ncon) if data.contact[coni].geom1 == geom1
                #      and data.contact[coni].geom2 == geom2 for geom2 in np.where(np.asarray(contact_pairs(geom1)))[0]]
                if (geom1 == con.geom1 and con.geom2 in self.contact_pairs[geom1]) \
                        or (geom1 == con.geom2 and con.geom1 in self.contact_pairs[geom1]):
                    # contact is in the pair list
                    d1.append(con.dist)

            if len(d1):
                dist[geom1] = d1
        return dist
