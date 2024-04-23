import numpy as np
import os
from math import ceil, isclose
from pose_retargeting.optimization.particle import Particle
from pose_retargeting.simulator.sim_mujoco import Mujoco
from mjrl.utils.gym_env import GymEnv
from multiprocessing import Pool, cpu_count
from pose_retargeting.rotations_mujoco import quat2euler
from pose_retargeting.optimization.miscellaneous import Targets, ConstantData, Weights


glob_constant_data = None
glob_sim_mujoco_worker = None
glob_particle_batch = None


class PSO:
    def __init__(self, mujoco_env, parameters=None, num_cpu=None, masked_indices='default'):
        self.num_cpu = num_cpu
        if self.num_cpu is None:
            self.num_cpu = cpu_count()

        if parameters is None:
            self.parameters = {'c1': 2.8, 'c2': 1.3}
        else:
            self.parameters = parameters
        psi = self.parameters['c1'] + self.parameters['c2']
        self.parameters['omega'] = 2/abs(2-psi-np.sqrt(psi*psi-4 * psi))

        self.n_particles = 16
        self.particles_in_batch = ceil(self.n_particles / self.num_cpu)
        self.iteration_count = 5
        self.dimension = mujoco_env.getNumberOfJoints()
        self.convergance_difference = 5e-05

        self.optimization_distance_threshold = 0.04

        self.particles = None
        self.best_global_fitness = float("inf")
        self.best_global_particle_position = None
        self.last_best_particle_position = None

        # joint_names_for_hand_pose_energy_angles = self.mujoco_env.model.joint_names[
        #                                           self.mujoco_env.model.joint_name2id('rh_FFJ4'):
        #                                           self.mujoco_env.model.joint_name2id('rh_THJ1')+1]

        if masked_indices == 'default':
            self.masked_indices = slice(0, 6)
        if masked_indices is None:
            self.masked_indices = slice(0, 0)

        self.constant_data = ConstantData(mujoco_env)

        # parallel workers with environments
        if self.num_cpu > 1:
            self.worker_pool = Pool(self.num_cpu, initializer=PSO.initialize_pool, initargs=[self.constant_data,
                                    self.parameters, self.getContactPairs(mujoco_env), mujoco_env, self.particles_in_batch,
                                    self.masked_indices])

        self.weights = Weights(self.constant_data.bodies_for_hand_pose_energy_position, mujoco_env)
        self.targets = Targets()

        self.obj_body_index = mujoco_env.env.model.body_name2id('Object')
        self.grasp_site_index = mujoco_env.env.model.site_name2id('S_grasp')

        self.mujoco_env = mujoco_env

    def __del__(self):
        try:
            self.worker_pool.close()
            self.worker_pool.terminate()
            self.worker_pool.join()
        except AttributeError:
            pass

    @staticmethod
    def getContactPairs(mujoco_env):
        geom1 = mujoco_env.env.model.pair_geom1
        geom2 = mujoco_env.env.model.pair_geom2
        # TODO: group the geoms into bodies
        pairs = {}
        if geom1 is not None and geom2 is not None:
            assert (len(geom1) == len(geom2))
            # group geom2 by geom1
            for elem in set(geom1):
                tmp = [geom2[i] for i in np.where(np.asarray(geom1) == elem)[0]]
                pairs[elem] = tmp
        return pairs

    def getDistanceBetweenObjectAndHand(self, mujoco_env):
        obj_pos = mujoco_env.env.data.body_xpos[self.obj_body_index].ravel()
        palm_pos = mujoco_env.env.data.site_xpos[self.grasp_site_index].ravel()
        return np.linalg.norm(obj_pos - palm_pos)

    def isObjectAboveTable(self, mujoco_env):
        obj_pos = mujoco_env.env.data.body_xpos[self.obj_body_index].ravel()
        return obj_pos[2] > 0.1

    @staticmethod
    def initialize_pool(constant_data, parameters, contact_pairs, mujoco_env, particles_in_batch, masked_indices):
        global glob_sim_mujoco_worker, glob_constant_data, glob_particle_batch
        np.random.seed(int.from_bytes(os.urandom(4), byteorder='little'))
        env_name = mujoco_env.env_name
        glob_constant_data = constant_data
        env = GymEnv(env_name)
        env.reset()
        glob_sim_mujoco_worker = Mujoco(env, env_name)
        glob_particle_batch = [Particle(mujoco_env, parameters, contact_pairs, masked_indices) for _ in range(particles_in_batch)]

    def initializeParticles(self, actions, simulator_state):
        if self.last_best_particle_position is not None:
            initial_particle_position = (actions + self.last_best_particle_position) / 2
        else:
            initial_particle_position = actions

        # simulator_state and initial_particle_position are not the same, the actions from inverse kin. are the latter
        inputs = [[initial_particle_position, actions, simulator_state, self.weights, self.targets] for _ in range(self.num_cpu)]
        fitness_positions = np.array(self._run_multiprocess(inputs, PSO.batchParticlesInitialization))
        fitness = fitness_positions[:, 0]
        lowest_batch_index = np.unravel_index(np.argmin(fitness, axis=None), fitness.shape)
        lowest_fitness = fitness[lowest_batch_index]
        self.best_global_particle_position = fitness_positions[lowest_batch_index][1]
        self.best_global_fitness = lowest_fitness

    def optimize(self, actions, mujoco_env):
        distance_between_object_and_hand = self.getDistanceBetweenObjectAndHand(mujoco_env)
        if distance_between_object_and_hand > self.optimization_distance_threshold:
            return actions
        if self.isObjectAboveTable(mujoco_env):
            this_iteration_count = int(np.floor(self.iteration_count/2))
        else:
            this_iteration_count = self.iteration_count

        self.weights.update_weights(distance_between_object_and_hand)
        simulator_state = mujoco_env.env.get_env_state()

        self.initializeParticles(actions, simulator_state)

        last_converged = False
        for i in range(0, this_iteration_count):
            inputs = [[self.weights, self.targets, self.best_global_particle_position] for _ in range(self.num_cpu)]
            fitness_positions = np.array(self._run_multiprocess(inputs, PSO.batchParticlesGeneration))
            fitness = fitness_positions[:, 0]
            lowest_batch_index = np.unravel_index(np.argmin(fitness, axis=None), fitness.shape)
            lowest_fitness = fitness[lowest_batch_index]
            converged = isclose(lowest_fitness, self.best_global_fitness, rel_tol=self.convergance_difference) or \
                self.best_global_fitness < lowest_fitness
            if lowest_fitness < self.best_global_fitness:
                self.best_global_particle_position = fitness_positions[lowest_batch_index][1]
                self.best_global_fitness = lowest_fitness
            if converged and last_converged:
                break
            last_converged = converged

        self.last_best_particle_position = self.best_global_particle_position
        return self.best_global_particle_position

    def _run_multiprocess(self, args_list, function):
        if self.num_cpu > 1:
            results = self.worker_pool.map(function, args_list)
        else:
            results = []
            if glob_sim_mujoco_worker is None:
                PSO.initialize_pool(self.constant_data, self.parameters, self.getContactPairs(self.mujoco_env),
                                    self.mujoco_env, self.particles_in_batch, self.masked_indices)
            for i in range(self.num_cpu):
                results.append(function(args_list[i]))
        return results

    def _setHandTargetPositionAndQuaternion(self, target_position, target_quaternion):
        self.targets.hand_target_position = target_position
        self.targets.hand_target_orientation = quat2euler(target_quaternion)
    
    def new_taget_pose(self, new_target_fingers_pose, target_position, target_quaternion):
        self.targets.target_joints_pose = new_target_fingers_pose
        self._setHandTargetPositionAndQuaternion(target_position, target_quaternion)

    @staticmethod
    def getHandPoseEnergyPosition(particle, weights, targets):
        global glob_constant_data
        energy = 0
        evaluated_hand_position = particle.sim_mujoco_worker.simulationObjectsPoseList(
            glob_constant_data.bodies_for_hand_pose_energy_position)
        for index, (target_pose, eval_hand_pose) in enumerate(
                zip(targets.target_joints_pose, evaluated_hand_position)):
            energy += weights.pose_weights[index] * np.linalg.norm(target_pose - eval_hand_pose)**2
        return energy / weights.sum_of_hand_pose_weights

    @staticmethod
    def getHandPoseEnergyAngles(particle, targets):
        global glob_constant_data
        fingers = [[0, 2, 9, 10, 11], [0, 3, 12, 13, 14], [0, 4, 15, 16, 17], [0, 5, 18, 19, 20], [0, 1, 6, 7, 8]]
        target_angles = []
        evaluated_joint_positions = []
        for finger_index, finger in enumerate(fingers):
            # TODO: handle thumb differently (proximal is different than [0, 0, 1]), later we have
            # again two joints in one point
            for index, this_point_index in enumerate(finger):
                if index - 1 < 0 or index + 1 >= len(finger):
                    continue
                previous_index = finger[index - 1]
                next_index = finger[index + 1]
                previous_vector_target = targets.target_joints_pose[this_point_index] - targets.target_joints_pose[
                    previous_index]
                next_vector_target = targets.target_joints_pose[next_index] -\
                                     targets.target_joints_pose[this_point_index]
                target_angles.append(np.arccos(np.dot(next_vector_target, previous_vector_target)))
                evaluated_joint_positions.append(particle.sim_mujoco_worker.getJointPosition(glob_constant_data.bodies_for_hand_pose_energy_angles[
                    this_point_index])[1])
        target_angles = np.array(target_angles)
        evaluated_joint_positions = np.array(evaluated_joint_positions)
        mse = (((evaluated_joint_positions - target_angles) ** 2).mean(axis=None) / np.pi**2)
        return mse

    @staticmethod
    def getTaskEnergy(particle, weights):
        global glob_constant_data
        missing_weight = 2
        margin = 0.04
        constant = 0.004
        contact_dist = particle.getActiveContactsDist()
        # add palm
        assert(isinstance(weights.palm_weight, int))  # palm weight to make it more important than the rest of the tips (must be integer)
        assert(isinstance(weights.thumb_weight, int))  # thumb weight the same story
        real_contact_distances = []
        if contact_dist:
            # find smallest distance for palm (we have many, many geoms currently for palm)
            palm_distances = []
            for key in contact_dist:
                # check if key (index) belongs to palm
                if glob_constant_data.palm_max_index >= key >= glob_constant_data.palm_min_index:
                    # if so, append
                    palm_distances.append(contact_dist[key])
                elif key in glob_constant_data.thumb_geom_indices:
                    for i in range(weights.thumb_weight - 1):
                        real_contact_distances.append(contact_dist[key])
                else:
                    # it is finger, just add to to real_contact_dist
                    real_contact_distances.append(contact_dist[key])
            if palm_distances:
                # only add the smallest palm distance to real_contact_dist for energy calculation
                smallest_distance = min(palm_distances)
                for i in range(weights.palm_weight - 1):
                    real_contact_distances.append(smallest_distance)  # add identical palm entries for the mean
        total = weights.sum_of_task_energy_weights
        if real_contact_distances:
            s = (total - len(real_contact_distances)) * missing_weight * (
                        (margin + constant) ** 2)  # punish for those that are not even in range
            for distance in real_contact_distances:
                # ideally the distance is less than zero, so add constant to make it positive
                s += (max(max(distance) + constant,
                          0)) ** 2  # we want it to be less than 0 so there applied force
            # normalise
            # coeff = (len(contact_dist) + (5 - len(contact_dist)) * missing_weight) * ((margin + constant) ** 2)
            s /= (len(real_contact_distances) + (total - len(real_contact_distances)) * missing_weight) * ((margin + constant) ** 2)
            return s
        else:
            return 1

    @staticmethod
    def batchParticlesInitialization(inputs):
        initial_particle_position = inputs[0]
        original_actions = inputs[1]
        simulator_state = inputs[2]
        weights = inputs[3]
        targets = inputs[4]
        global glob_sim_mujoco_worker, glob_particle_batch

        energies = []
        for particle in glob_particle_batch:
            particle.initializePosition(initial_particle_position, original_actions, simulator_state)  # also sets self best to current
            particle.simulationStep(glob_sim_mujoco_worker)
            this_energy = PSO.fitness(particle, weights, targets)
            energies.append(this_energy)
            particle.initializeVelocity()
        lowest_energy = min(energies)
        lowest_energy_position = glob_particle_batch[energies.index(lowest_energy)].position
        return [lowest_energy, lowest_energy_position]

    @staticmethod
    def batchParticlesGeneration(inputs):
        weights = inputs[0]
        targets = inputs[1]
        global_best_particle_position = inputs[2]
        global glob_sim_mujoco_worker, glob_particle_batch

        energies = []
        for particle in glob_particle_batch:
            particle.updateGlobalBest(global_best_particle_position)
            particle.updatePositionAndVelocity()
            particle.simulationStep(glob_sim_mujoco_worker)
            this_energy = PSO.fitness(particle, weights, targets)
            energies.append(this_energy)
            particle.updatePersonalBest(this_energy)
        lowest_energy = min(energies)
        lowest_energy_position = glob_particle_batch[energies.index(lowest_energy)].position
        return [lowest_energy, lowest_energy_position]

    @staticmethod
    def fitness(particle, weights, targets):
        global glob_constant_data
        return weights.weight_hand_pose_energy_position * PSO.getHandPoseEnergyPosition(particle, weights, targets) + \
               weights.weight_hand_pose_energy_angle * PSO.getHandPoseEnergyAngles(particle, targets) + \
               + weights.weight_task_energy * PSO.getTaskEnergy(particle, weights)

