import random
from math import cos, sin

import numpy as np

from system_manager_package.constants import (
    GA_COLLISION_VOLUMES,
    GA_CONVERGENCE_CHECK,
    GA_CONVERGENCE_TOLERANCE,
    GA_DISTANCE_SCALE_MM,
    GA_DISTANCE_WEIGHT,
    GA_ELITISM_PERCENT,
    GA_INITIAL_GENE_LENGTH_RANGE,
    GA_INITIAL_POS,
    GA_JOINT1_LIMITS,
    GA_JOINT2_LIMITS,
    GA_JOINT3_LIMITS,
    GA_LENGTH_PENALTY_WEIGHT,
    GA_LINK_1_MM,
    GA_LINK_2_MM,
    GA_LINK_3_MM,
    GA_MUTATION_RATE,
    GA_NON_ORTHO_APPROACH_WEIGHT,
    GA_NUM_GENERATIONS,
    GA_POPULATION_SIZE,
    GA_POSE_WEIGHT,
    GA_RANDOM_BACKFILL_PERCENT,
    GA_SELECTION_METHOD,
    GA_STEP_SIZE,
    GA_TARGET_DISTANCE_THRESHOLD_MM,
    GA_TOURNAMENT_K,
    GA_Z_GLOBAL_OFFSET_MM,
)

ELITISM_PERCENT = GA_ELITISM_PERCENT
MUTATION_RATE = GA_MUTATION_RATE
SELECTION_METHOD = GA_SELECTION_METHOD
TOURNAMENT_K = GA_TOURNAMENT_K
TARGET_DISTANCE_THRESHOLD_MM = GA_TARGET_DISTANCE_THRESHOLD_MM
POPULATION_SIZE = GA_POPULATION_SIZE
RANDOM_BACKFILL_PERCENT = GA_RANDOM_BACKFILL_PERCENT
STEP_SIZE = GA_STEP_SIZE
NUM_GENERATIONS = GA_NUM_GENERATIONS
INITIAL_GENE_LENGTH_RANGE = GA_INITIAL_GENE_LENGTH_RANGE

DISTANCE_WEIGHT = GA_DISTANCE_WEIGHT # 5.25
POSE_WEIGHT = GA_POSE_WEIGHT
NON_ORTHO_APPROACH_WEIGHT = GA_NON_ORTHO_APPROACH_WEIGHT
LENGTH_PENALTY_WEIGHT = GA_LENGTH_PENALTY_WEIGHT # 0.3
DISTANCE_SCALE_MM = GA_DISTANCE_SCALE_MM # This is 500mm
Z_GLOBAL_OFFSET_MM = GA_Z_GLOBAL_OFFSET_MM

CONVERGENCE_CHECK = GA_CONVERGENCE_CHECK
CONVERGENCE_TOLERANCE = GA_CONVERGENCE_TOLERANCE

FITNESS_PLOT_ENABLED = False
FITNESS_PLOT_FILENAME = "ga_fitness_history.png"
FITNESS_PLOT_DPI = 300 # Resolution for saved plot

# ----------------------------------------------------------------#
# IF YOU MODIFIED THE GA CODE, PASTE YOUR CODE BELOW THIS LINE AND IT WILL USE THE CONSTANTS DEFINED IN CONSTANTS.PY 
# YOU CAN REPLACE THE CONSTANTS ABOVE WITH HARD-CODED VALUES AND IT WILL CONTINUE TO WORK, BUT THE CONSTANTS IN CONSTANTS.PY WILL BECOME DEAD CODE.
# ----------------------------------------------------------------#


class GeneAlgo:
    ACTIONMAT = np.array([
        [0, 0, -1], [0, 0, 0], [0, 0, 1],
        [0, -1, -1], [0, -1, 0], [0, -1, 1],
        [0, 1, -1], [0, 1, 0], [0, 1, 1],
        [-1, 0, -1], [-1, 0, 0], [-1, 0, 1],
        [-1, -1, -1], [-1, -1, 0], [-1, -1, 1],
        [-1, 1, -1], [-1, 1, 0], [-1, 1, 1],
        [1, 0, -1], [1, 0, 0], [1, 0, 1],
        [1, -1, -1], [1, -1, 0], [1, -1, 1],
        [1, 1, -1], [1, 1, 0], [1, 1, 1],
    ], dtype=float)

    PLANAR_COLLISION_VOLUMES = GA_COLLISION_VOLUMES
    INITIAL_POS = np.array(GA_INITIAL_POS, dtype=float)

    def __init__(self, viz_enabled=False, viz_callback=None, viz_skip_gens=0, yaw_deg=0.0):
        """Initialize GA runtime state, goal defaults, and optional visualization callback settings."""
        self.goal = np.array([0.0, 0.0, Z_GLOBAL_OFFSET_MM], dtype=float)
        self.yaw_deg = float(yaw_deg)
        self.viz_enabled = bool(viz_enabled)
        self.viz_callback = viz_callback
        self.viz_skip_gens = max(0, int(viz_skip_gens))
        self.last_solution_success_percent = 0.0
        self.last_solution_min_distance_mm = None
        self.last_solution_fitness = None

    ### STEP 0: SET YAW FROM MAIN FUNCTION IF PROVIDED, OTHERWISE DEFAULT TO 0.0
    def setYaw(self, yaw_deg):
        """Set fixed base yaw used by `calHandPosition()` and `_build_motor_sets()`."""
        self.yaw_deg = float(yaw_deg)

    ### STEP 2: SET GOAL (X, Y, Z) IN MILLIMETRES FROM MAIN FUNCTION
    def setGoal(self, goal):
        """Store XYZ goal and derive yaw from XY azimuth for downstream FK simulation."""
        goal_vector = np.asarray(goal, dtype=float)
        if goal_vector.shape != (3,):
            raise ValueError("Goal must be a 3-element vector [X, Y, Z] in millimetres.")
        self.goal = goal_vector
        planar_norm = float(np.hypot(goal_vector[0], goal_vector[1]))
        if planar_norm <= 1e-9:
            self.yaw_deg = 0.0
        else:
            self.yaw_deg = float(np.degrees(np.arctan2(goal_vector[1], goal_vector[0])))

    def _simulate_action(self, state, action, step_size):
        """Apply one decoded action step, then evaluate limits/collision via `calHandPosition()` and `isWithin()`."""
        new_state = state + action * step_size
        position_xyz = self.calHandPosition(new_state)

        within_joint_limits = (
            GA_JOINT1_LIMITS[0] <= new_state[0] <= GA_JOINT1_LIMITS[1]
            and GA_JOINT2_LIMITS[0] <= new_state[1] <= GA_JOINT2_LIMITS[1]
            and GA_JOINT3_LIMITS[0] <= new_state[2] <= GA_JOINT3_LIMITS[1]
        )

        collision = self.isWithin(position_xyz)
        valid = within_joint_limits and not collision

        distance_to_goal = float(np.linalg.norm(self.goal - position_xyz))
        pose_score = self._pose_score(new_state)

        return {
            "state": new_state,
            "position": position_xyz,
            "distance": distance_to_goal,
            "pose_score": pose_score,
            "valid": valid,
        }

    def _pose_score(self, state):
        """Compute orthogonality score for one state (1.0 is vertical, 0.0 worst)."""
        theta = -state[0] + state[1] - state[2]
        return max(0.0, 1.0 - abs(theta - 180.0) / 180.0)

    def _distance_to_success_percent(self, min_distance):
        """Map minimum distance into an intuitive 0-100 success percentage for progress reporting."""
        if min_distance is None:
            return 0.0
        if min_distance <= 0.0:
            return 100.0

        normalized = max(0.0, min(1.0, 1.0 - (float(min_distance) / float(DISTANCE_SCALE_MM))))
        return (normalized ** 3) * 100.0

    ### STEP 10: CALCULATE WHERE THE HAND ENDS UP FOR THE BEST MOTOR SET USING FORWARD KINEMATICS
    def calHandPosition(self, state):
        """Run forward kinematics for [m1,m2,m3] and rotate into global XYZ using current yaw."""
        m1, m2, m3 = state
        rad1 = -np.deg2rad(m1)
        rad2 = np.deg2rad(m2)
        rad3 = -np.deg2rad(m3)
        yaw_rad = np.deg2rad(self.yaw_deg)

        d1, d2, d3 = GA_LINK_1_MM, GA_LINK_2_MM, GA_LINK_3_MM
        joint1_x = d1 * sin(rad1)
        joint1_z = d1 * cos(rad1)
        joint2_x = d2 * sin(rad1 + rad2) + joint1_x
        joint2_z = d2 * cos(rad1 + rad2) + joint1_z
        hand_x_local = d3 * sin(rad1 + rad2 + rad3) + joint2_x
        hand_z_local = d3 * cos(rad1 + rad2 + rad3) + joint2_z
        hand_z = hand_z_local + Z_GLOBAL_OFFSET_MM
        hand_x = hand_x_local * cos(yaw_rad)
        hand_y = hand_x_local * sin(yaw_rad)

        return np.array([hand_x, hand_y, hand_z], dtype=float)

    def isWithin(self, position_xyz):
        """Return True when XYZ lies inside any configured planar collision volume."""
        x_coord = float(position_xyz[0])
        z_coord = float(position_xyz[2])
        for min_x, min_z, max_x, max_z in self.PLANAR_COLLISION_VOLUMES:
            if min_x < x_coord < max_x and min_z < z_coord < max_z:
                return True
        return False

    ### STEP 3: MAKE INITIAL POPULATION OF NEW INDIVIDUAL CHROMOSOMES 
    def make_new_individual(self):
        """Create one random chromosome of random length from the action alphabet."""
        min_len, max_len = INITIAL_GENE_LENGTH_RANGE
        length = random.randint(int(min_len), int(max_len))
        return [random.randint(0, len(self.ACTIONMAT) - 1) for _ in range(length)]

    # ---------------------------------------------------------
    # UNIFIED REWARD FUNCTION — EDIT THIS TO CHANGE BEHAVIOR
    # ---------------------------------------------------------
    def _compute_reward(self, min_distance, best_state, avg_non_ortho_pose, chromosome_length):
        """Compute distance/pose/length components and return the combined chromosome fitness dict."""

        # catch for invalid solutions 
        if min_distance is None:
            length_penalty = LENGTH_PENALTY_WEIGHT * chromosome_length
            pose_score = self._pose_score(best_state) if best_state is not None else 0.0
            pose_reward = POSE_WEIGHT * pose_score
            non_ortho_approach_penalty = NON_ORTHO_APPROACH_WEIGHT * float(avg_non_ortho_pose)
            return {
                "minimum_distance": None,
                "pose_score": pose_score,
                "avg_non_ortho_pose": float(avg_non_ortho_pose),
                "distance_reward": 0.0,
                "pose_reward": pose_reward,
                "non_ortho_approach_penalty": non_ortho_approach_penalty,
                "length_penalty": length_penalty,
                "fitness": pose_reward - length_penalty - non_ortho_approach_penalty,
            }
        



        #################################################################################
        ############################## DISTANCE REWARD ##################################

        # CHANGE THIS REWARD TO A REWARD FOR BEING CLOSE TO THE TARGET
        # min_distance is the closest distance the chromosome got to the target during its simulation

        
        normalized_closeness = max(0.0, (DISTANCE_SCALE_MM - min_distance) / DISTANCE_SCALE_MM)
        # normalized_closeness = (DISTANCE_WEIGHT * (DISTANCE_SCALE_MM - min_distance) / DISTANCE_SCALE_MM)**2
        distance_reward = normalized_closeness * DISTANCE_WEIGHT


        #################################################################################





        #################################################################################
        ##############################   POSE REWARD   ##################################
        # Pose reward depends only on the final gene state.
        pose_score = self._pose_score(best_state) if best_state is not None else 0.0
        pose_reward = (POSE_WEIGHT * pose_score) ** 2

        # Penalize non-orthogonal approach over the full path.
        non_ortho_approach_penalty = NON_ORTHO_APPROACH_WEIGHT * float(avg_non_ortho_pose)

        #################################################################################





        #################################################################################
        ##############################  LENGTH PENALTY  #################################
        # Length penalty
        length_penalty = LENGTH_PENALTY_WEIGHT * chromosome_length # CHANGE THIS TO PENELIZE LONGER CHROMOSOMES (Function of chromosome_length)

        #################################################################################




        #################################################################################
        ########################## TOTAL CHROMOSOME FITNESS #############################

        # Add together your rewards and penalties to compute a single fitness score for this chromosome.
        fitness = distance_reward + pose_reward - length_penalty - non_ortho_approach_penalty

        #################################################################################




        return {
            "minimum_distance": min_distance,
            "pose_score": float(pose_score),
            "avg_non_ortho_pose": float(avg_non_ortho_pose),
            "distance_reward": distance_reward,
            "pose_reward": pose_reward,
            "non_ortho_approach_penalty": non_ortho_approach_penalty,
            "length_penalty": length_penalty,
            "fitness": fitness,
        }
    # ---------------------------------------------------------

    def _compose_chromosome_fitness(self, best_record, best_state, avg_non_ortho_pose, chromosome_length):
        """Bridge simulation output into `_compute_reward()` inputs for one chromosome."""
        if best_record is None:
            return self._compute_reward(None, best_state, avg_non_ortho_pose, chromosome_length)

        return self._compute_reward(
            best_record["distance"],
            best_state,
            avg_non_ortho_pose,
            chromosome_length
        )

    ### STEP 5: EVALUATE THE CHROMOSOMES AND FIND WHEN THE CHROMOSOME GETS CLOSEST TO THE TARGET
    def evaluate_chromosome(self, chromosome, step_size):
        """Roll out one chromosome with `_simulate_action()`, optional trim, then fitness composition."""
        state = self.INITIAL_POS.copy()
        best_record = None
        trim_length = None

        for gene_index, gene in enumerate(chromosome, start=1):
    ### STEP 6: IF SELECTED, SIMULATE THE ACTIONS IN RVIZ
            step_record = self._simulate_action(state, self.ACTIONMAT[gene], step_size)
            state = step_record["state"]

            if step_record["valid"]:
                if best_record is None or step_record["distance"] < best_record["distance"]:
                    best_record = step_record

                if trim_length is None and step_record["distance"] <= TARGET_DISTANCE_THRESHOLD_MM:
                    trim_length = gene_index

        trimmed_chromosome = list(chromosome if trim_length is None else chromosome[:trim_length])
        best_state = self.INITIAL_POS.copy()
        pose_scores_per_gene = []
        for gene in trimmed_chromosome:
            best_state = best_state + self.ACTIONMAT[gene] * step_size
            pose_scores_per_gene.append(self._pose_score(best_state))

        avg_pose_score = float(np.mean(pose_scores_per_gene)) if pose_scores_per_gene else 0.0
        avg_non_ortho_pose = 1.0 - avg_pose_score
        fitness_components = self._compose_chromosome_fitness(
            best_record,
            best_state,
            avg_non_ortho_pose,
            len(trimmed_chromosome),
        )

        return {
            "chromosome": trimmed_chromosome,
            "best_state": best_state,
            "best_record": best_record,
            "fitness_components": fitness_components,
        }

    ### STEP 4: COMPUTE THE FITNESS FOR EACH CHROMOSOME IN THE POPULATION
    def computeFitness(self, population, step_size):
        """Evaluate the full population and return ranked fitness metadata plus trimmed chromosomes."""
        fitness_list = []
        chromosome_list = []
        evaluation_list = []

        for index, chromosome in enumerate(population):
            evaluation = self.evaluate_chromosome(chromosome, step_size)
            trimmed_chromosome = evaluation["chromosome"]
            population[index] = trimmed_chromosome

            fitness_list.append([evaluation["fitness_components"]["fitness"], index])
            chromosome_list.append(trimmed_chromosome)
            evaluation_list.append(evaluation)

        return fitness_list, chromosome_list, evaluation_list

    ### STEP 9: BUILD THE MOTOR SETS FOR THE BEST CHROMOSOMES
    def _build_motor_sets(self, chromosome, step_size):
        """Decode chromosome genes into cumulative [yaw,m1,m2,m3] waypoint commands."""
        motor_sets = np.zeros((len(chromosome), 4), dtype=float)
        state = self.INITIAL_POS.copy()

        for index, gene in enumerate(chromosome):
            state = state + self.ACTIONMAT[gene] * step_size
            motor_sets[index, 0] = float(self.yaw_deg)
            motor_sets[index, 1:] = state

        return motor_sets
    
    ### Two different styles for selection: Tournament & Roulette.
    def select_parent_tournament(self, ranked_fitness):
        """Select one parent by k-way tournament over ranked fitness entries."""
        sample_size = min(TOURNAMENT_K, len(ranked_fitness))
        contestants = random.sample(ranked_fitness, sample_size)
        return max(contestants, key=lambda item: item[0])

    def select_parent_roulette(self, ranked_fitness):
        """Select one parent with probability proportional to shifted non-negative fitness."""
        if not ranked_fitness:
            return None
        min_fit = min(item[0] for item in ranked_fitness)
        offset = -min_fit + 1e-6 if min_fit <= 0.0 else 0.0
        weights = [item[0] + offset for item in ranked_fitness]
        total_weight = sum(weights)
        if total_weight <= 0.0:
            return random.choice(ranked_fitness)

        draw = random.uniform(0.0, total_weight)
        for item, weight in zip(ranked_fitness, weights):
            draw -= weight
            if draw <= 0.0:
                return item
        return ranked_fitness[-1]

    def select_parent(self, ranked_fitness):
        """Dispatch parent selection strategy based on `SELECTION_METHOD`."""
        if SELECTION_METHOD == "roulette":
            return self.select_parent_roulette(ranked_fitness)
        return self.select_parent_tournament(ranked_fitness)

    def crossover(self, parent_a, parent_b):
        """Create one child via single-point crossover of two parent chromosomes."""
        if not parent_a:
            return list(parent_b)
        if not parent_b:
            return list(parent_a)
        crossover_index = random.randint(0, min(len(parent_a), len(parent_b)))
        return list(parent_a[:crossover_index] + parent_b[crossover_index:])

    def mutate(self, chromosome):
        """Apply per-gene random mutation using `MUTATION_RATE`."""
        mutated = list(chromosome)
        for index in range(len(mutated)):
            if random.random() < MUTATION_RATE:
                mutated[index] = random.randint(0, len(self.ACTIONMAT) - 1)
        return mutated

    ### STEP 11: BUILD THE NEXT GENERATION OF CHROMOSOMES USING SELECTION, CROSSOVER, AND MUTATION 
    ### THEN THE STEPS REPEAT FROM STEP 4
    def build_next_generation(self, population, ranked_fitness):
        """Build next population via elitism, parent selection, crossover/mutation, and random backfill."""
        population_size = len(population)
        elite_count = max(1, min(population_size, int(round(population_size * ELITISM_PERCENT))))
        backfill_count = min(population_size - elite_count, int(round(population_size * RANDOM_BACKFILL_PERCENT)))

        new_population = [list(population[item[1]]) for item in ranked_fitness[:elite_count]]

        while len(new_population) < population_size - backfill_count:
            parent_one = self.select_parent(ranked_fitness)
            parent_two = self.select_parent(ranked_fitness)
            if parent_one is None or parent_two is None:
                new_population.append(self.make_new_individual())
                continue
            child = self.crossover(population[parent_one[1]], population[parent_two[1]])
            child = self.mutate(child)
            if not child:
                child = [random.randint(0, len(self.ACTIONMAT) - 1)]
            new_population.append(child)

        while len(new_population) < population_size:
            new_population.append(self.make_new_individual())

        return new_population

    ### STEP 8: UPDATE THE GENERATION AFTER EVALUATING THE CHROMOSOMES, FIND THE BEST
    def _handle_generation_update(self, generation_index, epochs, best_fitness, best_chromosome, evaluation, step_size):
        """Emit generation diagnostics, build best motor sets, and publish to viz callback when enabled."""
        motor_sets = self._build_motor_sets(best_chromosome, step_size)
        if motor_sets.size == 0:
            return motor_sets

        if generation_index % (self.viz_skip_gens + 1) == 0:
            best_record = evaluation["best_record"]
            best_position = self.calHandPosition(motor_sets[-1][1:4])
            if best_record is not None:
                best_position = best_record["position"]
            fitness_components = evaluation["fitness_components"]
            print(
                f"Gen {generation_index + 1:>3}/{epochs} | "
                f"Fitness: {best_fitness:8.3f} | "
                f"Len: {len(best_chromosome):3d} | "
                f"Goal XYZ: [{self.goal[0]:7.2f}, {self.goal[1]:7.2f}, {self.goal[2]:7.2f}] mm | "
                f"Pos XYZ: [{best_position[0]:7.2f}, {best_position[1]:7.2f}, {best_position[2]:7.2f}] mm | "
                f"MinDist: {fitness_components['minimum_distance'] if fitness_components['minimum_distance'] is not None else float('nan'):7.2f} mm | "
                f"Success: {self._distance_to_success_percent(fitness_components['minimum_distance']):6.2f}% | "
                f"Rdist: {fitness_components['distance_reward']:7.2f} | "
                f"Rpose: {fitness_components['pose_reward']:7.2f} | "
                f"PnonOrtho: {fitness_components['non_ortho_approach_penalty']:7.2f} | "
                f"Penalty: {fitness_components['length_penalty']:7.2f} | "
                f"Collision: {self.isWithin(best_position)}"
            )
            if self.viz_enabled and callable(self.viz_callback):
                self.viz_callback(motor_sets.copy())

        return motor_sets

    def _save_fitness_plot(self, generations, best_fitness_history, avg_fitness_history):
        """Persist best/average fitness curves for offline tuning when plotting is enabled."""
        if not FITNESS_PLOT_ENABLED or not generations:
            return

        try:
            import matplotlib.pyplot as plt
        except Exception:
            return

        fig, axis = plt.subplots(figsize=(8, 4.5), dpi=FITNESS_PLOT_DPI)
        axis.plot(generations, best_fitness_history, label="Best Fitness", linewidth=2.0)
        axis.plot(generations, avg_fitness_history, label="Average Fitness", linewidth=1.8)
        axis.set_xlabel("Generation")
        axis.set_ylabel("Fitness")
        axis.set_title("GA Fitness vs Generation")
        axis.grid(True, linestyle="--", linewidth=0.6, alpha=0.5)
        axis.legend()
        fig.tight_layout()
        fig.savefig(FITNESS_PLOT_FILENAME, dpi=FITNESS_PLOT_DPI)
        plt.close(fig)


    ### STEP 1: CALL SOLVE FROM THE MAIN PYTHON FILE

    def solve(self, goal, population_size=None, step_size=None, epochs=None, yaw_deg=None, convergence_check=None, convergence_tolerance=None):
        """Run the end-to-end GA loop by chaining evaluation, reporting, and `build_next_generation()`."""
        self.setGoal(goal)
        if yaw_deg is not None:
            self.setYaw(yaw_deg) # Set yaw if it hasn't been set yet

        self.last_solution_success_percent = 0.0
        self.last_solution_min_distance_mm = None
        self.last_solution_fitness = None

        if population_size is None:
            population_size = POPULATION_SIZE
        if step_size is None:
            step_size = STEP_SIZE
        if epochs is None:
            epochs = NUM_GENERATIONS
        if convergence_check is None:
            convergence_check = CONVERGENCE_CHECK
        if convergence_tolerance is None:
            convergence_tolerance = CONVERGENCE_TOLERANCE

        population_size = int(population_size)
        step_size = float(step_size)
        convergence_check = bool(convergence_check)
        convergence_tolerance = float(convergence_tolerance)
        if population_size <= 0:
            raise ValueError("population_size must be a positive integer.")
        if step_size <= 0.0:
            raise ValueError("step_size must be positive.")

        population = [self.make_new_individual() for _ in range(population_size)]
        best_motor_sets = np.zeros((1, 4), dtype=float)
        generation_history = []
        best_fitness_history = []
        avg_fitness_history = []
        generations_without_improvement = 0

        for generation_index in range(epochs):
            fitness_list, chromosome_list, evaluation_list = self.computeFitness(population, step_size)
            ranked_fitness = sorted(fitness_list, reverse=True)
            best_index = ranked_fitness[0][1]
            best_fitness = ranked_fitness[0][0]
            avg_fitness = float(np.mean([item[0] for item in fitness_list]))
            best_chromosome = chromosome_list[best_index]
            best_evaluation = evaluation_list[best_index]

            minimum_distance = best_evaluation["fitness_components"]["minimum_distance"]
            self.last_solution_min_distance_mm = None if minimum_distance is None else float(minimum_distance)
    ### STEP 7: AFTER EVALUATING THE CHROMOSOMES, UPDATE THE BEST SOLUTION AND FITNESS, AND PLOT IF DESIRED
            self.last_solution_success_percent = self._distance_to_success_percent(minimum_distance)
            self.last_solution_fitness = float(best_fitness)

            generation_history.append(generation_index + 1)
            best_fitness_history.append(float(best_fitness))
            avg_fitness_history.append(avg_fitness)

            # Check for convergence
            if convergence_check and len(best_fitness_history) > 10:
                fitness_10_gens_ago = best_fitness_history[-11]
                if abs(best_fitness - fitness_10_gens_ago) < convergence_tolerance:
                    generations_without_improvement += 1
                    if generations_without_improvement >= 10:
                        break
                else:
                    generations_without_improvement = 0

            best_motor_sets = self._handle_generation_update(
                generation_index,
                epochs,
                best_fitness,
                best_chromosome,
                best_evaluation,
                step_size,
            )

            population = self.build_next_generation(population, ranked_fitness)

        self._save_fitness_plot(generation_history, best_fitness_history, avg_fitness_history)
        return best_motor_sets