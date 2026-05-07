import random
from math import cos, sin

import numpy as np


ELITISM_PERCENT = 0.20  # Fraction of best chromosomes preserved each generation
MUTATION_RATE = 0.10  # Probability of gene mutation per position
SELECTION_METHOD = "tournament"  # Parent selection method: "tournament" or "roulette"
TOURNAMENT_K = 3  # Number of contestants in tournament selection
TARGET_DISTANCE_THRESHOLD_MM = 1.0  # Distance threshold for trimming chromosomes
POPULATION_SIZE = 150  # Number of chromosomes per generation
RANDOM_BACKFILL_PERCENT = 0.15  # Fraction of new random individuals per generation
EE_Z_OFFSET_MM = 200.0  # End-effector Z-axis offset in millimeters
FF_X_OFFSET_MM = 25.0  # Forward-facing X-axis offset in millimeters to account for hand geometry and reachability
STEP_SIZE = 2  # Step size for each action in joint space
NUM_GENERATIONS = 100 # Default number of generations to evolve
INITIAL_GENE_LENGTH_RANGE = (10, 60)  # Range for initial chromosome length

DISTANCE_WEIGHT = 95.0  # Weight for distance-to-goal reward
POSE_WEIGHT = 5.0  # Weight for pose correctness reward
LENGTH_PENALTY_WEIGHT = 0.5  # Weight for chromosome length penalty
DISTANCE_SCALE_MM = 500.0  # Scaling factor for distance-based fitness calculation

CONVERGENCE_CHECK = True  # Enable early stopping on convergence 
CONVERGENCE_TOLERANCE = 0.01  # Minimum fitness improvement threshold. Will exit if improvement is less than this value for 10 consecutive generations.
POSITION_CONVERGENCE_PATIENCE = 20  # Stop if best end position changes very little for this many consecutive generations.
POSITION_CONVERGENCE_TOLERANCE_MM = 0.5  # Maximum change in best end position (mm) considered "no change".

FITNESS_PLOT_ENABLED = False  # Enable fitness history plotting
FITNESS_PLOT_FILENAME = "ga_fitness_history.png"  # Output plot filename
FITNESS_PLOT_DPI = 300  # Resolution for saved plot


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

    PLANAR_COLLISION_VOLUMES = [[-500.0, 0.0, 133.0, 198.0]]
    INITIAL_POS = np.array([-14.25, 76.75, 0.0], dtype=float)

    def __init__(self, viz_enabled=False, viz_callback=None, viz_skip_gens=0, yaw_deg=0.0):
        self.goal = np.array([0.0, 0.0, 0.0], dtype=float)
        self.yaw_deg = float(yaw_deg)
        self.viz_enabled = bool(viz_enabled)
        self.viz_callback = viz_callback
        self.viz_skip_gens = max(0, int(viz_skip_gens))
        self.last_solution_success_percent = 0.0
        self.last_solution_min_distance_mm = None
        self.last_solution_fitness = None

    ### STEP 0: SET YAW FROM MAIN FUNCTION IF PROVIDED, OTHERWISE DEFAULT TO 0.0
    def setYaw(self, yaw_deg):
        self.yaw_deg = float(yaw_deg)

    ### STEP 2: SET GOAL (X, Y, Z) IN MILLIMETRES FROM MAIN FUNCTION
    def setGoal(self, goal):
        goal_vector = np.asarray(goal, dtype=float)
        if goal_vector.shape != (3,):
            raise ValueError("Goal must be a 3-element vector [X, Y, Z] in millimetres.")
        self.goal = goal_vector
        self.goal[0] += FF_X_OFFSET_MM  # Apply X offset to account for hand geometry and reachability
        self.goal[2] -= EE_Z_OFFSET_MM  # Apply end-effector Z offset so arm reaches ground instead of hovering above it
        planar_norm = float(np.hypot(goal_vector[0], goal_vector[1]))
        if planar_norm <= 1e-9:
            self.yaw_deg = 0.0
        else:
            self.yaw_deg = float(np.degrees(np.arctan2(goal_vector[1], goal_vector[0])))

    def _simulate_action(self, state, action, step_size):
        new_state = state + action * step_size
        position_xyz = self.calHandPosition(new_state)

        within_joint_limits = (
            -90.0 <= new_state[0] <= 90.0
            and -120.0 <= new_state[1] <= 120.0
            and -120.0 <= new_state[2] <= 120.0
        )

        collision = self.isWithin(position_xyz)
        valid = within_joint_limits and not collision

        distance_to_goal = float(np.linalg.norm(self.goal - position_xyz))

        return {
            "state": new_state,
            "position": position_xyz,
            "distance": distance_to_goal,
            "valid": valid,
        }

    def _distance_to_success_percent(self, min_distance):
        if min_distance is None:
            return 0.0
        if min_distance <= 0.0:
            return 100.0

        normalized = max(0.0, min(1.0, 1.0 - (float(min_distance) / float(DISTANCE_SCALE_MM))))
        return (normalized ** 3) * 100.0

    def _compute_step_pose_score(self, step_state, step_position):
        """Compute per-step pose score used for chromosome trajectory averaging."""
        theta = -step_state[0] + step_state[1] - step_state[2]
        wrist_orientation_score = max(0.0, 1.0 - abs(theta - 180.0) / 180.0)

        distance_scale = max(1e-6, float(DISTANCE_SCALE_MM))
        step_pose_error_mm = float(np.linalg.norm(np.asarray(step_position, dtype=float) - self.goal))
        step_position_score = max(0.0, 1.0 - (step_pose_error_mm / distance_scale))

        # Favor perpendicular wrist orientation while still rewarding proximity.
        return 0.65 * wrist_orientation_score + 0.35 * step_position_score

    ### STEP 10: CALCULATE WHERE THE HAND ENDS UP FOR THE BEST MOTOR SET USING FORWARD KINEMATICS
    def calHandPosition(self, state):
        m1, m2, m3 = state
        rad1 = -np.deg2rad(m1)
        rad2 = np.deg2rad(m2)
        rad3 = -np.deg2rad(m3)
        yaw_rad = np.deg2rad(self.yaw_deg)

        d1, d2, d3 = 102.0, 98.0, 155.0
        joint1_x = d1 * sin(rad1)
        joint1_z = d1 * cos(rad1)
        joint2_x = d2 * sin(rad1 + rad2) + joint1_x
        joint2_z = d2 * cos(rad1 + rad2) + joint1_z
        hand_x_local = d3 * sin(rad1 + rad2 + rad3) + joint2_x
        hand_z = d3 * cos(rad1 + rad2 + rad3) + joint2_z
        hand_x = hand_x_local * cos(yaw_rad)
        hand_y = hand_x_local * sin(yaw_rad)

        return np.array([hand_x, hand_y, hand_z], dtype=float)

    def isWithin(self, position_xyz):
        x_coord = float(position_xyz[0])
        z_coord = float(position_xyz[2])
        for min_x, min_z, max_x, max_z in self.PLANAR_COLLISION_VOLUMES:
            if min_x < x_coord < max_x and min_z < z_coord < max_z:
                return True
        return False

    ### STEP 3: MAKE INITIAL POPULATION OF NEW INDIVIDUAL CHROMOSOMES 
    def make_new_individual(self):
        min_len, max_len = INITIAL_GENE_LENGTH_RANGE
        length = random.randint(int(min_len), int(max_len))
        return [random.randint(0, len(self.ACTIONMAT) - 1) for _ in range(length)]

    # ---------------------------------------------------------
    # UNIFIED REWARD FUNCTION — EDIT THIS TO CHANGE BEHAVIOR
    # ---------------------------------------------------------
    def _compute_reward(self, min_distance, average_pose_score, chromosome_length):
        """
        Unified reward function.
        Modify this function to change how the GA evaluates solutions.
        """

        # catch for invalid solutions 
        if min_distance is None:
            length_penalty = 0
            return {
                "minimum_distance": None,
                "pose_score": 0.0,
                "distance_reward": 0.0,
                "pose_reward": 0.0,
                "length_penalty": length_penalty,
                "fitness": -length_penalty,
            }
        



        #################################################################################
        ############################## DISTANCE REWARD ##################################

        # CHANGE THIS REWARD TO A REWARD FOR BEING CLOSE TO THE TARGET
        # min_distance is the closest distance the chromosome got to the target during its simulation

        distance_reward = (5.25 *(500 - min_distance) / 500)**2


        #################################################################################





        #################################################################################
        ##############################   POSE REWARD   ##################################
        # Pose reward uses average pose quality across all genes in this chromosome.
        pose_score = max(0.0, float(average_pose_score))

        pose_reward = (5.0 * pose_score) ** 2

        #################################################################################





        #################################################################################
        ##############################  LENGTH PENALTY  #################################
        # Length penalty
        length_penalty = 0.3 * chromosome_length # CHANGE THIS TO PENALIZE LONGER CHROMOSOMES (Function of chromosome_length)

        #################################################################################




        #################################################################################
        ########################## TOTAL CHROMOSOME FITNESS #############################

        # Add together your rewards and penalties to compute a single fitness score for this chromosome.
        fitness = distance_reward + pose_reward - length_penalty

        #################################################################################




        return {
            "minimum_distance": min_distance,
            "pose_score": pose_score,
            "distance_reward": distance_reward,
            "pose_reward": pose_reward,
            "length_penalty": length_penalty,
            "fitness": fitness,
        }
    # ---------------------------------------------------------

    def _compose_chromosome_fitness(self, best_record, average_pose_score, chromosome_length):
        if best_record is None:
            return self._compute_reward(None, average_pose_score, chromosome_length)

        return self._compute_reward(
            best_record["distance"],
            average_pose_score,
            chromosome_length,
        )

    ### STEP 5: EVALUATE THE CHROMOSOMES AND FIND WHEN THE CHROMOSOME GETS CLOSEST TO THE TARGET
    def evaluate_chromosome(self, chromosome, step_size, generation_index):
        state = self.INITIAL_POS.copy()
        best_record = None
        trim_length = None
        pose_scores_per_step = []

        for action_index, gene in enumerate(chromosome, start=1):
    ### STEP 6: IF SELECTED, SIMULATE THE ACTIONS IN RVIZ
            step_record = self._simulate_action(state, self.ACTIONMAT[gene], step_size)
            state = step_record["state"]
            pose_scores_per_step.append(
                self._compute_step_pose_score(step_record["state"], step_record["position"])
            )

            if step_record["valid"]:
                if best_record is None or step_record["distance"] < best_record["distance"]:
                    best_record = step_record

                if trim_length is None and step_record["distance"] <= TARGET_DISTANCE_THRESHOLD_MM:
                    trim_length = action_index

        trimmed_chromosome = list(chromosome if trim_length is None else chromosome[:trim_length])
        best_state = self.INITIAL_POS.copy() if best_record is None else best_record["state"].copy()
        final_pose = self.calHandPosition(state)
        average_pose_score = float(np.mean(pose_scores_per_step)) if pose_scores_per_step else 0.0
        fitness_components = self._compose_chromosome_fitness(
            best_record,
            average_pose_score,
            len(trimmed_chromosome),
        )

        return {
            "chromosome": trimmed_chromosome,
            "best_state": best_state,
            "final_pose": final_pose,
            "average_pose_score": average_pose_score,
            "pose_sample_count": len(pose_scores_per_step),
            "best_record": best_record,
            "fitness_components": fitness_components,
        }

    ### STEP 4: COMPUTE THE FITNESS FOR EACH CHROMOSOME IN THE POPULATION
    def computeFitness(self, population, step_size, generation_index):
        fitness_list = []
        chromosome_list = []
        evaluation_list = []

        for gene_index, chromosome in enumerate(population):
            evaluation = self.evaluate_chromosome(chromosome, step_size, generation_index)
            trimmed_chromosome = evaluation["chromosome"]
            population[gene_index] = trimmed_chromosome
            evaluation["gene_index"] = int(gene_index)

            fitness_list.append([evaluation["fitness_components"]["fitness"], gene_index])
            chromosome_list.append(trimmed_chromosome)
            evaluation_list.append(evaluation)

        return fitness_list, chromosome_list, evaluation_list

    ### STEP 9: BUILD THE MOTOR SETS FOR THE BEST CHROMOSOMES
    def _build_motor_sets(self, chromosome, step_size):
        motor_sets = np.zeros((len(chromosome), 4), dtype=float)
        state = self.INITIAL_POS.copy()

        for index, gene in enumerate(chromosome):
            state = state + self.ACTIONMAT[gene] * step_size
            motor_sets[index, 0] = float(self.yaw_deg)
            motor_sets[index, 1:] = state

        return motor_sets
    
    ### Two different styles for selection: Tournament & Roulette.
    def select_parent_tournament(self, ranked_fitness):
        sample_size = min(TOURNAMENT_K, len(ranked_fitness))
        contestants = random.sample(ranked_fitness, sample_size)
        return max(contestants, key=lambda item: item[0])

    def select_parent_roulette(self, ranked_fitness):
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
        if SELECTION_METHOD == "roulette":
            return self.select_parent_roulette(ranked_fitness)
        return self.select_parent_tournament(ranked_fitness)

    def crossover(self, parent_a, parent_b):
        if not parent_a:
            return list(parent_b)
        if not parent_b:
            return list(parent_a)
        crossover_index = random.randint(0, min(len(parent_a), len(parent_b)))
        return list(parent_a[:crossover_index] + parent_b[crossover_index:])

    def mutate(self, chromosome):
        mutated = list(chromosome)
        for index in range(len(mutated)):
            if random.random() < MUTATION_RATE:
                mutated[index] = random.randint(0, len(self.ACTIONMAT) - 1)
        return mutated

    ### STEP 11: BUILD THE NEXT GENERATION OF CHROMOSOMES USING SELECTION, CROSSOVER, AND MUTATION 
    ### THEN THE STEPS REPEAT FROM STEP 4
    def build_next_generation(self, population, ranked_fitness):
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
                f"AvgPose: {evaluation['average_pose_score']:6.3f} ({evaluation['pose_sample_count']:3d}) | "
                f"Rdist: {fitness_components['distance_reward']:7.2f} | "
                f"Rpose: {fitness_components['pose_reward']:7.2f} | "
                f"Penalty: {fitness_components['length_penalty']:7.2f} | "
                f"Collision: {self.isWithin(best_position)}"
            )
            if self.viz_enabled and callable(self.viz_callback):
                self.viz_callback(motor_sets.copy())

        return motor_sets

    def _save_fitness_plot(self, generations, best_fitness_history, avg_fitness_history):
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
        fitness_stagnation_count = 0
        position_stagnation_count = 0
        previous_best_position = None

        for generation_index in range(epochs):
            generation_count = generation_index + 1
            fitness_list, chromosome_list, evaluation_list = self.computeFitness(
                population,
                step_size,
                generation_count,
            )
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

            # Check for fitness convergence.
            if convergence_check and len(best_fitness_history) > 10:
                fitness_10_gens_ago = best_fitness_history[-11]
                if abs(best_fitness - fitness_10_gens_ago) < convergence_tolerance:
                    fitness_stagnation_count += 1
                    if fitness_stagnation_count >= 10:
                        print(
                            f"Early stop at generation {generation_index + 1}: "
                            f"fitness change < {convergence_tolerance} for 10 consecutive checks."
                        )
                        break
                else:
                    fitness_stagnation_count = 0

            # Check for position convergence (best end-effector position stability).
            if convergence_check:
                best_record = best_evaluation["best_record"]
                if best_record is not None:
                    current_best_position = np.asarray(best_record["position"], dtype=float)
                else:
                    current_best_position = self.calHandPosition(np.asarray(best_evaluation["best_state"], dtype=float))

                if previous_best_position is not None:
                    position_delta_mm = float(np.linalg.norm(current_best_position - previous_best_position))
                    if position_delta_mm < POSITION_CONVERGENCE_TOLERANCE_MM:
                        position_stagnation_count += 1
                        if position_stagnation_count >= POSITION_CONVERGENCE_PATIENCE:
                            print(
                                f"Early stop at generation {generation_index + 1}: "
                                f"best position change < {POSITION_CONVERGENCE_TOLERANCE_MM:.3f} mm "
                                f"for {POSITION_CONVERGENCE_PATIENCE} consecutive generations."
                            )
                            break
                    else:
                        position_stagnation_count = 0

                previous_best_position = current_best_position.copy()

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