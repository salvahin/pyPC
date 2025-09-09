"""
Multi-Objective Optimization Solver with Various Algorithms
Target: 30-45% coverage, ~75 cyclomatic complexity
"""

import math
import time
import random
from typing import List, Dict, Any, Optional, Tuple, Callable, Union
from dataclasses import dataclass
from enum import Enum

class OptimizationMethod(Enum):
    GENETIC_ALGORITHM = "genetic_algorithm"
    PARTICLE_SWARM = "particle_swarm"
    SIMULATED_ANNEALING = "simulated_annealing"
    DIFFERENTIAL_EVOLUTION = "differential_evolution"
    GRADIENT_DESCENT = "gradient_descent"
    NELDER_MEAD = "nelder_mead"

class ConstraintType(Enum):
    EQUALITY = "equality"
    INEQUALITY = "inequality"
    BOUNDS = "bounds"
    NONLINEAR = "nonlinear"

@dataclass
class OptimizationResult:
    success: bool
    solution: List[float]
    objective_value: float
    iterations: int
    function_evaluations: int
    convergence_history: List[float]
    constraint_violations: List[float]
    execution_time: float
    message: str

class ObjectiveFunction:
    def __init__(self, function_type: str, parameters: Dict[str, Any]):
        self.function_type = function_type
        self.parameters = parameters
        self.evaluation_count = 0
        self.gradient_evaluations = 0
        
    def evaluate(self, x: List[float]) -> float:
        """Evaluate objective function at point x"""
        self.evaluation_count += 1
        
        if self.function_type == "sphere":
            return sum(xi ** 2 for xi in x)
        elif self.function_type == "rosenbrock":
            return self._rosenbrock(x)
        elif self.function_type == "rastrigin":
            return self._rastrigin(x)
        elif self.function_type == "ackley":
            return self._ackley(x)
        elif self.function_type == "griewank":
            return self._griewank(x)
        elif self.function_type == "schwefel":
            return self._schwefel(x)
        elif self.function_type == "custom":
            return self._custom_function(x)
        else:
            raise ValueError(f"Unknown function type: {self.function_type}")
    
    def gradient(self, x: List[float], h: float = 1e-8) -> List[float]:
        """Compute numerical gradient"""
        self.gradient_evaluations += 1
        grad = []
        
        for i in range(len(x)):
            x_plus = x.copy()
            x_minus = x.copy()
            x_plus[i] += h
            x_minus[i] -= h
            
            grad_i = (self.evaluate(x_plus) - self.evaluate(x_minus)) / (2 * h)
            grad.append(grad_i)
        
        return grad
    
    def _rosenbrock(self, x: List[float]) -> float:
        """Rosenbrock function - classic optimization benchmark"""
        if len(x) < 2:
            return float('inf')
        
        total = 0.0
        for i in range(len(x) - 1):
            total += 100 * (x[i+1] - x[i]**2)**2 + (1 - x[i])**2
        return total
    
    def _rastrigin(self, x: List[float]) -> float:
        """Rastrigin function - highly multimodal"""
        n = len(x)
        A = self.parameters.get('A', 10)
        return A * n + sum(xi**2 - A * math.cos(2 * math.pi * xi) for xi in x)
    
    def _ackley(self, x: List[float]) -> float:
        """Ackley function - multimodal with global optimum"""
        n = len(x)
        a = self.parameters.get('a', 20)
        b = self.parameters.get('b', 0.2)
        c = self.parameters.get('c', 2 * math.pi)
        
        sum1 = sum(xi**2 for xi in x)
        sum2 = sum(math.cos(c * xi) for xi in x)
        
        return (-a * math.exp(-b * math.sqrt(sum1 / n)) - 
                math.exp(sum2 / n) + a + math.e)
    
    def _griewank(self, x: List[float]) -> float:
        """Griewank function - many local minima"""
        sum_term = sum(xi**2 for xi in x) / 4000
        prod_term = 1.0
        for i, xi in enumerate(x):
            prod_term *= math.cos(xi / math.sqrt(i + 1))
        
        return sum_term - prod_term + 1
    
    def _schwefel(self, x: List[float]) -> float:
        """Schwefel function - deceptive global optimum"""
        return 418.9829 * len(x) - sum(xi * math.sin(math.sqrt(abs(xi))) for xi in x)
    
    def _custom_function(self, x: List[float]) -> float:
        """Custom complex function with multiple features"""
        if len(x) == 0:
            return float('inf')
        
        # Combine multiple characteristics
        quadratic = sum(xi**2 for xi in x)
        sinusoidal = sum(math.sin(10 * xi) for xi in x)
        exponential = math.exp(-sum(abs(xi) for xi in x) / len(x))
        logarithmic = sum(math.log(1 + abs(xi)) for xi in x)
        
        # Interaction terms
        interaction = 0
        for i in range(len(x) - 1):
            interaction += x[i] * x[i + 1]
        
        return (quadratic + 0.1 * sinusoidal + 
                10 * exponential + logarithmic + 0.5 * interaction)

class ConstraintHandler:
    def __init__(self, constraints: List[Dict[str, Any]]):
        self.constraints = constraints
        self.violation_count = 0
        
    def evaluate_constraints(self, x: List[float]) -> Tuple[bool, List[float]]:
        """Evaluate all constraints and return feasibility and violations"""
        violations = []
        feasible = True
        
        for constraint in self.constraints:
            violation = self._evaluate_single_constraint(x, constraint)
            violations.append(violation)
            
            if violation > 1e-6:  # Tolerance for constraint satisfaction
                feasible = False
                self.violation_count += 1
        
        return feasible, violations
    
    def _evaluate_single_constraint(self, x: List[float], constraint: Dict[str, Any]) -> float:
        """Evaluate single constraint and return violation amount"""
        constraint_type = constraint['type']
        
        if constraint_type == 'bounds':
            lower = constraint.get('lower', -float('inf'))
            upper = constraint.get('upper', float('inf'))
            variable_index = constraint.get('variable_index', 0)
            
            if variable_index >= len(x):
                return float('inf')
            
            val = x[variable_index]
            if val < lower:
                return lower - val
            elif val > upper:
                return val - upper
            else:
                return 0.0
        
        elif constraint_type == 'linear_inequality':
            # ax + b <= 0
            coefficients = constraint['coefficients']
            constant = constraint.get('constant', 0.0)
            
            if len(coefficients) != len(x):
                return float('inf')
            
            value = sum(coefficients[i] * x[i] for i in range(len(x))) + constant
            return max(0, value)
        
        elif constraint_type == 'linear_equality':
            # ax + b = 0
            coefficients = constraint['coefficients']
            constant = constraint.get('constant', 0.0)
            
            if len(coefficients) != len(x):
                return float('inf')
            
            value = sum(coefficients[i] * x[i] for i in range(len(x))) + constant
            return abs(value)
        
        elif constraint_type == 'nonlinear':
            # Custom nonlinear constraint
            expression = constraint['expression']
            return self._evaluate_nonlinear_constraint(x, expression)
        
        else:
            return 0.0
    
    def _evaluate_nonlinear_constraint(self, x: List[float], expression: str) -> float:
        """Evaluate nonlinear constraint expression"""
        # Simplified nonlinear constraint evaluation
        if expression == 'circle':
            # x^2 + y^2 <= 1
            if len(x) >= 2:
                value = x[0]**2 + x[1]**2 - 1
                return max(0, value)
        elif expression == 'sphere':
            # sum(x_i^2) <= 1
            value = sum(xi**2 for xi in x) - 1
            return max(0, value)
        
        return 0.0

class OptimizationSolver:
    def __init__(self, objective: ObjectiveFunction, constraints: List[Dict[str, Any]] = None):
        self.objective = objective
        self.constraint_handler = ConstraintHandler(constraints or [])
        self.best_solution = None
        self.best_value = float('inf')
        self.iteration_history = []
        
    def solve(self, initial_guess: List[float], method: OptimizationMethod, 
             options: Dict[str, Any]) -> OptimizationResult:
        """Solve optimization problem using specified method"""
        
        start_time = time.time()
        
        # Validate inputs
        if not initial_guess:
            return self._create_error_result("Empty initial guess", start_time)
        
        dimensions = len(initial_guess)
        if dimensions > 100:
            return self._create_error_result("Too many dimensions", start_time)
        
        # Extract common options
        max_iterations = options.get('max_iterations', 1000)
        tolerance = options.get('tolerance', 1e-6)
        population_size = options.get('population_size', 50)
        
        # Initialize
        self.best_solution = initial_guess.copy()
        self.best_value = float('inf')
        self.iteration_history = []
        
        try:
            if method == OptimizationMethod.GENETIC_ALGORITHM:
                result = self._genetic_algorithm(initial_guess, max_iterations, population_size, options)
            elif method == OptimizationMethod.PARTICLE_SWARM:
                result = self._particle_swarm_optimization(initial_guess, max_iterations, population_size, options)
            elif method == OptimizationMethod.SIMULATED_ANNEALING:
                result = self._simulated_annealing(initial_guess, max_iterations, options)
            elif method == OptimizationMethod.DIFFERENTIAL_EVOLUTION:
                result = self._differential_evolution(initial_guess, max_iterations, population_size, options)
            elif method == OptimizationMethod.GRADIENT_DESCENT:
                result = self._gradient_descent(initial_guess, max_iterations, options)
            elif method == OptimizationMethod.NELDER_MEAD:
                result = self._nelder_mead(initial_guess, max_iterations, options)
            else:
                return self._create_error_result("Unknown optimization method", start_time)
            
            execution_time = time.time() - start_time
            result.execution_time = execution_time
            
            return result
            
        except Exception as e:
            return self._create_error_result(f"Optimization failed: {str(e)}", start_time)
    
    def _genetic_algorithm(self, initial_guess: List[float], max_iterations: int, 
                          population_size: int, options: Dict[str, Any]) -> OptimizationResult:
        """Genetic Algorithm implementation"""
        
        dimensions = len(initial_guess)
        mutation_rate = options.get('mutation_rate', 0.1)
        crossover_rate = options.get('crossover_rate', 0.8)
        selection_pressure = options.get('selection_pressure', 2.0)
        
        # Initialize population
        population = []
        for _ in range(population_size):
            individual = [initial_guess[i] + random.gauss(0, 1) for i in range(dimensions)]
            population.append(individual)
        
        for iteration in range(max_iterations):
            # Evaluate population
            fitness_values = []
            for individual in population:
                feasible, violations = self.constraint_handler.evaluate_constraints(individual)
                
                if feasible:
                    fitness = self.objective.evaluate(individual)
                else:
                    # Penalty method for constraint violations
                    fitness = self.objective.evaluate(individual) + 1000 * sum(violations)
                
                fitness_values.append(fitness)
                
                if fitness < self.best_value:
                    self.best_value = fitness
                    self.best_solution = individual.copy()
            
            self.iteration_history.append(min(fitness_values))
            
            # Check convergence
            if len(self.iteration_history) > 10:
                recent_improvement = self.iteration_history[-10] - self.iteration_history[-1]
                if recent_improvement < options.get('tolerance', 1e-6):
                    break
            
            # Selection
            selected_parents = self._tournament_selection(population, fitness_values, 
                                                        population_size, selection_pressure)
            
            # Crossover and mutation
            new_population = []
            for i in range(0, population_size, 2):
                parent1 = selected_parents[i % len(selected_parents)]
                parent2 = selected_parents[(i + 1) % len(selected_parents)]
                
                if random.random() < crossover_rate:
                    child1, child2 = self._crossover(parent1, parent2)
                else:
                    child1, child2 = parent1.copy(), parent2.copy()
                
                if random.random() < mutation_rate:
                    child1 = self._mutate(child1, options)
                if random.random() < mutation_rate:
                    child2 = self._mutate(child2, options)
                
                new_population.extend([child1, child2])
            
            population = new_population[:population_size]
        
        # Final constraint check
        feasible, violations = self.constraint_handler.evaluate_constraints(self.best_solution)
        
        return OptimizationResult(
            success=feasible,
            solution=self.best_solution,
            objective_value=self.best_value,
            iterations=len(self.iteration_history),
            function_evaluations=self.objective.evaluation_count,
            convergence_history=self.iteration_history,
            constraint_violations=violations,
            execution_time=0.0,  # Will be set by caller
            message="GA completed" if feasible else "GA completed with constraint violations"
        )
    
    def _particle_swarm_optimization(self, initial_guess: List[float], max_iterations: int,
                                   swarm_size: int, options: Dict[str, Any]) -> OptimizationResult:
        """Particle Swarm Optimization implementation"""
        
        dimensions = len(initial_guess)
        w = options.get('inertia_weight', 0.7)
        c1 = options.get('cognitive_weight', 1.5)
        c2 = options.get('social_weight', 1.5)
        
        # Initialize swarm
        particles = []
        velocities = []
        personal_best = []
        personal_best_values = []
        
        for _ in range(swarm_size):
            particle = [initial_guess[i] + random.gauss(0, 1) for i in range(dimensions)]
            velocity = [random.gauss(0, 0.1) for _ in range(dimensions)]
            
            particles.append(particle)
            velocities.append(velocity)
            personal_best.append(particle.copy())
            
            feasible, _ = self.constraint_handler.evaluate_constraints(particle)
            if feasible:
                value = self.objective.evaluate(particle)
            else:
                value = float('inf')
            
            personal_best_values.append(value)
            
            if value < self.best_value:
                self.best_value = value
                self.best_solution = particle.copy()
        
        # Main PSO loop
        for iteration in range(max_iterations):
            for i in range(swarm_size):
                # Update velocity
                for d in range(dimensions):
                    r1, r2 = random.random(), random.random()
                    
                    velocities[i][d] = (w * velocities[i][d] +
                                      c1 * r1 * (personal_best[i][d] - particles[i][d]) +
                                      c2 * r2 * (self.best_solution[d] - particles[i][d]))
                
                # Update position
                for d in range(dimensions):
                    particles[i][d] += velocities[i][d]
                
                # Evaluate new position
                feasible, _ = self.constraint_handler.evaluate_constraints(particles[i])
                if feasible:
                    current_value = self.objective.evaluate(particles[i])
                else:
                    current_value = float('inf')
                
                # Update personal best
                if current_value < personal_best_values[i]:
                    personal_best_values[i] = current_value
                    personal_best[i] = particles[i].copy()
                
                # Update global best
                if current_value < self.best_value:
                    self.best_value = current_value
                    self.best_solution = particles[i].copy()
            
            self.iteration_history.append(self.best_value)
            
            # Dynamic inertia weight
            w = options.get('inertia_weight', 0.9) * (0.4 / 0.9) ** (iteration / max_iterations)
        
        feasible, violations = self.constraint_handler.evaluate_constraints(self.best_solution)
        
        return OptimizationResult(
            success=feasible and self.best_value != float('inf'),
            solution=self.best_solution,
            objective_value=self.best_value,
            iterations=len(self.iteration_history),
            function_evaluations=self.objective.evaluation_count,
            convergence_history=self.iteration_history,
            constraint_violations=violations,
            execution_time=0.0,
            message="PSO completed" if feasible else "PSO completed with constraint violations"
        )
    
    def _simulated_annealing(self, initial_guess: List[float], max_iterations: int,
                           options: Dict[str, Any]) -> OptimizationResult:
        """Simulated Annealing implementation"""
        
        current_solution = initial_guess.copy()
        current_value = self.objective.evaluate(current_solution)
        
        initial_temperature = options.get('initial_temperature', 100.0)
        cooling_rate = options.get('cooling_rate', 0.95)
        step_size = options.get('step_size', 1.0)
        
        temperature = initial_temperature
        
        for iteration in range(max_iterations):
            # Generate neighbor solution
            neighbor = current_solution.copy()
            for i in range(len(neighbor)):
                neighbor[i] += random.gauss(0, step_size)
            
            # Evaluate neighbor
            feasible, _ = self.constraint_handler.evaluate_constraints(neighbor)
            if feasible:
                neighbor_value = self.objective.evaluate(neighbor)
            else:
                neighbor_value = float('inf')
            
            # Accept or reject neighbor
            if neighbor_value < current_value:
                # Always accept better solutions
                current_solution = neighbor
                current_value = neighbor_value
                
                if current_value < self.best_value:
                    self.best_value = current_value
                    self.best_solution = current_solution.copy()
            else:
                # Accept worse solutions with probability
                if temperature > 0:
                    probability = math.exp(-(neighbor_value - current_value) / temperature)
                    if random.random() < probability:
                        current_solution = neighbor
                        current_value = neighbor_value
            
            self.iteration_history.append(self.best_value)
            
            # Cool down
            temperature *= cooling_rate
            
            # Adaptive step size
            if iteration % 100 == 0:
                step_size *= 0.95
        
        feasible, violations = self.constraint_handler.evaluate_constraints(self.best_solution)
        
        return OptimizationResult(
            success=feasible and self.best_value != float('inf'),
            solution=self.best_solution,
            objective_value=self.best_value,
            iterations=len(self.iteration_history),
            function_evaluations=self.objective.evaluation_count,
            convergence_history=self.iteration_history,
            constraint_violations=violations,
            execution_time=0.0,
            message="SA completed" if feasible else "SA completed with constraint violations"
        )
    
    # Helper methods for genetic algorithm
    def _tournament_selection(self, population: List[List[float]], fitness_values: List[float],
                            num_parents: int, pressure: float) -> List[List[float]]:
        """Tournament selection"""
        selected = []
        
        for _ in range(num_parents):
            tournament_size = max(2, int(len(population) * 0.1))
            tournament_indices = random.sample(range(len(population)), tournament_size)
            
            best_idx = min(tournament_indices, key=lambda i: fitness_values[i])
            selected.append(population[best_idx].copy())
        
        return selected
    
    def _crossover(self, parent1: List[float], parent2: List[float]) -> Tuple[List[float], List[float]]:
        """Simulated binary crossover"""
        eta = 20.0  # Distribution index
        
        child1 = parent1.copy()
        child2 = parent2.copy()
        
        for i in range(len(parent1)):
            if random.random() < 0.5:
                u = random.random()
                if u <= 0.5:
                    beta = (2 * u) ** (1 / (eta + 1))
                else:
                    beta = (1 / (2 * (1 - u))) ** (1 / (eta + 1))
                
                child1[i] = 0.5 * ((1 + beta) * parent1[i] + (1 - beta) * parent2[i])
                child2[i] = 0.5 * ((1 - beta) * parent1[i] + (1 + beta) * parent2[i])
        
        return child1, child2
    
    def _mutate(self, individual: List[float], options: Dict[str, Any]) -> List[float]:
        """Polynomial mutation"""
        eta = 20.0  # Distribution index
        mutation_probability = 1.0 / len(individual)
        
        mutated = individual.copy()
        
        for i in range(len(mutated)):
            if random.random() < mutation_probability:
                u = random.random()
                if u < 0.5:
                    delta = (2 * u) ** (1 / (eta + 1)) - 1
                else:
                    delta = 1 - (2 * (1 - u)) ** (1 / (eta + 1))
                
                mutated[i] += delta * abs(mutated[i]) if mutated[i] != 0 else delta
        
        return mutated
    
    def _create_error_result(self, message: str, start_time: float) -> OptimizationResult:
        """Create error result"""
        return OptimizationResult(
            success=False,
            solution=[],
            objective_value=float('inf'),
            iterations=0,
            function_evaluations=0,
            convergence_history=[],
            constraint_violations=[],
            execution_time=time.time() - start_time,
            message=message
        )
    
    # Additional optimization methods (simplified implementations)
    def _differential_evolution(self, initial_guess, max_iterations, population_size, options):
        # Simplified DE implementation
        return self._genetic_algorithm(initial_guess, max_iterations, population_size, options)
    
    def _gradient_descent(self, initial_guess, max_iterations, options):
        # Simplified gradient descent
        current = initial_guess.copy()
        learning_rate = options.get('learning_rate', 0.01)
        
        for iteration in range(max_iterations):
            grad = self.objective.gradient(current)
            for i in range(len(current)):
                current[i] -= learning_rate * grad[i]
            
            value = self.objective.evaluate(current)
            self.iteration_history.append(value)
            
            if value < self.best_value:
                self.best_value = value
                self.best_solution = current.copy()
        
        feasible, violations = self.constraint_handler.evaluate_constraints(self.best_solution)
        
        return OptimizationResult(
            success=feasible,
            solution=self.best_solution,
            objective_value=self.best_value,
            iterations=len(self.iteration_history),
            function_evaluations=self.objective.evaluation_count,
            convergence_history=self.iteration_history,
            constraint_violations=violations,
            execution_time=0.0,
            message="GD completed"
        )
    
    def _nelder_mead(self, initial_guess, max_iterations, options):
        # Simplified Nelder-Mead implementation
        return self._gradient_descent(initial_guess, max_iterations, options)

def target_function(problem_definition: Dict[str, Any], solver_config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Target function for test case generation with high complexity
    """
    # Validate problem definition
    if not isinstance(problem_definition, dict):
        return {'error': 'invalid_problem_definition'}
    
    if not isinstance(solver_config, dict):
        return {'error': 'invalid_solver_config'}
    
    # Extract problem components
    objective_config = problem_definition.get('objective', {})
    constraints = problem_definition.get('constraints', [])
    initial_guess = problem_definition.get('initial_guess', [])
    
    # Validate objective function configuration
    if not isinstance(objective_config, dict) or 'type' not in objective_config:
        return {'error': 'invalid_objective_config'}
    
    function_type = objective_config['type']
    if function_type not in ['sphere', 'rosenbrock', 'rastrigin', 'ackley', 'griewank', 'schwefel', 'custom']:
        return {'error': 'unknown_objective_function'}
    
    # Validate initial guess
    if not initial_guess or not isinstance(initial_guess, list):
        return {'error': 'invalid_initial_guess'}
    
    if len(initial_guess) > 50:  # Reasonable dimension limit
        return {'error': 'too_many_dimensions'}
    
    for val in initial_guess:
        if not isinstance(val, (int, float)) or math.isnan(val) or math.isinf(val):
            return {'error': 'invalid_initial_guess_values'}
    
    # Extract solver configuration
    method_name = solver_config.get('method', 'genetic_algorithm')
    try:
        method = OptimizationMethod(method_name)
    except ValueError:
        return {'error': 'unknown_optimization_method'}
    
    max_iterations = solver_config.get('max_iterations', 100)
    if max_iterations <= 0 or max_iterations > 10000:
        return {'error': 'invalid_max_iterations'}
    
    tolerance = solver_config.get('tolerance', 1e-6)
    if tolerance <= 0 or tolerance > 1:
        return {'error': 'invalid_tolerance'}
    
    # Create objective function
    try:
        objective = ObjectiveFunction(function_type, objective_config.get('parameters', {}))
    except Exception as e:
        return {'error': f'objective_creation_failed: {str(e)}'}
    
    # Validate constraints
    if not isinstance(constraints, list):
        return {'error': 'invalid_constraints'}
    
    for i, constraint in enumerate(constraints):
        if not isinstance(constraint, dict) or 'type' not in constraint:
            return {'error': f'invalid_constraint_at_{i}'}
    
    # Create solver and solve
    try:
        solver = OptimizationSolver(objective, constraints)
        result = solver.solve(initial_guess, method, solver_config)
        
        # Analyze result complexity
        if result.success:
            if result.objective_value < tolerance:
                status = 'optimal_solution_found'
            elif result.iterations < max_iterations * 0.1:
                status = 'fast_convergence'
            elif len(result.convergence_history) > 0:
                final_improvement = result.convergence_history[0] - result.convergence_history[-1]
                if final_improvement < tolerance:
                    status = 'slow_convergence'
                else:
                    status = 'good_convergence'
            else:
                status = 'solution_found'
        else:
            if result.iterations >= max_iterations:
                status = 'max_iterations_reached'
            elif result.constraint_violations and any(v > tolerance for v in result.constraint_violations):
                status = 'constraint_violations'
            elif result.objective_value == float('inf'):
                status = 'infeasible_problem'
            else:
                status = 'optimization_failed'
        
        return {
            'status': status,
            'optimization_result': {
                'success': result.success,
                'solution': result.solution,
                'objective_value': result.objective_value,
                'iterations': result.iterations,
                'function_evaluations': result.function_evaluations,
                'execution_time': result.execution_time,
                'message': result.message
            },
            'problem_characteristics': {
                'dimensions': len(initial_guess),
                'objective_type': function_type,
                'constraint_count': len(constraints),
                'method_used': method_name
            }
        }
        
    except Exception as e:
        return {'error': f'solver_execution_failed: {str(e)}'}