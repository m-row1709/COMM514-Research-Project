"""
This file contains the necessary functions for implementing the custom NSGA2 algorithm developed for this research project.
These functions include:
    - A custom weighted accuracy measure.
    - A method for fixing the features selected by the algorithm in the initial population.
    - A custom method for defining how Variable Interaction Graph impacts the algorithm's mutation operator.
    - The complete custom NSGA2 method.
    - A function for externally validating the performance of a solution.
    - Wrappers for performing a single NSGA2 optimisation run as well as repeated optimisation runs.
    - Wrappers for performing a single Lasso-MO optimisation run as well as repeated optimisation runs.
"""

# Key modules and libraries
import os
import numpy as np
from joblib import Parallel, delayed

# Pymoo packages
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.optimize import minimize
from pymoo.operators.crossover.ux import UniformCrossover
from pymoo.core.sampling import Sampling
from pymoo.core.mutation import Mutation
from pymoo.core.problem import ElementwiseProblem, StarmapParallelization
import multiprocessing as mp

# Classification modelling packages
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import accuracy_score, f1_score



# -------------------------------------
# Custom Weighted Accuracy Evaluation
# -------------------------------------

def weighted_accuracy_score(y_true, y_pred, label_weights):
    """
    Custom evaluation function. 
    Returns the per-label accuracy according to a pre-defined set of label weights.

    Args:
        - y_true (ndarray): A 1D NumPy array of length (n_records). Contains ground truth labels.
        - y_pred (ndarray): A 1D NumPy array of length (n_records). Contains predicted labels.
        - label_weights (dict): Contains the relative weights for each of the labels "1", "2" and "3".

    Returns:
        - weighted_accuracy (float): The evaluated weighted accuracy score.
    """
    # Get list of labels
    unique_labels = np.unique(y_true)

    weighted_accuracy_sum = 0.0

    for lbl in unique_labels:
        label_mask = (y_true == lbl)

        # Calculate accuracy for each class
        label_accuracy = accuracy_score(y_true[label_mask], y_pred[label_mask])

        # Multiply accuracy by the assigned weight
        weighted_accuracy_sum += label_accuracy * label_weights.get(lbl, 1)

    # Normalise in case of label weights that do not sum to 1
    weighted_accuracy = weighted_accuracy_sum / sum(label_weights.values())

    return weighted_accuracy



# --------------------------------
# Implement Customisation Methods
# --------------------------------

class FixedFeatureSampling(Sampling):
    """
    A structure for introducing control over the selection of features for the initial population.

    Attributes:
        - data (ndarray): A 2D NumPy array of shape (n_records, n_features) - contains counts data.
        - features (ndarray): A 1D NumPy array of shape (n_features) - contains a list of feature names.
        - num_features (int): The maximum number of features a solution in the initial population may have.
        - required_features (ndarray): A 1D NumPy array - contains a list of all features required to be in every solution.

    Method:
        - _do(): defines initial population solutions according to the provided conditions.
    """

    def __init__(self, data, features, num_features = 10, required_features = np.array([])):
        self.data = data
        self.features = features
        self.num_features = num_features
        self.vtype = bool
        self.repair = None
        self.required_features = required_features


    def _do(self, problem, n_samples, **kwargs):
        # Get indices of all required genes
        required_indices = []
        for i in range(len(self.required_features)):
            required_indices.append(self.features.index(self.required_features[i]))

        # Start with all zeros
        pop = np.zeros((n_samples, problem.n_var)) 

        # Create initial list to select from for initial population
        select_from_indices = list(range(problem.n_var))
        for index in required_indices:
            # Ensure that required indices cannot be selected at random too
            select_from_indices.remove(index)

        for i in range(n_samples):
            # Pick a random solution length between at least 3 and at most the num_features argument
            random_solution_length = np.random.randint(np.max([3, len(required_indices) + 1]), self.num_features + 1)
            # Ensures always at least 1 feature at random included in solution
            selected_indices = np.random.choice(select_from_indices, random_solution_length - len(required_indices), replace = False)

            # Add the required indices
            selected_indices = np.append(selected_indices, required_indices)

            pop[i, selected_indices] = 1  # Activate exactly 10 features

        return pop
    


class VIGGuidedMutation(Mutation):
    """
    A custom mutation operator, adapted to incorporate knowledge of a Variable Interaction Graph.

    Attributes:
        - data (ndarray): A 2D NumPy array of shape (n_records, n_features) - contains counts data.
        - features (ndarray): A 1D NumPy array of shape (n_features) - contains a list of feature names.
        - subfunctions (dict): Contains a list of related features for each feature in the dataset.
        - required_features (ndarray): A 1D NumPy array - contains a list of all features required to be in every solution.
        - prob (float): The probability with which each solution in the population will be mutated.
        - prob_var (float): The probability with which each feature of a solution will be mutated.

    Methods:
        - _do(): defines the custom mutation operator implementation.
        - get_related_variables(): Extracts the list of related features from subfunctions for a given feature.
    """

    def __init__(self, data, features, subfunctions, required_features, prob = 1, prob_var = 0.01):
        super().__init__()
        self.data = data
        self.features = features
        self.subfunctions = subfunctions
        self.required_features = required_features
        self.prob = prob # Individual mutation probability
        self.prob_var_add = prob_var / 2 # Variable mutation probability for feature addition
        self.prob_var_drop = prob_var # Variable mutation probability for feature removal


    def _do(self, problem, X, **kwargs):
        # Get the column indices of our list of required features
        required_indices = []
        for i in range(len(self.required_features)):
            required_indices.append(self.features.index(self.required_features[i]))
        
        # Make a true copy of the population
        X_new = X.copy()

        # Iterate through each solution
        for i in range(X.shape[0]):
            # Only mutate each solution with probability equal to prob
            if np.random.rand() < self.prob:
                # Iterate through each feature
                for var in range(X.shape[1]):
                    # Ensure that required features are always included
                    if var in required_indices:
                        X_new[i, var] = 1

                    # Mutate to remove feature with probability equal to prob_var_drop
                    elif X[i, var] == 1 and np.random.rand() < self.prob_var_drop:
                        # Perform mutation
                        X_new[i, var] = 1 - X[i, var]

                        # Next mutate interactive features
                        related = self.get_related_variables(var)

                        for r in related:
                            # Ensure that required features are not disrupted
                            if r in required_indices:
                                X_new[i, r] = 1

                            # Mutate related features with probability proportional to the degree of connectivity
                            elif np.random.rand() < 1 / (1 + len(related)):
                                X_new[i, r] = 1 - X[i, r]
                    
                    # Mutate to add feature with probability equal to prob_var_add
                    elif X[i, var] == 0 and np.random.rand() < self.prob_var_add:
                        # Perform mutation
                        X_new[i, var] = 1 - X[i, var]

                        # Next mutate interactive features
                        related = self.get_related_variables(var)

                        for r in related:
                            # Ensure that required features are not disrupted
                            if r in required_indices:
                                X_new[i, r] = 1

                            # Mutate related features with probability proportional to the degree of connectivity
                            elif np.random.rand() < 1 / (1 + len(related)):
                                X_new[i, r] = 1 - X[i, r]

            else:
                continue

        return X_new
    

    def get_related_variables(self, var):
        """
        Returns a list of all subfunctions that the provided feature appears in.

        Args:
            - var (int): The index of the provided feature.

        Returns:
            - related (list): A list of each subfunction the feature appears within.
        """
        related = set()

        # Check for the presence of var within each subfunction set
        for group in self.subfunctions.values():
            if var in group:
                related.update(group)
        
        # Remove itself from this list
        related.discard(var)

        return list(related)



# ----------------------------
# Implement ClassifyOptimise
# ----------------------------

class ClassifyOptimise(ElementwiseProblem):
    """
    A custom NSGA2 algorithm implementation, adapted to optimise the classification 

    Attributes:
        - data (ndarray): A 2D NumPy array of shape (n_records, n_features) - contains counts data.
        - labels (ndarray): A 1D NumPy array of shape (n_records) - contains ground truth labels data.
        - features (ndarray): A 1D NumPy array of shape (n_features) - contains a list of feature names.
        - label_weights (dict): Contains the relative weights for each of the labels "1", "2" and "3".
        - log_file (str): Defines the filename for the logged outputs.
        - shared_results (proxy object): Proxy for storing a shared list of complete results across generations.

    Methods:
        - _evaluate(): defines the custom solution evaluation function, performed on each solution of a generation in parallel.
        - evaluate_performance(): trains and evaluates a solution, returning its performance metrics.
    """

    def __init__(self, data, labels, features, label_weights, log_file, shared_results = None, **kwargs):
        super().__init__(n_var=data.shape[1],   # Number of features
                         n_obj=2,               # Two objectives
                         n_constr=0,            # No constraints
                         xl=0,                  # Lower bound (feature excluded)
                         xu=1,                  # Upper bound (feature included)
                         **kwargs)
        self.data = data
        self.labels = labels
        self.features = features
        self.label_weights = label_weights
        self.log_file = log_file
        self.generation_number = 0
        self.complete_results = shared_results if shared_results is not None else []
    

    def _evaluate(self, x, out, *args, **kwargs):
        """
        Evaluation function called for each solution of a population.
        """
        # Objective 1: Minimize the number of features
        f1 = np.sum(x)

        # Objective 2: Maximize classification performance metric
        mean_weighted_accuracy, fold_accuracies = self.evaluate_performance(x)
        f2 = -mean_weighted_accuracy

        # Return objective values
        out["F"] = [f1, f2]

        # Update results for iterated generation, and store in detailed archive
        solution_results = {
            "solution": x,
            "number of features": f1,
            "mean weighted accuracy": mean_weighted_accuracy, 
            "fold accuracies": fold_accuracies
        }
        self.complete_results.append(solution_results)

        # Update log file
        with open(f"{self.log_file}", "a") as f:
            f.write(f"Evaluating on PID {os.getpid()}, Number of features: {f1}, Mean weighted accuracy: {mean_weighted_accuracy:.4f}\n")
    

    def evaluate_performance(self, x):
        """
        Performs support vector machine classification and returns the evaluation of the predicted labels.

        Args:
            - x (ndarray): A 1D NumPy array of length (n_features) - solution contains a binary string denoting feature inclusion.
        
        Returns:
            - mean_weighted_accuracy (float): The averaged evaluation metric across the folds.
            - fold_accuracies (list): A list of the individual accuracies for each of the k folds.
        """
        selected_features = x.astype(bool)
        X = self.data[:, selected_features]

        # Punish the algorithm for selecting 0 features
        if X.shape[1] == 0:
            return 0.0
        
        # Perform 5-fold data stratification
        skf = StratifiedKFold(n_splits = 5)

        fold_accuracies = []

        for k, (train_idx, test_idx) in enumerate(skf.split(X, self.labels)):
            # Split data into training and testing sets
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = self.labels[train_idx], self.labels[test_idx]

            # Train classifier
            clf = SVC(kernel = "rbf", class_weight = self.label_weights)
            clf.fit(X_train, y_train)

            # Predict on test set
            y_pred = clf.predict(X_test)
            weighted_accuracy = weighted_accuracy_score(y_test, y_pred, label_weights = self.label_weights)

            fold_accuracies.append(weighted_accuracy)

        # Record mean accuracy
        mean_weighted_accuracy = np.mean(fold_accuracies)

        return mean_weighted_accuracy, fold_accuracies
    

# ---------------------
# External Validation
# ---------------------

def perform_external_validation(counts_data, labels_data, label_weights, final_results):
    """
    Externally validates the performance of a given solution, according to the associated data and label weights.

    Args:
        - counts_data (ndarray): A 2D NumPy array of shape (n_records, n_features) - contains counts data.
        - labels_data (ndarray): A 1D NumPy array of shape (n_records) - contains ground truth labels data.
        - label_weights (dict): Contains the relative weights for each of the labels "1", "2" and "3".
        - final_results: Pymoo algorithm output object.

    Returns:
        - external_validation_accuracies (list): A list of externally validated accuracies for each solution in the Pareto front.
    """

    external_validation_accuracies = []

    # Validate each solution in the final pareto front
    for i in range(len(final_results.X)):

        # Extract solution from ith result, and filter counts data for those features
        best_solution = final_results.X[i]
        X = counts_data[:, best_solution]
        
        # Perform 5-fold data stratification
        skf = StratifiedKFold(n_splits = 5)

        def process_fold(train_idx, test_idx, k):
            # Split data into training and testing sets
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = labels_data[train_idx], labels_data[test_idx]

            # Train classifier
            clf = SVC(kernel = "rbf", class_weight = label_weights)
            clf.fit(X_train, y_train)

            # Predict on test set
            y_pred = clf.predict(X_test)

            # Evaluate accuracy for each class
            accuracies = {}
            for label in np.unique(y_test):
                accuracies[label] = f1_score(y_test, y_pred, labels = [label], average = "macro")

            return accuracies

        # Calculates fold accuracies for each fold in parallel
        fold_accuracies = Parallel(n_jobs = -1)(delayed(process_fold)(train_idx, test_idx, k + 1) 
                                            for k, (train_idx, test_idx) in enumerate(skf.split(X, labels_data)))

        # Aggregate per-class accuracy across the folds
        class_labels = np.unique(labels_data)
        mean_class_accuracies = {label: np.mean([fold[label] for fold in fold_accuracies]) for label in class_labels}

        # Attach accuracies for solution i to complete external validation results
        external_validation_accuracies.append(mean_class_accuracies)
    
    return external_validation_accuracies



# ------------------------------------
# Optimisation Performance Functions
# ------------------------------------

def run_optimisation(counts_data, labels_data, feature_list, label_weights,
                     pop_size, n_generations, initial_pop_num_features, required_features, 
                     subfunctions, ind_mutation_prob, var_mutation_prob, n_processes, 
                     log_file, random_seed):
    """
    Performs a single run of the optimisation algorithm.

    Args:
        - counts_data (ndarray): A 2D NumPy array of shape (n_records, n_features) - contains counts data.
        - labels_data (ndarray): A 1D NumPy array of shape (n_records) - contains ground truth labels data.
        - feature_list (ndarray): A 1D NumPy array of shape (n_features) - contains a list of feature names. 
        - label_weights (dict): Contains the relative weights for each of the labels "1", "2" and "3".
        - pop_size (int): The population size of each generation of NSGA2.
        - n_generations (int): The number of generations NSGA2 should iterate for.
        - initial_pop_num_features (int): The maximum number of features a solution in the initial population may have.
        - required_features (ndarray): A 1D NumPy array - contains a list of all features required to be in every solution.
        - subfunctions (dict): Contains a list of related features for each feature in the dataset.
        - ind_mutation_prob (float): The probability with which each solution in the population will be mutated.
        - var_mutation_prob (float): The probability with which each feature of a solution will be mutated.
        - n_processes (int): The number of core processes the algorithm should utilise.
        - log_file (str): Defines the filename for the logged outputs.
        - random_seed (int): The pre-defined random seed that NSGA2 should operate upon.

    Returns:
        - results: Pymoo algorithm output object.
        - complete_results: Manager object, containing detailed list compendium of all individual solutions and results.
    """
    
    # Initialise multi-processing setup
    mp.set_start_method("spawn", force = True)

    # Initialise manager for storing complete optimisation results
    manager = mp.Manager()
    shared_results = manager.list()

    # Set up multiprocessing pool and runner
    pool = mp.Pool(n_processes)
    runner = StarmapParallelization(pool.starmap)

    # Initalise instance of optimiser
    problem_instance = ClassifyOptimise(data = counts_data, labels = labels_data, 
                                        features = feature_list, label_weights = label_weights, 
                                        shared_results = shared_results, log_file = log_file, 
                                        elementwise_runner = runner)

    # Initialise instance of NSGA2
    algorithm = NSGA2(pop_size = pop_size,
                      sampling = FixedFeatureSampling(data = counts_data,
                                                      features = feature_list,
                                                      num_features = initial_pop_num_features,
                                                      required_features = required_features),
                      crossover = UniformCrossover(),
                      mutation = VIGGuidedMutation(data = counts_data, features = feature_list, subfunctions = subfunctions, 
                                                   required_features = required_features, prob = ind_mutation_prob, prob_var = var_mutation_prob))

    # Perform minimisation
    results = minimize(problem = problem_instance,
                       algorithm = algorithm,
                       termination = ('n_gen', n_generations),
                       verbose = True,
                       seed = random_seed)

    pool.close()
    pool.join()

    return results, results.problem.complete_results



def classify_optimise_repeat(counts_data, labels_data, feature_list, label_weights,
                             pop_size, n_generations, initial_pop_num_features, required_features,
                             subfunctions, ind_mutation_prob, var_mutation_prob, n_processes, 
                             n_repeats, log_file, random_seed = 42):
    """
    Performs a set number of repeat runs of the optimisation algorithm.
    
    Args:
        - counts_data (ndarray): A 2D NumPy array of shape (n_records, n_features) - contains counts data.
        - labels_data (ndarray): A 1D NumPy array of shape (n_records) - contains ground truth labels data.
        - feature_list (ndarray): A 1D NumPy array of shape (n_features) - contains a list of feature names. 
        - label_weights (dict): Contains the relative weights for each of the labels "1", "2" and "3".
        - pop_size (int): The population size of each generation of NSGA2.
        - n_generations (int): The number of generations NSGA2 should iterate for.
        - initial_pop_num_features (int): The maximum number of features a solution in the initial population may have.
        - required_features (ndarray): A 1D NumPy array - contains a list of all features required to be in every solution.
        - subfunctions (dict): Contains a list of related features for each feature in the dataset.
        - ind_mutation_prob (float): The probability with which each solution in the population will be mutated.
        - var_mutation_prob (float): The probability with which each feature of a solution will be mutated.
        - n_processes (int): The number of core processes the algorithm should utilise.
        - n_repeats (int): The number of repeats to perform.
        - log_file (str): Defines the filename for the logged outputs.
        - random_seed (int): The pre-defined random seed that NSGA2 should operate upon.

    Returns:
        - all_repeat_results (dict): Contains the results, complete solution results, external validation results for each repeat performed.
    """
    
    # Define random seed
    np.random.seed(random_seed)

    # Initialise empty dictionary for all repeat results
    all_repeat_results = {}

    # Create an empty log file 
    with open(f"{log_file}", "w") as f:
        f.write("Beginning optimisation cycle...\n")  

    for r in range(n_repeats):
        # Initialise dictionary for results of generation r
        single_repeat_results = {}

        print(f"Starting Repeat {r+1} out of {n_repeats}...")

        # Update log file
        with open(f"{log_file}", "a") as f:
            f.write(f"Beginning repeat {r+1}...\n")  

        # Generate random seed for repeat optimisation run
        repeat_random_seed = np.random.randint(1, 100001)

        final_results, complete_results = run_optimisation(counts_data = counts_data,
                                                           labels_data = labels_data, 
                                                           feature_list = feature_list, 
                                                           label_weights = label_weights,
                                                           pop_size = pop_size,
                                                           n_generations = n_generations,
                                                           initial_pop_num_features = initial_pop_num_features,
                                                           required_features = required_features,
                                                           subfunctions = subfunctions, 
                                                           ind_mutation_prob = ind_mutation_prob,
                                                           var_mutation_prob = var_mutation_prob,
                                                           n_processes = n_processes,
                                                           log_file = log_file,
                                                           repeat = r+1,
                                                           random_seed = repeat_random_seed)
        
        print(f"Completed optimisation of Repeat {r+1} out of {n_repeats}.")

        print(f"Performing External Validation...")

        external_validation_results = perform_external_validation(counts_data = counts_data,
                                                                  labels_data = labels_data,
                                                                  label_weights = label_weights, 
                                                                  final_results = final_results)
        
        # Update dictionary to store results of this repeat
        single_repeat_results.update({
            "final results": final_results,
            "complete results": list(complete_results),
            "external validation results": external_validation_results
        })

        # Update complete results dictionary to store results of this repeat
        all_repeat_results.update({
            f"{r+1}": single_repeat_results
        })
        
    return all_repeat_results
        


# -----------------------
# Lasso-MO Optimisation
# -----------------------

def lasso_mo(X, y, label_weights, initial_C = 1.0, decay = 0.99, random_seed = 42):
    """
    Implements LASSO-MO to explore trade-offs between accuracy and sparsity.

    Args:
        - X (pd.DataFrame): A Pandas dataframe of shape (n_records, n_features) - contains counts data.
        - y (pd.Series): A Pandas series of shape (n_records) - contains ground truth labels data.
        - label_weights (dict): Contains the relative weights for each of the labels "1", "2" and "3".
        - initial_C (float): Initial inverse regularization strength.
        - decay (float): Factor with which C is reduced each generation.
        - random_seed (int): The pre-defined random seed that Lasso-MO should operate upon.

    Returns:
        - solutions (list): Contains a list of tuples (C, weighted accuracy, number of features, selected features).
    """
    np.random.seed(random_seed)

    C = initial_C
    solutions = []

    # Repeat until regularization removes all features
    while C > 1e-4:  
        # Generate random seed for each solver instance
        instance_random_seed = np.random.randint(1, 100001)

        # Initialise model
        model = LogisticRegression(penalty = "l1", solver = "saga", multi_class = "multinomial", 
                                   C = C, max_iter = 5000, random_state = instance_random_seed)
        
        # Split data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, stratify = y, random_state = instance_random_seed)

        # Train model
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        # Evaluate performance
        weighted_accuracy = weighted_accuracy_score(y_test, y_pred, label_weights)

        # Get the indices of the features selected by the model
        selected_features = np.where(np.any(model.coef_ != 0, axis = 0))[0]

        if len(selected_features) == 1:
            break

        # Store results
        solution_results = {
            "C": C,
            "weighted accuracy": weighted_accuracy, 
            "number of features": len(selected_features),
            "selected features": selected_features
        }
        
        solutions.append(solution_results)

        # Decay current value of C by decay parameter
        C *= decay

        print(solution_results)
        print(f"C = {C}")

    return solutions



def lasso_mo_repeat(X, y, label_weights, n_repeats, initial_C = 1.0, decay = 0.99, random_seed = 42):
    """
    Args:
        - X (pd.DataFrame): A Pandas dataframe of shape (n_records, n_features) - contains counts data.
        - y (pd.Series): A Pandas series of shape (n_records) - contains ground truth labels data.
        - label_weights (dict): Contains the relative weights for each of the labels "1", "2" and "3".
        - n_repeats (int): The number of repeats to perform.
        - initial_C (float): Initial inverse regularization strength.
        - decay (float): Factor with which C is reduced each generation.
        - random_seed (int): The pre-defined random seed that Lasso-MO should operate upon.

    Returns:
        - all_repeat_results (dict): Contains the results, complete solution results, external validation results for each repeat performed.
    """
    np.random.seed(random_seed)

    # Generate list of random seeds for each lasso repeat
    repeat_seeds = np.random.randint(1, 100001, size = n_repeats)

    # Initialise empty dictionary for all repeat results
    all_repeat_results = {}

    for r in range(n_repeats):
        print(f"Starting Repeat {r+1} out of {n_repeats}...")
        
        # Perform repeat optimisation run
        solutions = lasso_mo(X = X,
                             y = y, 
                             label_weights = label_weights,
                             initial_C = initial_C,
                             decay = decay,
                             random_seed = repeat_seeds[r])
        
        print(f"Completed optimisation of Repeat {r+1} out of {n_repeats}.")
        
        # Update complete results dictionary to store results of this repeat
        all_repeat_results.update({
            f"{r+1}": solutions
        })

    return all_repeat_results



if __name__ == "__main__":
    pass
