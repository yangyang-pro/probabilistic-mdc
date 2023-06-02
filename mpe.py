import numpy as np

from tqdm import tqdm
from collections import defaultdict

from pgmpy.models import BayesianNetwork
from pgmpy.factors import factor_product


# Compute the most probable explanation (MPE) task
def maximize(phi_o, variables, inplace=True):
    if isinstance(variables, str):
        raise TypeError("variables: Expected type list or array-like, got type str")

    phi = phi_o if inplace else phi_o.copy()

    for var in variables:
        if var not in phi.variables:
            raise ValueError(f"{var} not in scope.")

    # get the indices of the input variables
    var_indexes = [phi.variables.index(var) for var in variables]

    # get the indices of the rest variabels
    index_to_keep = sorted(set(range(len(phi_o.variables))) - set(var_indexes))
    # the new factor with the rest variables
    phi.variables = [phi.variables[index] for index in index_to_keep]
    # the new factor with the cardinality of the rest variables
    phi.cardinality = phi.cardinality[index_to_keep]
    # delete the eliminated variables
    phi.del_state_names(variables)

    var_assig = np.argmax(phi.values, axis=var_indexes[0])
    phi.values = np.max(phi.values, axis=tuple(var_indexes))

    if not inplace:
        return phi, var_assig


def compute_mpe(graph, variables, evidence, elimination_order, joint=True, show_progress=True):
    # get working factors
    factors = defaultdict(list)
    for node in graph.nodes():
        cpd = graph.get_cpds(node)
        cpd = cpd.to_factor()
        for var in cpd.scope():
            factors[var].append(cpd)

    working_factors = {
        node: {(factor, None) for factor in factors[node]}
        for node in factors
    }

    to_eliminate = (
            set(graph.nodes)
            - set(variables)
            - set(evidence.keys() if evidence else [])
    )

    # get elimination order
    # Step 1: If elimination_order is a list, verify it's correct and return.
    # Step 1.1: Check that not of the `variables` and `evidence` is in the elimination_order.
    if hasattr(elimination_order, "__iter__") and (not isinstance(elimination_order, str)):
        if any(var in elimination_order for var in set(variables).union(set(evidence.keys() if evidence else []))):
            raise ValueError(
                "Elimination order contains variables which are in"
                " variables or evidence args"
            )
        # Step 1.2: Check if elimination_order has variables which are not in the model.
        elif any(var not in graph.nodes() for var in elimination_order):
            elimination_order = list(filter(lambda t: t in graph.nodes(), elimination_order))

        # Step 1.3: Check if the elimination_order has all the variables that need to be eliminated.
        elif to_eliminate != set(elimination_order):
            raise ValueError(
                f"Elimination order doesn't contain all the variables "
                f"which need to be eliminated. The variables which need to "
                f"be eliminated are {to_eliminate}")

    # Step 2: If elimination order is None or a Markov model, return a random order.
    elif elimination_order is None:
        elimination_order = list(to_eliminate)
    else:
        elimination_order = None

    # max_marginal
    if not variables:
        variables = []

    common_vars = set(evidence if evidence is not None else []).intersection(
        set(variables if variables is not None else [])
    )
    if common_vars:
        raise ValueError(
            f"Can't have the same variables in both `variables` and `evidence`. Found in both: {common_vars}")

    # variable elimination
    # Step 1: Deal with the input arguments.
    if isinstance(variables, str):
        raise TypeError("variables must be a list of strings")
    if isinstance(evidence, str):
        raise TypeError("evidence must be a list of strings")

    # Dealing with the case when variables is not provided.
    if not variables:
        all_factors = []
        for factor_li in graph.get_factors():
            all_factors.extend(factor_li)
        if joint:
            pi = factor_product(*all_factors)
        else:
            pi = set(all_factors)

    # Step 2: Prepare data structures to run the algorithm.
    eliminated_variables = set()
    # Get working factors and elimination order
    # working_factors = working_factors
    # elimination_order = elimination_order

    assignments = {node: None for node in graph.nodes}
    eliminated_assignments = {node: (None, None) for node in elimination_order}

    # Step 3: Run variable elimination
    if show_progress:
        pbar = tqdm(elimination_order)
    else:
        pbar = elimination_order

    for var in pbar:
        if show_progress:
            pbar.set_description(f"Eliminating: {var}")
        # Removing all the factors containing the variables which are
        # eliminated (as all the factors should be considered only once)
        factors = [factor for factor, _ in working_factors[var] if
                   not set(factor.scope()).intersection(eliminated_variables)]
        phi = factor_product(*factors)
        phi, var_assignment = maximize(phi, [var], inplace=False)
        # phi = getattr(phi, operation)([var], inplace=False)
        del working_factors[var]
        for variable in phi.variables:
            working_factors[variable].add((phi, var))
        eliminated_variables.add(var)
        eliminated_assignments[var] = (var_assignment, phi.variables)

    # Step 4: Prepare variables to be returned.
    final_distribution = set()
    for node in working_factors:
        for factor, origin in working_factors[node]:
            if not set(factor.variables).intersection(eliminated_variables):
                final_distribution.add((factor, origin))
    final_distribution = [factor for factor, _ in final_distribution]

    if joint:
        if isinstance(graph, BayesianNetwork):
            final_distribution = factor_product(*final_distribution).normalize(inplace=False)
        else:
            final_distribution = factor_product(*final_distribution)
    else:
        query_var_factor = {}
        for query_var in variables:
            phi = factor_product(*final_distribution)
            query_var_factor[query_var] = phi.marginalize(list(set(variables) - set([query_var])),
                                                          inplace=False).normalize(inplace=False)
        final_distribution = query_var_factor

    max_assign = np.unravel_index(np.argmax(final_distribution.values, axis=None), final_distribution.values.shape)
    for (node, assign) in zip(final_distribution.variables, max_assign):
        assignments[node] = assign
    elimination_order.reverse()
    for node in elimination_order:
        ind = []
        for variable in eliminated_assignments[node][1]:
            ind.append(assignments[variable])
        assignments[node] = eliminated_assignments[node][0][tuple(ind)]
    # max_assign = np.argmax(final_distribution.values)
    max_prob = np.max(final_distribution.values)
    return max_prob, assignments
