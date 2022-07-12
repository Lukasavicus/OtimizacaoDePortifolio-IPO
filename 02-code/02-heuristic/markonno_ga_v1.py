import pygad, numpy, pandas as pd, time
from itertools import product

# === INPUT ====================================================================

# 0.0298, 56.3, 8.87, 46.40
# 0, 56, 8, 46
# 0 3 2 1

# === Helper ===================================================================
def agroup(list_, group_size):
    return [list_[i*group_size:(i+1)*group_size] for i in range(len(list_)//group_size)]

def norm01(metric, min_, max_):
    return (metric - min_) / (max_ - min_)
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

# === Markowitz & Konno and Yamazaki ===========================================
def markonno_risk(mtx, decision_variables, n=1, mu=2):
    """
        mtx is expected to be a pandas.DataFrame containing
        ROWS - date range
        COLS - assets
    """
    n_rows, n_cols = mtx.shape
    n_cols -= 1  # disconsider the date range label column
    means = list(mtx.mean().values)
    mtx_values = [list(l)[1:] for l in list(mtx.values)]
    linear_rtildes = [v - means[i] for r in mtx_values for i, v in enumerate(r)]
    rtildes = agroup(linear_rtildes, n_cols)

    rtilde_per_dv = [(decision_variables[i] * ri) for r in rtildes for i, ri in enumerate(r)]
    rtilde_per_dv_agrouped = agroup(rtilde_per_dv, n_cols)
    sum_rtilde_per_dv_agrouped = [sum(r) for r in rtilde_per_dv_agrouped]
    total_sd_or_var = sum([(srdv**n)/n_cols for srdv in sum_rtilde_per_dv_agrouped])

    total_mean = sum([m*dv for m, dv in zip(means, decision_variables)])

    return mu * total_sd_or_var - total_mean
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

def wrapper_fitness_func(static_data, hist_data):
    BIG_M = 1*(10**4)
    budget = 10000
    DIVERSITY_IMPORTANCE = 10
    LIQUIDITY_IMPORTANCE = 10
    RISK_IMPORTANCE = 10

    MAX_CONCENTRATION_PERCENTAGE = 0.30
    MAX_CONCENTRATION_PERCENTAGE_INTER_SECTOR = 0.30
    number_of_features = 3 # profit, liquidity, diversity
    risk_aversion = 0.8

    def fitness_func(solution, solution_idx):

        prices = [d['price'] for d in static_data]
        dys = [d['dy'] for d in static_data]

        total_price = numpy.sum(solution*prices)
        total_dy = numpy.sum(solution*dys)

        max_concentration_value = budget * MAX_CONCENTRATION_PERCENTAGE
        applications = [s*p for s, p in zip(solution, prices)]
        flag_max_concentration_percentage = any(map(lambda x : x > max_concentration_value , applications))

        flag_budget_limit = total_price > budget
        flag_non_negative_limit = any(map(lambda x : x < 0 , solution))

        min_diversity = 1;
        max_diversity = 2; # PARAMETER
        diversity = len(list(filter(lambda x : x > 0, solution)))
        normalized_diversity = norm01(diversity, min_diversity, max_diversity)


        min_liquidity = budget / max(prices)
        max_liquidity = budget / min(prices)
        liquidity = numpy.sum(solution)
        normalized_liquidity = norm01(liquidity, min_liquidity, max_liquidity)

        min_profit = total_dy * 0.90; # PARAMETER
        max_profit = total_dy * 1.10; # PARAMETER
        normalized_profit = norm01(total_dy, min_profit, max_profit)

        risk = markonno_risk(hist_data, solution)

        NEGATIVE_PART = (
            (BIG_M * flag_budget_limit) +
            (BIG_M * flag_non_negative_limit) +
            (BIG_M * flag_max_concentration_percentage) + 
            (RISK_IMPORTANCE * (-1 * risk))
        )

        POSITIVE_PART = (
            total_dy +
            (DIVERSITY_IMPORTANCE * diversity) +
            (LIQUIDITY_IMPORTANCE * liquidity)
        )

        # #maximize Z:
        # # normalized_profit
        # # + (normalized_liquidity * LIQUIDITY_IMPORTANCE)
        # # + (normalized_diversity * DIVERSITY_IMPORTANCE)
        # # + risk;

        # param risk_aversion = 0.80;
        # param number_of_features = 3;

        Z = (1-risk_aversion)*(
            (
            normalized_profit
            + (normalized_liquidity * LIQUIDITY_IMPORTANCE)
            + (normalized_diversity * DIVERSITY_IMPORTANCE)
            ) / (LIQUIDITY_IMPORTANCE*DIVERSITY_IMPORTANCE) * number_of_features
        ) + risk_aversion * (
            risk
        )

        # print(NEGATIVE_PART, POSITIVE_PART)

        fitness = POSITIVE_PART - NEGATIVE_PART
        return fitness
    return fitness_func

def read_data(static_file, hist_file):
    # --- Static Data ----------------------------------------------------------
    static_df = [
        {'asset' : 'MXRF11', 'sector' : 'Híbrido', 'price' : 9.58, 'lqdt' : 447468.0, 'div' : 0.10, 'dy' : 1.03},
        {'asset' : 'BCFF11', 'sector' : 'Títulos e Val. Mob.', 'price' : 64.17, 'lqdt' : 29039.0, 'div' : 0.54, 'dy' : 0.85},
        {'asset' : 'RECR11', 'sector' : 'Títulos e Val. Mob.', 'price' : 101.08, 'lqdt' : 56282.0, 'div' : 1.72, 'dy' : 1.69},
        {'asset' : 'IRDM11', 'sector' : 'Títulos e Val. Mob.', 'price' : 104.0, 'lqdt' : 57885.0, 'div' : 1.35, 'dy' : 1.28},
        {'asset' : 'XPLG11', 'sector' : 'Logística', 'price' : 94.0, 'lqdt' : 36645.0, 'div' : 0.70, 'dy' : 0.75},
    ]
    # --- Hist Data ----------------------------------------------------------
    hist_df = pd.read_excel(hist_file, engine='openpyxl', sheet_name='dy')
    return static_df, hist_df

def main():

    static_df, hist_df = read_data('', 'data_sample.xlsx')
    # print(static_df, '\n\n')
    # print(hist_df)

    # --- CONFIG ---------------------------------------------------------------
    # fitness_function = fitness_func
    fitness_function = wrapper_fitness_func(static_df, hist_df)

    num_generations = 2000
    num_parents_mating = 10

    sol_per_pop = 10
    num_genes = len(static_df)

    init_range_low = -2
    init_range_high = 5

    # --- CONFIG ---------------------------------------------------------------

    parent_selection_type = "sss"
    keep_parents = 2

    crossover_type = "single_point"

    mutation_type = "random"
    mutation_percent_genes = 30

    ga_instance = pygad.GA(num_generations=num_generations,
                        # gene_type=int,
                        num_parents_mating=num_parents_mating,
                        fitness_func=fitness_function,
                        sol_per_pop=sol_per_pop,
                        num_genes=num_genes,
                        #    init_range_low=0,
                        #    init_range_high=init_range_high,
                        parent_selection_type=parent_selection_type,
                        keep_parents=keep_parents,
                        crossover_type=crossover_type,
                        mutation_type=mutation_type,
                        mutation_percent_genes=mutation_percent_genes)

    ga_instance.run()

    solution, solution_fitness, solution_idx = ga_instance.best_solution()
    solution = [int(s) for s in solution]
    print("Parameters of the best solution : {solution}".format(solution=solution))
    print("Fitness value of the best solution = {solution_fitness}".format(solution_fitness=solution_fitness))

    prediction = numpy.sum(numpy.array([d['price'] for d in static_df])*solution)
    print("Predicted output based on the best solution : {prediction}".format(prediction=prediction))

    profit = numpy.sum(numpy.array([d['dy'] for d in static_df])*solution)
    print("Profit output based on the best solution : {profit}".format(profit=profit))

    risk = markonno_risk(hist_df, solution)
    print(f"RISK: {risk}")

    print(numpy.array([d['price'] for d in static_df])*solution)
    return solution_fitness

def evaluate_model(**params):
    ...

def grid_search():
    no_trains = 5
    # mean, best, worst
    metric = 'mean'
    params = {
        'num_generations' : [2000],
        'num_parents_mating' : [10],
        'sol_per_pop' : [10],
        'parent_selection_type' : ['sss'],
        'keep_parents' : [1, 2, 10],
        'crossover_type' : ['single_point'],
        'mutation_type' : ['random'],
        'mutation_percent_genes' : [30]
    }
    grid = []
    combinations = list(product(*list(params.values())))
    for i, comb in enumerate(combinations):
        performance = 0
        measured_time = 0
        for _ in range(no_trains):
            # decorate
            start = time.time()
            performance += evaluate_model(comb)
            end = time.time()
            total_time = end - start
            measured_time += total_time
        metric = performance / no_trains
        measured_time = measured_time / no_trains
        grid.append(
            {'test_num' : i, 'params': comb, 'metric' : metric, 'time' : measured_time}
        )



if(__name__ == '__main__'):
    print("START - Markonno GA")
    start = time.time()
    main()
    end = time.time()
    total_time = end - start
    print(f"\n{str(total_time)} secs.")