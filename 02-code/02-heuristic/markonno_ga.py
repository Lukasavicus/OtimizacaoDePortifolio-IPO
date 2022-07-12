import pygad, numpy, pandas as pd, time, json
from itertools import product
from copy import deepcopy

# === INPUT ====================================================================

# 0.0298, 56.3, 8.87, 46.40
# 0, 56, 8, 46
# 0 3 2 1

# === Helper ===================================================================
DEBUG = True
def agroup(list_, group_size):
    return [list_[i*group_size:(i+1)*group_size] for i in range(len(list_)//group_size)]

def norm01(metric, min_, max_):
    return (metric - min_) / (max_ - min_)
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

# === Markowitz & Konno and Yamazaki ===========================================
def markonno_risk(mtx, decision_variables, n=1, mu=2, export_min_max=False):
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
    sd_or_vars = [(srdv**n)/n_cols for srdv in sum_rtilde_per_dv_agrouped]
    total_sd_or_var = sum(sd_or_vars)

    means_per_dv = [m*dv for m, dv in zip(means, decision_variables)]
    total_mean = sum(means_per_dv)
    risk = mu * total_sd_or_var - total_mean

    if(export_min_max):
        max_risk = mu * max(sd_or_vars) - total_mean
        min_risk = mu * min(sd_or_vars) - total_mean
        return (risk, min_risk, max_risk)
    else:
        return risk
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

def wrapper_fitness_func(static_data, hist_data, budget, export_statistics=False):
    BIG_M = 1*(10**12)
    DIVERSITY_IMPORTANCE = 10
    LIQUIDITY_IMPORTANCE = 10
    RISK_IMPORTANCE = 10

    MAX_CONCENTRATION_PERCENTAGE = 0.30
    MAX_CONCENTRATION_PERCENTAGE_INTER_SECTOR = 0.30
    number_of_features = 3 # profit, liquidity, diversity
    risk_aversion = 0.8

    max_diversity = 91 # PARAMETER
    min_profit = 0.1 # PARAMETER
    max_profit = 1178.1 # PARAMETER

    # prices = [d['price'] for d in static_data]
    # dys = [d['dy'] for d in static_data]
    prices_and_dys = [(p, dy) for p, dy in zip(list(static_data['price']), list(static_data['dy'])) if ((p > 0) and (dy > 0))]
    prices = [pdy[0] for pdy in prices_and_dys]
    dys = [pdy[1] for pdy in prices_and_dys]

    def fitness_func(solution, solution_idx):


        total_price = numpy.sum(solution*prices)
        profit = numpy.sum(solution*dys) # TOTAL_DYs

        max_concentration_value = budget * MAX_CONCENTRATION_PERCENTAGE
        applications = [s*p for s, p in zip(solution, prices)]
        flag_max_concentration_percentage = any(map(lambda x : x > max_concentration_value , applications))

        flag_budget_limit = total_price > budget
        flag_non_negative_limit = any(map(lambda x : x < 0 , solution))

        min_diversity = 1;
        diversity = len(list(filter(lambda x : x > 0, solution)))
        normalized_diversity = norm01(diversity, min_diversity, max_diversity)


        min_liquidity = budget / max([p for p in prices if p < budget])
        max_liquidity = budget / min(prices)
        liquidity = numpy.sum(solution)
        normalized_liquidity = norm01(liquidity, min_liquidity, max_liquidity)

        normalized_profit = norm01(profit, min_profit, max_profit)

        risk, min_risk, max_risk = markonno_risk(hist_data, solution, export_min_max=True)
        normalized_risk = norm01(risk, min_risk, max_risk)

        RESTRICTIONS = (
            (BIG_M * flag_budget_limit) +
            (BIG_M * flag_non_negative_limit) +
            (BIG_M * flag_max_concentration_percentage)
        )

        if(risk < 0):
            risk *= -1

        NEGATIVE_PART = (
            RESTRICTIONS + 
            (RISK_IMPORTANCE * (risk))
        )

        POSITIVE_PART = (
            profit +
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

        POSITIVE_PART = (
            (
            normalized_profit
            + (normalized_liquidity * LIQUIDITY_IMPORTANCE)
            + (normalized_diversity * DIVERSITY_IMPORTANCE)
            ) / (LIQUIDITY_IMPORTANCE*DIVERSITY_IMPORTANCE) * number_of_features
        )

        NEGATIVE_PART = (RISK_IMPORTANCE * risk)

        val = ((1-risk_aversion) * POSITIVE_PART) - (risk_aversion * NEGATIVE_PART) - RESTRICTIONS

        # print(NEGATIVE_PART, POSITIVE_PART)

        # fitness = POSITIVE_PART - NEGATIVE_PART
        fitness = val
        statistics = {
                'diversity' : (min_diversity, max_diversity, diversity, normalized_diversity),
                'liquidity' : (min_liquidity, max_liquidity, liquidity, normalized_liquidity),
                'profit' : (min_profit, max_profit, profit, normalized_profit),
                'risk' : (min_risk, max_risk, risk, normalized_risk),
                'flags' : (flag_budget_limit, flag_non_negative_limit, flag_max_concentration_percentage),
                'dbg' : (
                    ((1-risk_aversion) * (POSITIVE_PART)),
                    ((  risk_aversion) * (NEGATIVE_PART)),
                    RESTRICTIONS,
                    ((1-risk_aversion) * (POSITIVE_PART))-
                    ((  risk_aversion) * (NEGATIVE_PART)) - RESTRICTIONS
                ),
                'parts' : (POSITIVE_PART, NEGATIVE_PART),
                'restrictions' : RESTRICTIONS,
                'val' : val
            }

        # print('='*100)
        # print('PORTIFOLIO PRICE:', total_price)
        # print(((1-risk_aversion) * (POSITIVE_PART))-((  risk_aversion) * (NEGATIVE_PART)) - RESTRICTIONS)
        # # print(solution, prices)
        # print(diversity, min_diversity, max_diversity, normalized_diversity)

        # print(val, normalized_profit, normalized_liquidity, normalized_diversity, risk, RESTRICTIONS)
        # print(solution)
        # print('-'*100)
        # # input()
        # print(statistics)

        if(export_statistics):
            return statistics
        else:
            return fitness
    return fitness_func

def read_data(static_file, hist_file):
    # --- Static Data ----------------------------------------------------------
    # static_df = [
    #     {'asset' : 'MXRF11', 'sector' : 'Híbrido', 'price' : 9.58, 'lqdt' : 447468.0, 'div' : 0.10, 'dy' : 1.03},
    #     {'asset' : 'BCFF11', 'sector' : 'Títulos e Val. Mob.', 'price' : 64.17, 'lqdt' : 29039.0, 'div' : 0.54, 'dy' : 0.85},
    #     {'asset' : 'RECR11', 'sector' : 'Títulos e Val. Mob.', 'price' : 101.08, 'lqdt' : 56282.0, 'div' : 1.72, 'dy' : 1.69},
    #     {'asset' : 'IRDM11', 'sector' : 'Títulos e Val. Mob.', 'price' : 104.0, 'lqdt' : 57885.0, 'div' : 1.35, 'dy' : 1.28},
    #     {'asset' : 'XPLG11', 'sector' : 'Logística', 'price' : 94.0, 'lqdt' : 36645.0, 'div' : 0.70, 'dy' : 0.75},
    # ]
    static_df = pd.read_csv(static_file)
    # --- Hist Data ----------------------------------------------------------
    # hist_df = pd.read_excel(hist_file, engine='openpyxl', sheet_name='dy')
    hist_df = pd.read_csv(hist_file)
    return static_df, hist_df

def main():

    static_df, hist_df = read_data('../../01-data/static_data.csv', '../../01-data/hist_data.csv')
    print(static_df, '\n\n')
    print(hist_df)

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
                        num_parents_mating=num_parents_mating,
                        fitness_func=fitness_function,
                        sol_per_pop=sol_per_pop,
                        num_genes=num_genes,
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

def evaluate_model_wrapper(budget):
    # static_df, hist_df = read_data('', 'data_sample.xlsx')
    static_df, hist_df = read_data('../../01-data/static_data.csv', '../../01-data/hist_data.csv')

    # -------------------------------------------------
    # this should be moved away to pre-proc script:
    f = static_df.apply(lambda row : row['dy'] > 0 and row['price'] > 0, axis=1)
    static_df = static_df[f]
    assets = list(static_df['asset'])
    for c in hist_df.columns:
        if( c not in assets):
            del hist_df[c]
    print(static_df.shape, hist_df.shape)
    # -------------------------------------------------




    fitness_function = wrapper_fitness_func(static_df, hist_df, budget)
    num_genes = len(static_df)

    def evaluate_model(params):
        parameters = deepcopy(params)

        # num_generations = 2000
        # num_parents_mating = 10
        # sol_per_pop = 10
        # parent_selection_type = "sss"
        # keep_parents = 2
        # crossover_type = "single_point"
        # mutation_type = "random"
        # mutation_percent_genes = 30

        complete_params = parameters
        complete_params['fitness_func'] = fitness_function
        complete_params['num_genes'] = num_genes
        complete_params['gene_space'] = [range(0, budget)] * num_genes

        ga_instance = pygad.GA(**complete_params)

        ga_instance.run()

        solution, solution_fitness, _ = ga_instance.best_solution()
        solution = [int(s) for s in solution]

        return solution_fitness, solution
    return evaluate_model

def grid_search():
    budget = 10000
    evaluate_model = evaluate_model_wrapper(budget)
    no_trains = 4
    # mean, best, worst
    metric = 'mean'
    # params = {
    #     'num_generations' : [5000, 10000, 20000],
    #     'num_parents_mating' : [5, 7],
    #     'sol_per_pop' : [5, 7],
    #     'parent_selection_type' : ['sss', 'rws', 'sus', 'rank'],
    #     'keep_parents' : [1, 2],
    #     'crossover_type' : ['single_point', 'uniform'],
    #     'mutation_type' : ['random', 'swap', 'adaptive'],
    #     'mutation_percent_genes' : [10]
    # }
    params = {
        'num_generations' : [1000],
        'num_parents_mating' : [5],
        'sol_per_pop' : [5],
        'parent_selection_type' : ['sss', 'rws', 'sus', 'rank'],
        'keep_parents' : [2],
        'crossover_type' : ['single_point'],
        'mutation_type' : ['random'],
        'mutation_percent_genes' : [10]
    }
    list_key = list(params.keys())
    grid = []
    combinations = list(product(*list(params.values())))
    for i, comb in enumerate(combinations):
        parameters = {k:v for k, v in zip(list_key, comb)}
        print(f"Testing {i+1} of {len(combinations)}, with params:", parameters)
        performance = 0
        measured_time = 0
        best_sol = None
        best_perform = 0
        print('-'*10)
        for j in range(no_trains):
            print(f"T{j+1}", end= ' ', flush=True)
            
            # decorate
            try:

                start = time.time()
                
                perform, solution = evaluate_model(parameters)
                
                end = time.time()
                total_time = end - start

                performance += perform
                if(perform > best_perform):
                    best_sol = solution
                measured_time += total_time
            except Exception as e:
                if(DEBUG): raise e
                print('ERR', e)
        print('\n')
        metric = performance / no_trains
        measured_time = measured_time / no_trains
        print(metric, measured_time)

        parameters['fitness_func'] = None
        parameters['gene_space'] = None

        grid.append(
            {'test_num' : i, 'params': parameters, 'metric' : metric, 'time' : measured_time, 'solution' : best_sol}
        )
        # print(grid)
    return grid

def linear_search():
    budget = 10000
    evaluate_model = evaluate_model_wrapper(budget)
    no_trains = 4
    # mean, best, worst
    metric = 'mean'
    # params = {
    #     'num_generations' : [5000, 10000, 20000],
    #     'num_parents_mating' : [5, 7],
    #     'sol_per_pop' : [5, 7],
    #     'parent_selection_type' : ['sss', 'rws', 'sus', 'rank'],
    #     'keep_parents' : [1, 2],
    #     'crossover_type' : ['single_point', 'uniform'],
    #     'mutation_type' : ['random', 'swap', 'adaptive'],
    #     'mutation_percent_genes' : [10]
    # }
    params = {
        'num_parents_mating' : [3, 5, 7, 10],
        'sol_per_pop' : [10, 15, 20],
        'keep_parents' : [1, 2, 3],
        'parent_selection_type' : ['sss', 'rank', 'sus', 'rws'],
        'crossover_type' : ['single_point', 'uniform'],
        'mutation_type' : ['random', 'swap', 'adaptive'],
        'mutation_percent_genes' : [10, 20, 30],
        'num_generations' : [1000, 5000, 10000, 20000, 100000]
    }
    params = {
        'num_parents_mating' : [3],
        'sol_per_pop' : [20],
        'keep_parents' : [2],
        'parent_selection_type' : ['rank'],
        'crossover_type' : ['uniform'],
        'mutation_percent_genes' : [10, 20, 30],
        'mutation_type' : ['random', 'swap'],
        'num_generations' : [1000, 5000, 10000, 20000, 100000]
    }
    list_key = list(params.keys())
    global_grid = []
    for i1 in range(len(params)):
        print(params)
        local_grid = []
        local_params = {k : v if i1 == j else [v[0]]  for j, (k, v) in enumerate(params.items())}
        combinations = list(product(*list(local_params.values())))
        for i2, comb in enumerate(combinations):
            parameters = {k:v for k, v in zip(list_key, comb)}
            print(f"Testing {i2+1} of {len(combinations)}, with params:", parameters)
            performance = 0
            measured_time = 0
            best_sol = None
            best_perform = 0
            print('-'*10)
            for j in range(no_trains):
                print(f"T{j+1}", end= ' ', flush=True)
                
                # decorate
                try:

                    start = time.time()
                    
                    perform, solution = evaluate_model(parameters)
                    
                    end = time.time()
                    total_time = end - start

                    performance += perform
                    if(perform > best_perform):
                        best_sol = solution
                    measured_time += total_time
                except Exception as e:
                    if(DEBUG): raise e
                    print('ERR', e)
            print('\n')
            metric = performance / no_trains
            measured_time = measured_time / no_trains
            print(metric, measured_time)

            parameters['fitness_func'] = None
            parameters['gene_space'] = None

            result = {'test_num' : i2, 'params': parameters, 'metric' : metric, 'time' : measured_time, 'solution' : best_sol}
            global_grid.append(result)
            local_grid.append(result)
            # print(grid)
        lg_sorted = sorted(local_grid, key=lambda x : x['metric'], reverse=True)
        best_params = lg_sorted[0]['params']
        k = list_key[i1]
        params[k] = [best_params[k]]

        # print(lg_sorted, best_params, k, params)

        evaluated = [(sol['params'][k], sol['metric']) for sol in lg_sorted]
        print('-'*80)
        print('evaluated', evaluated)
        print(f"best {k} : {best_params[k]}")
        print('-'*80)
    return global_grid

def timed_main():
    print("START - Markonno GA")
    start = time.time()
    main()
    end = time.time()
    total_time = end - start
    print(f"\n{str(total_time)} secs.")

def test():
    budget = 10000
    static_df, hist_df = read_data('../../01-data/static_data.csv', '../../01-data/hist_data.csv')
    # print(static_df, '\n\n')
    # print(hist_df)

    # --- CONFIG ---------------------------------------------------------------
    # fitness_function = fitness_func
    fitness_function = wrapper_fitness_func(static_df, hist_df, budget, True)

    solution = numpy.array([-98, 2, 4, -1, -13, -1, 0, -14, 11, 1, -13, -5, 2, -7, 9, 5, -5, -12, -1, 0, 0, -14, -3, -15, 20, -44, -9, -10, -16, 2, -2, 6, 3, 8, 9, -5, 0, 7, 13, -5, -4, 3, -5, -10, -10, 1, 1, 9, -8, 10, 0, 1, 3, 4, 2, 1, -9, -13, 0, 0, -2, 10, 32, -59, 5, -4, 0, -2, -16, 1, -10, 11, -11, -2, -1, -5, -3, 1, -4, 7, -6, 0, 3, 2, 8, -1, 0, -5, 11, 1, 1, -14, -25, -13, 0, 3, -2, 4, 5, 3, -3, 4, -2, -1, 1, 6, 1, 0, -15, -9, 1, -4, -6, -2, -5, 29, -4, -3, -7, 9, -3, 5, 4, -1, 5, -9, -2, -3, 3, -3, 11, -8, 6, 3, -3, 9, 1, -20, 4, 7, 13, -3, 6, -5, -5, -7, -8, -15, -1, -3, 0, 3, 0, 0, 0, 2, -4, 6, -7, -1, -2, 1, 8, 3, -7, -1, -2, 2, -2, -11, -7, 2, -19, 1, -6, -5, -2, 4, 1, 1, 3, 0, 2, 7, -15, 2, -9, 6, 13, -4, -15, 2, 7, 1, -10, 5, -4, -5, -7, -2, 11, -18, -10, -4, -3, 0, 10, -16, 0, 0, 7, 8, 3, -6, -3, 0, -6, 0, 9, -36, 13, 0, 17, 0, -7, -19, -9, 17, -11, -4, -13, 3, -1, -7, -4, 0, 15, -20, 11, -6, -5, 1, 10, -5, -7, 1, 0, -3, -7, -15, 4, -8, -167, -149])
    data = fitness_function(solution, None)
    print('data', data)

def exec(type):
    if(type == 'grid'):
        grid = grid_search()
    elif(type == 'linear'):
        grid = linear_search()
    grid_sorted = sorted(grid, key=lambda x : x['metric'], reverse=True)
    print(grid_sorted)
    with open('data.json', 'w') as f:
        json.dump(grid_sorted, f)

if(__name__ == '__main__'):
    # test()
    exec('linear')
    # exec('grid')