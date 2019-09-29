import math
import numpy as np
import random
import matplotlib.pyplot as plt


def make_aptitude_positive(A, bias):
    a_min = A[0]
    idx_min = 0
    for i in range(A.shape[0] - 1):
        if (A[i+1] < a_min):
            idx_min = i+1
            a_min = A[i+1]
    if a_min < 0:
        A = A - a_min
        A = A + bias
    return A

def linear_normalization(X, A, v_min, v_max):
    m = A.shape[0]
    idx_sorted = np.argsort(A)
    A_new = np.zeros(m)
    for i in range(m):
        A_new[idx_sorted[i]] = v_min + ((v_max - v_min) / (m-1)) * i
    return A_new

def roulette_selection(A, X, n, sel_population, idx):
    B = np.zeros(A.shape[0])
    B[0] = A[0]
    for i in range(A.shape[0] - 1):
        B[i + 1] = A[i + 1] + B[i]
   
    for i in range(n):
        u = B[A.shape[0] - 1] * random.uniform(0, 1)
        j = 0
        while u > B[j]:
            j += 1
        sel_population[idx] = X[j]
        idx += 1
    
    return sel_population

def stochastic_leftover_selection(A, X, n, sel_population):
    b_m = np.sum(A)
    m = A.shape[0]
    E = m * (A / b_m)
    
    int_part = np.trunc(E)
    int_part = int_part.astype(np.int64)
    leftover = E - int_part
    idx = 0
    i = 0
    j = 0
    while i < int_part.shape[0] and idx < sel_population.shape[0]:
        while j < int_part[i] and idx < sel_population.shape[0]:
            sel_population[idx] = X[i]
            idx += 1            
            j += 1
        i += 1

    size_solution = X[0].shape[0]
    sel_population = roulette_selection(leftover, X, n - idx, sel_population, idx)

    return sel_population

def arithmetic_cross(A, B):
    alpha = random.uniform(0, 1)
    children1 = alpha * A + (1 - alpha) * B
    children2 = (1 - alpha) * A + alpha * B

    return children1, children2

def simple_cross(A, B):
    size_parent = A.shape[0]
    u = random.randrange(size_parent - 1)
    children1 = np.zeros(size_parent)
    children2 = np.zeros(size_parent)
    i = 0
    while i < u + 1:
        children1[i] = A[i]
        children2[i] = B[i]
        i += 1
    while i < size_parent:
        children1[i] = B[i]
        children2[i] = A[i]
        i += 1

    return children1, children2

def heuristic_cross(A, B, A_apt, B_apt, max_iter, left_array, right_array):
    children = np.zeros(A.shape[0])
    valid = False
    count = 0
    while valid == False and count < max_iter:
        r = random.uniform(0, 1)
        if A_apt > B_apt:
            children = r * (A - B) + A
        else:
            children = r * (B - A) + B
        
        valid = True
        i = 0
        while i < children.shape[0] and valid == True:
            if children[i] < left_array[i] or children[i] > right_array[i]:
                valid = False
            i += 1
        count += 1
    return children, valid

def get_best_individual(A):
    a_best = A[0]
    idx_best = 0
    for i in range(A.shape[0] - 1):
        if A[i+1] > a_best:
            a_best = A[i+1]
            idx_best = i+1

    return idx_best

def uniform_mutation(x_n, left_array, right_array):
    k = random.randrange(x_n.shape[0])
    u = random.uniform(0, 1)
    x_k = left_array[k] + (right_array[k] - left_array[k]) * u
    x_n[k] = x_k
    return x_n

def boundary_mutation(x_n, left_array, right_array):
    k = random.randrange(x_n.shape[0])
    b = random.uniform(0, 1)
    x_k = 0
    if b <= 0.5:
        x_k = left_array[k]
    else:
        x_k = right_array[k]
    x_n[k] = x_k
    return x_n

def delta(t, y, T, beta):
    r = random.uniform(0, 1)
    return y * r * math.pow(1 - t/T, beta)

def non_uniform_mutation(x_n, left_array, right_array, t, T, beta):
    k = random.randrange(x_n.shape[0])
    b = random.uniform(0, 1)
    if b <= 0.5:
        x_k = x_n[k] + delta(t, right_array[k] - x_n[k], T, beta)
    else:
        x_k = x_n[k] - delta(t, x_n[k] - left_array[k], T, beta)
    x_n[k] = x_k
    return x_n

def fn(x_n):
    sum_power_x_n = np.sum(np.power(x_n, 2))
    return 0.5 - (np.power(np.sin(np.sqrt(sum_power_x_n)), 2) - 0.5) / (np.power(1.0 + 0.001*sum_power_x_n, 2))

def evaluate_fn(X):
    size_pop = X.shape[0]
    A = np.zeros(size_pop)
    for i in range(size_pop):
        A[i] = fn(X[i])
    return A

def real_coded_genetic_algorithm(eval, f, max_gen, size_population, left_array, right_array,
                                 cross_prob, mutation_prob, elitism, cross, mutation, linear_norm):
    t = 0
    A_best_generations = np.zeros(max_gen)
    A_mean_generations = np.zeros(max_gen)
    size_solution = left_array.shape[0]
    X = np.zeros((size_population, size_solution))
    for i in range(size_population):
        for j in range(size_solution):
            X[i, j] = random.uniform(left_array[j], right_array[j])
    A = eval(X)
    while t < max_gen:
        idx_best = get_best_individual(A)
        A_best_generations[t] = A[idx_best]
        A_mean_generations[t] = np.mean(A)

        if linear_norm[0] == False:
            A = make_aptitude_positive(A, linear_norm[1])
        else:
            A = linear_normalization(X, A, linear_norm[1], linear_norm[2])
        n = 0
        if A.shape[0] % 2 == 0:
            n = A.shape[0]
        else:
            n = A.shape[0] - 1

        sel_population = np.zeros((n, size_solution))
        sel_population = stochastic_leftover_selection(A, X, n, sel_population)
        
        k = 0
        new_population = np.zeros((n, size_solution))
        if elitism == True:
            new_population = np.zeros((n + 1, size_solution))
            new_population[k] = X[idx_best]
            k += 1
            
        for i in range(n//2):
            r_cross = random.uniform(0, 1)
            parent1 = sel_population[i*2]
            parent2 = sel_population[i*2 + 1]
            if r_cross < cross_prob:
                if cross == "arithmetic cross":
                    children1, children2 = arithmetic_cross(parent1, parent2)
                elif cross == "simple cross":
                    children1, children2 = simple_cross(parent1, parent2)
                else:    # == "heuristic cross"
                    children1, valid1 = heuristic_cross(parent1, parent2, f(parent1), f(parent2), 30, left_array, right_array)
                    if valid1 == False:
                        children1 = parent1                        
                    children2, valid2 = heuristic_cross(parent1, parent2, f(parent1), f(parent2), 30, left_array, right_array)
                    if valid2 == False:
                        children2 = parent2
            else:
                children1, children2 = parent1, parent2
            
            r_mutation = random.uniform(0, 1)
            if r_mutation < mutation_prob:
                if mutation == "uniform mutation":
                    children1 = uniform_mutation(children1, left_array, right_array)
                elif mutation == "boundary mutation":
                    children1 = boundary_mutation(children1, left_array, right_array)
                else:        #== "non uniform mutation"
                    children1 = non_uniform_mutation(children1, left_array, right_array, t, max_gen, 2)
            r_mutation = random.uniform(0, 1)
            if r_mutation < mutation_prob:
                if mutation == "uniform mutation":
                    children2 = uniform_mutation(children2, left_array, right_array)
                elif mutation == "boundary mutation":
                    children2 = boundary_mutation(children2, left_array, right_array)
                else:        #== "non uniform mutation"
                    children2 = non_uniform_mutation(children2, left_array, right_array, t, max_gen, 2)
            
            new_population[k] = children1
            k += 1
            new_population[k] = children2
            k += 1

        X = new_population
        A = eval(X)
        #print("X:\n", X)
        #print("A:\n", A)
        t += 1
    return X, A, A_best_generations, A_mean_generations

def get_accum_mean(vec):
    vec_accum = np.zeros(vec.shape[0])
    vec_accum[0] = vec[0]
    for i in range(vec.shape[0] - 1):
        vec_accum[i+1] = vec[i+1] + vec_accum[i]
    div = np.arange(vec.shape[0]) + 1
    vec_accum = vec_accum / div
    return vec_accum

def graph_monitor_genetic_alg(A_best_generations, A_best_generations_accum, A_mean_generations_accum, title):
    fig, ax = plt.subplots()

    ax.plot(np.arange(A_best_generations.shape[0]), A_best_generations, 'b', label='Curva de mejores elementos')
    ax.plot(np.arange(A_best_generations_accum.shape[0]), A_best_generations_accum, 'g', label='Curva offline')
    ax.plot(np.arange(A_mean_generations_accum.shape[0]), A_mean_generations_accum, 'r', label='Curva online')
    plt.ylabel('aptitud')
    plt.xlabel('generacion')
    legend = plt.legend(loc=0, shadow=True, fontsize='x-large')
    plt.title(title)
    major_ticks_y = np.arange(0, 1.05, 0.2)
    minor_ticks_y = np.arange(0, 1.05, 0.05)
    major_ticks_x = np.arange(0, 41, 5)
    minor_ticks_x = np.arange(0, 41, 1)

    ax.set_xticks(major_ticks_x)
    ax.set_xticks(minor_ticks_x, minor=True)
    ax.set_yticks(major_ticks_y)
    ax.set_yticks(minor_ticks_y, minor=True)

    # And a corresponding grid
    ax.grid(which='both')

    # Or if you want different settings for the grids:
    ax.grid(which='minor', alpha=0.2)
    ax.grid(which='major', alpha=0.5)
    #ax.grid(True)

    plt.show()

def experiment1(num_executions):
    max_gen = 40
    size_population = 100
    n_vec = np.array([2, 10, 50])
    cross_prob = 0.65
    mutation_prob = 0.008
    elitism = True
    cross_vec = ["arithmetic cross", "simple cross", "heuristic cross"]
    mutation_vec = ["uniform mutation", "boundary mutation", "non uniform mutation"]
    linear_norm = [False, 5]
    #linear_norm = [True, 5, 100]

    for cross in cross_vec:
        for mutation in mutation_vec:
            for n in n_vec:
                left_array = np.full((n), -100.0)
                right_array = np.full((n), 100.0)
                A_best_generations_avg = np.zeros(max_gen)
                A_best_generations_accum_avg = np.zeros(max_gen)
                A_mean_generations_accum_avg = np.zeros(max_gen)
                for i in range(num_executions):
                    X, A, A_best_generations, A_mean_generations = real_coded_genetic_algorithm(evaluate_fn, fn, max_gen, size_population, left_array, right_array,
                                                                                                cross_prob, mutation_prob, elitism, cross, mutation, linear_norm)
                    
                    print("A_best_generations:\n", A_best_generations)
                    print("A_mean_generations:\n", A_mean_generations)
                    
                    A_best_generations_avg += A_best_generations
                    A_best_generations_accum_avg += get_accum_mean(A_best_generations)
                    A_mean_generations_accum_avg += get_accum_mean(A_mean_generations)
                A_best_generations_avg /= num_executions
                A_best_generations_accum_avg /= num_executions
                A_mean_generations_accum_avg /= num_executions
                graph_monitor_genetic_alg(A_best_generations_avg, A_best_generations_accum_avg, A_mean_generations_accum_avg, "cross=%s, mutation=%s, n=%d" % (cross, mutation, n))

def main():
    max_gen = 40
    size_population = 100
    left_array = np.array([-100, -100])
    right_array = np.array([100, 100])
    cross_prob = 0.65
    mutation_prob = 0.008
    elitism = True
    #cross = "arithmetic cross"
    cross = "simple cross"
    #cross = "heuristic cross"
    mutation = "uniform mutation"
    #mutation = "boundary mutation"
    #mutation = "non uniform mutation"
    linear_norm = [False, 5]
    #linear_norm = [True, 5, 100]

    X, A, A_best_generations, A_mean_generations = real_coded_genetic_algorithm(evaluate_fn, fn, max_gen, size_population, left_array, right_array,
                                                                                cross_prob, mutation_prob, elitism, cross, mutation, linear_norm)

    print("A_best_generations:\n", A_best_generations)
    print("A_mean_generations:\n", A_mean_generations)

    A_best_generations_accum = get_accum_mean(A_best_generations)
    A_mean_generations_accum = get_accum_mean(A_mean_generations)
    graph_monitor_genetic_alg(A_best_generations, A_best_generations_accum, A_mean_generations_accum, 'Prueba')

    num_executions = 20
    experiment1(num_executions)


main()