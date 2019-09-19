import math
import numpy as np
import random
import matplotlib.pyplot as plt

def real_to_binary(vector, precision_array, left_array, right_array):
    l = np.ceil(np.log2(right_array - left_array) + precision_array * np.log2(10))
    l = l.astype(np.int64)
    ints = np.rint((vector - left_array) / (1 / np.power(10, precision_array)))
    ints = ints.astype(np.int64)
    binary = np.zeros(np.sum(l, dtype=np.int64), dtype=np.int64)

    i = 0
    l_accum = 0
    for integer in ints:
        quotient = integer
        bin = []
        while quotient != 0:
            remainder = np.rint(quotient % 2).astype(np.int64)
            quotient = quotient // 2             
            bin.append(remainder)

        idx = l[i] - 1 + l_accum
        k = 0
        for j in range(len(bin)):
            binary[idx] = bin[k]
            idx -= 1
            k += 1
        l_accum += l[i]
        i += 1
    return binary, l
        
def binary_to_real(binary_vector, precision_array, left_array, l):
    accum_vector = np.zeros(l.shape[0])
    idx = 0
    for j in range(l.shape[0]):
        for i in reversed(range(l[j])):
            accum_vector[j] += (2 ** i) * binary_vector[idx]
            idx += 1
    real_vector = left_array + (accum_vector * (1 / np.power(10, precision_array)))
    return real_vector

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

def roulette_selection(A, X, n, sel_population):
    B = np.zeros(A.shape[0])
    B[0] = A[0]
    for i in range(A.shape[0] - 1):
        B[i + 1] = A[i + 1] + B[i]
   
    for i in range(n):
        u = B[A.shape[0] - 1] * random.uniform(0, 1)
        j = 0
        while u > B[j]:
            j += 1
        sel_population = np.append(sel_population, X[j])
    
    return sel_population

def stochastic_leftover_selection(A, X, n, sel_population):
    b_m = np.sum(A)
    m = A.shape[0]
    E = m * (A / b_m)
    
    int_part = np.trunc(E)
    int_part = int_part.astype(np.int64)
    leftover = E - int_part
    for i in range(int_part.shape[0]):
        for j in range(int_part[i]):
            sel_population = np.append(sel_population, X[i])

    size_solution = X[0].shape[0]
    sel_population = roulette_selection(leftover, X, n - sel_population.shape[0]//size_solution, sel_population)

    return sel_population

def one_point_cross(A, B):
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

def two_point_cross(A, B):
    size_parent = A.shape[0]
    u1 = random.randrange(size_parent - 1)
    u2 = random.randrange(size_parent - 1)
    while(u2 == u1):
        u2 = random.randrange(size_parent - 1)

    if u1 > u2:
        aux = u1
        u1 = u2
        u2 = aux

    children1 = np.zeros(size_parent)
    children2 = np.zeros(size_parent)
    i = 0
    while i < u1 + 1:
        children1[i] = A[i]
        children2[i] = B[i]
        i += 1
    while i < u2 + 1:
        children1[i] = B[i]
        children2[i] = A[i]
        i += 1
    while i < size_parent:
        children1[i] = A[i]
        children2[i] = B[i]
        i += 1

    return children1, children2

def uniform_cross(A, B):
    size_parent = A.shape[0]
    children1 = np.zeros(size_parent)
    children2 = np.zeros(size_parent)
    for i in range(size_parent):
        u = random.uniform(0, 1)
        if u < 0.5:
            children1[i] = A[i]
            children2[i] = B[i]
        else:
            children1[i] = B[i]
            children2[i] = A[i]

    return children1, children2

def linear_normalization(X, A, v_min, v_max):
    m = A.shape[0]
    idx_sorted = np.argsort(A)
    A_new = np.zeros(m)
    for i in range(m):
        A_new[idx_sorted[i]] = v_min + ((v_max - v_min) / (m-1)) * i
    return A_new

def get_best_individual(A):
    a_best = -1
    idx_best = -1
    for i in range(A.shape[0]):
        if A[i] > a_best:
            a_best = A[i]
            idx_best = i

    return idx_best

def mutation(x_bin):
    size_x_bin = x_bin.shape[0]
    u = random.randrange(size_x_bin)
    if (x_bin[u] == 1):
        x_bin[u] = 0
    else:
        x_bin[u] = 1
    return x_bin

def f1(x1, x2):
    return 0.5 - ((math.sin(math.sqrt(x1**2 + x2**2))**2 )-0.5 / (1.0+0.001*(x1**2 + x2**2))**2)

def evaluate_f1(X):
    size_pop = X.shape[0]
    A = np.zeros(size_pop)
    for i in range(size_pop):
        A[i] = f1(X[i][0], X[i][1])
    return A

def traditional_genetic_algorithm(eval, max_gen, size_population, left_array, right_array, precision_array,
                                  cross_prob, mutation_prob, elitism, cross, selection, linear_norm, bias):
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
            A = make_aptitude_positive(A, bias)
        else:
            A = linear_normalization(X, A, linear_norm[1], linear_norm[2])
        n = 0
        if A.shape[0] % 2 == 0:
            n = A.shape[0]
        else:
            n = A.shape[0] - 1
        sel_population = np.array([])
        if selection == "roulette":
            sel_population = roulette_selection(A, X, n, sel_population)
        else: # == "stochastic leftover"
            sel_population = stochastic_leftover_selection(A, X, n, sel_population)
        j = 0
        k = 0
        if elitism == True:
            new_population = np.zeros((n + 1, size_solution))
            idx_best = get_best_individual(A)
            new_population[k] = X[idx_best]
            k += 1
        else:
            new_population = np.zeros((n, size_solution))
        for i in range(n//2):
            parent_binary1, l = real_to_binary(sel_population[j], precision_array, left_array, right_array)
            j += 1
            parent_binary2, l = real_to_binary(sel_population[j], precision_array, left_array, right_array)
            j += 1
            r_cross = random.uniform(0, 1)
            if r_cross < cross_prob:
                if cross == "one point cross":
                    children1, children2 = one_point_cross(parent_binary1, parent_binary2)
                elif cross == "two point cross":
                    children1, children2 = two_point_cross(parent_binary1, parent_binary2)
                else:    # == "uniform cross"
                    children1, children2 = uniform_cross(parent_binary1, parent_binary2)
            else:
                children1, children2 = parent_binary1, parent_binary2
            
            r_mutation = random.uniform(0, 1)
            if r_mutation < mutation_prob:
                children1 = mutation(children1)
            r_mutation = random.uniform(0, 1)
            if r_mutation < mutation_prob:
                children2 = mutation(children2)
            
            ch1_real = binary_to_real(children1, precision_array, left_array, l)
            ch2_real = binary_to_real(children2, precision_array, left_array, l)
            new_population[k] = ch1_real
            k += 1
            new_population[k] = ch2_real
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
    left_array = np.array([-100, -100])
    right_array = np.array([100, 100])
    precision_array = np.array([7, 8])
    cross_prob = 0.65
    mutation_prob = 0.008
    elitism = True
    cross_vec = ["one point cross", "two point cross", "uniform cross"]
    selection = "roulette"
    linear_norm = [False, 0, 0]
    bias = 5 # para make_aptitude_positive()

    for cross in cross_vec:
        A_best_generations_avg = np.zeros(max_gen)
        A_best_generations_accum_avg = np.zeros(max_gen)
        A_mean_generations_accum_avg = np.zeros(max_gen)
        for i in range(num_executions):
            X, A, A_best_generations, A_mean_generations = traditional_genetic_algorithm(evaluate_f1, max_gen, size_population, left_array, right_array, precision_array,
                                                                                         cross_prob, mutation_prob, elitism, cross, selection, linear_norm, bias)
            
            print("A_best_generations:\n", A_best_generations)
            print("A_mean_generations:\n", A_mean_generations)
            
            A_best_generations_avg += A_best_generations
            A_best_generations_accum_avg += get_accum_mean(A_best_generations)
            A_mean_generations_accum_avg += get_accum_mean(A_mean_generations)
        A_best_generations_avg /= num_executions
        A_best_generations_accum_avg /= num_executions
        A_mean_generations_accum_avg /= num_executions
        graph_monitor_genetic_alg(A_best_generations_avg, A_best_generations_accum_avg, A_mean_generations_accum_avg, cross)

def experiment2(num_executions):
    max_gen = 40
    size_population = 100
    left_array = np.array([-100, -100])
    right_array = np.array([100, 100])
    precision_array = np.array([7, 8])
    params = np.array([[0.65, 0.008],
                       [0.85, 0.008],
                       [0.55, 0.008],
                       [0.25, 0.008],
                       [0.65, 0.004],
                       [0.65, 0.01],
                       [0.65, 0.02],
                       [0.65, 0.05]])
    elitism = True
    cross = "uniform cross"
    selection = "stochastic leftover"
    linear_norm = [False, 0, 0]
    bias = 5 # para make_aptitude_positive()

    for i in range(params.shape[0]):
        A_best_generations_avg = np.zeros(max_gen)
        A_best_generations_accum_avg = np.zeros(max_gen)
        A_mean_generations_accum_avg = np.zeros(max_gen)
        for j in range(num_executions):
            X, A, A_best_generations, A_mean_generations = traditional_genetic_algorithm(evaluate_f1, max_gen, size_population, left_array, right_array, precision_array,
                                                                                         params[i, 0], params[i, 1], elitism, cross, selection, linear_norm, bias)
            print("A_best_generations:\n", A_best_generations)
            print("A_mean_generations:\n", A_mean_generations)
            
            A_best_generations_avg += A_best_generations
            A_best_generations_accum_avg += get_accum_mean(A_best_generations)
            A_mean_generations_accum_avg += get_accum_mean(A_mean_generations)
        A_best_generations_avg /= num_executions
        A_best_generations_accum_avg /= num_executions
        A_mean_generations_accum_avg /= num_executions
        graph_monitor_genetic_alg(A_best_generations_avg, A_best_generations_accum_avg, A_mean_generations_accum_avg, "prob_cruce=%f, prob_mut=%f" % (params[i, 0], params[i, 1]))
    
def experiment3(num_executions):
    max_gen = 40
    size_population = 100
    left_array = np.array([-100, -100])
    right_array = np.array([100, 100])
    precision_array = np.array([7, 8])
    cross_prob = 0.25
    mutation_prob = 0.008
    elitism = False
    cross = "uniform cross"
    selection = "stochastic leftover"
    linear_norm = [False, 0, 0]
    bias = 5 # para make_aptitude_positive()

    A_best_generations_avg = np.zeros(max_gen)
    A_best_generations_accum_avg = np.zeros(max_gen)
    A_mean_generations_accum_avg = np.zeros(max_gen)

    for j in range(num_executions):
        X, A, A_best_generations, A_mean_generations = traditional_genetic_algorithm(evaluate_f1, max_gen, size_population, left_array, right_array, precision_array,
                                                                                     cross_prob, mutation_prob, elitism, cross, selection, linear_norm, bias)
        print("A_best_generations:\n", A_best_generations)
        print("A_mean_generations:\n", A_mean_generations)
        
        A_best_generations_avg += A_best_generations
        A_best_generations_accum_avg += get_accum_mean(A_best_generations)
        A_mean_generations_accum_avg += get_accum_mean(A_mean_generations)
    A_best_generations_avg /= num_executions
    A_best_generations_accum_avg /= num_executions
    A_mean_generations_accum_avg /= num_executions
    graph_monitor_genetic_alg(A_best_generations_avg, A_best_generations_accum_avg, A_mean_generations_accum_avg, "Ni normalizacion lineal ni elitismo")

    
    elitism = True

    A_best_generations_avg = np.zeros(max_gen)
    A_best_generations_accum_avg = np.zeros(max_gen)
    A_mean_generations_accum_avg = np.zeros(max_gen)

    for j in range(num_executions):
        X, A, A_best_generations, A_mean_generations = traditional_genetic_algorithm(evaluate_f1, max_gen, size_population, left_array, right_array, precision_array,
                                                                                     cross_prob, mutation_prob, elitism, cross, selection, linear_norm, bias)
        print("A_best_generations:\n", A_best_generations)
        print("A_mean_generations:\n", A_mean_generations)
        
        A_best_generations_avg += A_best_generations
        A_best_generations_accum_avg += get_accum_mean(A_best_generations)
        A_mean_generations_accum_avg += get_accum_mean(A_mean_generations)
    A_best_generations_avg /= num_executions
    A_best_generations_accum_avg /= num_executions
    A_mean_generations_accum_avg /= num_executions
    graph_monitor_genetic_alg(A_best_generations_avg, A_best_generations_accum_avg, A_mean_generations_accum_avg, "Solo elitismo")


    linear_norm = [True, 10, 60]

    A_best_generations_avg = np.zeros(max_gen)
    A_best_generations_accum_avg = np.zeros(max_gen)
    A_mean_generations_accum_avg = np.zeros(max_gen)

    for j in range(num_executions):
        X, A, A_best_generations, A_mean_generations = traditional_genetic_algorithm(evaluate_f1, max_gen, size_population, left_array, right_array, precision_array,
                                                                                     cross_prob, mutation_prob, elitism, cross, selection, linear_norm, bias)
        print("A_best_generations:\n", A_best_generations)
        print("A_mean_generations:\n", A_mean_generations)
        
        A_best_generations_avg += A_best_generations
        A_best_generations_accum_avg += get_accum_mean(A_best_generations)
        A_mean_generations_accum_avg += get_accum_mean(A_mean_generations)
    A_best_generations_avg /= num_executions
    A_best_generations_accum_avg /= num_executions
    A_mean_generations_accum_avg /= num_executions
    graph_monitor_genetic_alg(A_best_generations_avg, A_best_generations_accum_avg, A_mean_generations_accum_avg, "Norm Lineal 10 - 60 y elitismo")


    linear_norm = [True, 1, 200]

    A_best_generations_avg = np.zeros(max_gen)
    A_best_generations_accum_avg = np.zeros(max_gen)
    A_mean_generations_accum_avg = np.zeros(max_gen)

    for j in range(num_executions):
        X, A, A_best_generations, A_mean_generations = traditional_genetic_algorithm(evaluate_f1, max_gen, size_population, left_array, right_array, precision_array,
                                                                                     cross_prob, mutation_prob, elitism, cross, selection, linear_norm, bias)
        print("A_best_generations:\n", A_best_generations)
        print("A_mean_generations:\n", A_mean_generations)
        
        A_best_generations_avg += A_best_generations
        A_best_generations_accum_avg += get_accum_mean(A_best_generations)
        A_mean_generations_accum_avg += get_accum_mean(A_mean_generations)
    A_best_generations_avg /= num_executions
    A_best_generations_accum_avg /= num_executions
    A_mean_generations_accum_avg /= num_executions
    graph_monitor_genetic_alg(A_best_generations_avg, A_best_generations_accum_avg, A_mean_generations_accum_avg, "Norm Lineal 1 - 200 y elitismo")



def main():
    max_gen = 40
    size_population = 100
    left_array = np.array([-100, -100])
    right_array = np.array([100, 100])
    precision_array = np.array([6, 8])
    cross_prob = 0.25
    mutation_prob = 0.008
    elitism = True
    #cross = "one point cross"
    cross = "two point cross"
    #cross = "uniform cross"
    #selection = "roulette"
    selection = "stochastic leftover"
    #linear_norm = [False, 0, 0]
    linear_norm = [True, 5, 100]
    bias = 5 # para make_aptitude_positive()

    X, A, A_best_generations, A_mean_generations = traditional_genetic_algorithm(evaluate_f1, max_gen, size_population, left_array, right_array, precision_array,
                                                                                 cross_prob, mutation_prob, elitism, cross, selection, linear_norm, bias)

    print("A_best_generations:\n", A_best_generations)
    print("A_mean_generations:\n", A_mean_generations)

    A_best_generations_accum = get_accum_mean(A_best_generations)
    A_mean_generations_accum = get_accum_mean(A_mean_generations)
    graph_monitor_genetic_alg(A_best_generations, A_best_generations_accum, A_mean_generations_accum, 'Prueba')

    num_executions = 20
    #experiment1(num_executions)
    #experiment2(num_executions)
    experiment3(num_executions)

main()