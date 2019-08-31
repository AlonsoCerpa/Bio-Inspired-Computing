import numpy as np
import math
from random import randrange
import random

def hartley_function(vec_prob):
    return math.log(len(vec_prob), 2)

def shannon_entropy(vec_prob):
    entropy = 0.0
    for prob in vec_prob:
        if prob > 0:
            entropy += -prob * math.log(prob, 2)
    return entropy

def vec_prob_from_text(file_name):
    alphabet_min = "abcdefghijklmnñopqrstuvwxyzáéíóú"
    alphabet_may = "ABCDEFGHIJKLMNÑOPQRSTUVWXYZÁÉÍÓÚ"
    space = " "
    with open(file_name, 'r') as file:
        data = file.read()
    i = 0
    vec_prob = np.zeros((len(alphabet_min) + 1))
    global_count = 0
    while i < len(alphabet_min):
        count = data.count(str(alphabet_min[i]))
        count += data.count(str(alphabet_may[i]))
        vec_prob[i] = count
        global_count += count
        i += 1
    count = data.count(str(space))
    vec_prob[i] = count
    global_count += count
    for i in range(len(vec_prob)):
        vec_prob[i] /= global_count
    return vec_prob
        
vec_prob = vec_prob_from_text("data.txt")
#vec_prob = vec_prob_from_text("lipograma1.txt")
#vec_prob = vec_prob_from_text("lipograma2.txt")

#print(vec_prob)
print(hartley_function(vec_prob))
print(shannon_entropy(vec_prob))

def generate_random_text(n, file_name):
    alphabet = "abcdefghijklmnñopqrstuvwxyzáéíóú "
    random_text = ""
    for i in range(n):
        random_text += alphabet[randrange(len(alphabet))]
    text_file = open(file_name, "w")
    text_file.write(random_text)
    text_file.close()

def generate_text_with_vec_prob(n, file_name, vec_prob):
    accum_probs = []
    aux_prob = 0
    for i in range(len(vec_prob)):
        aux_prob += vec_prob[i]
        accum_probs.append(aux_prob)
    alphabet = "abcdefghijklmnñopqrstuvwxyzáéíóú "
    random_text = ""
    for i in range(n):
        x = random.uniform(0, 1)
        j = 0
        while x > accum_probs[j]:
            j += 1
        random_text += alphabet[j]
    text_file = open(file_name, "w")
    text_file.write(random_text)
    text_file.close()

def random_permutation(file_name, output_file_name, num_swaps):
    with open(file_name, 'r') as file:
        text = file.read()
    text = list(text)
    for i in range(num_swaps):
        idx1 = randrange(len(text))
        idx2 = randrange(len(text))
        char_aux = text[idx1]
        text[idx1] = text[idx2]
        text[idx2] = char_aux
    text_file = open(output_file_name, "w")
    text_file.write("".join(text))
    text_file.close()

generate_random_text(4000, "random_text.txt")
rand_vec_prob = vec_prob_from_text("random_text.txt")
print(rand_vec_prob)
print(hartley_function(rand_vec_prob))
print(shannon_entropy(rand_vec_prob))


generate_text_with_vec_prob(4000, "vec_prob_text.txt", vec_prob)
vec_prob_text = vec_prob_from_text("vec_prob_text.txt")
print(vec_prob_text)
print(hartley_function(vec_prob_text))
print(shannon_entropy(vec_prob_text))

random_permutation("data.txt", "data_permuted.txt", 10000)
vec_prob_perm = vec_prob_from_text("data_permuted.txt")
print(hartley_function(vec_prob_perm))
print(shannon_entropy(vec_prob_perm))