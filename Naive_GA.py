#from _typeshed import Self
from CNF_Creator import CNF_Creator
import numpy as np
import time
import timeit
import matplotlib.pyplot as plt


#---Parameters-----------------------------------------------

num_of_literals = 50    # Number of literals
pop_size = 10           # Population size of each generation
time_limit = 45
p_mutate = 0.2          # Probability of Mutation
max_stagnate_cnt = 3000  #Maximum number of epochs for which max_fitness is allowed to stagnate

#=============================================================

class CNF_Model:
    def __init__(self, num_of_literals, arr=None):
        self.num_of_literals = num_of_literals
        if (arr is None):
            self.truth_vals = np.random.randint(0,2, size=num_of_literals)
        else:
            self.truth_vals = arr
        self.fitness_score = -1

    def fitness_eval(self, cnf_statement):
        score = 0.0
        for row in cnf_statement:
            valid =  False
            for i in row:
                if ((i>0 and self.truth_vals[abs(i)-1]==1) or (i<0 and self.truth_vals[abs(i)-1]==0)):
                    valid = True
                    break
            if (valid):
                score+=1
        
        self.fitness_score = score/len(cnf_statement)*100
        return self.fitness_score
    
    def get_fitness_score(self):
        return self.fitness_score
    
    def get_truth_values(self):
        '''
            Returns Representation of the truth value of the CNF_Model Found
        '''
        result = []
        for i in range(len(self.truth_vals)):
            if (self.truth_vals[i]==1):
                result.append(i+1)
            else:
                result.append(-i-1)
        return result


class Genetic_Algorithm:

    def __init__(self, num_of_clauses, population_size = 10, num_of_literals = 50):
        self.mutate_p = p_mutate
        self.max_fitness_scores = []
        self.num_of_clauses = num_of_clauses
        self.population_size = population_size
        self.num_of_clauses = num_of_clauses
        self.num_of_literals = num_of_literals

    def init_population(self):
        population = []
        for i in range(self.population_size):
            population.append(CNF_Model(self.num_of_literals))
        return population
    
    def Weights(self, models):
        #Calculates weights for each model
        weights = np.zeros(self.population_size)
        for i in range(self.population_size):
            weights[i] = models[i].get_fitness_score()
        
        sum = weights.sum()
        weights = weights/sum
        return weights

    def reproduce(self, parent_1, parent_2):
        length = self.num_of_literals
        pivot = np.random.randint(length)
        child_arr = parent_1.truth_vals[:pivot]
        child_arr = np.append(child_arr,parent_2.truth_vals[pivot:])
        child = CNF_Model(length, child_arr)
        return child

    def Mutate(self, child):
        pivot = np.random.randint(self.num_of_literals)
        child.truth_vals[pivot] = 1-child.truth_vals[pivot]
        return
    
    def run_algorithm2(self, cnf_statement, debug_stmt = False):
        start_time = time.time()
        max_fitness = 0
        population = self.init_population()
        epoch = 1

        while(max_fitness<100.0):
            weights = self.Weights(population)
            population2 = []
            max_fitness = 0
            population = self.init_population()

            for i in range(self.population_size):
                parent1, parent2 = np.random.choice(population,2, p=weights)
                child = self.reproduce(parent1, parent2)
                child.fitness_eval(cnf_statement)

                if(np.random.random() < self.mutate_p):
                    self.Mutate(child)
                
                population2.append(child)
                max_fitness = max([max_fitness, child.get_fitness_score()])
            
            self.max_fitness_scores.append(max_fitness)

            if(epoch%1000 == 1 and debug_stmt):
                print(f"{epoch} epoch: Fitness score {max_fitness}%\n")
            population = population2.copy()
            epoch+=1

            if (time.time() - start_time > 45-0.01):
                if (debug_stmt):
                    print("\nTime limit exceeded, couldn't find a solution\n")
                break
        
        for p in population:
            if p.get_fitness_score()==max_fitness:
                return p
        return None
    
    def Max_fitness(self, population):
        '''
            Finds the fitness of the most fit model in the population
            Argument: population -> Most fit population
            Returns: max_fitness -> Max fitness value
        '''
        max_fitness = 0
        for k in population:
            max_fitness = max(max_fitness, k.get_fitness_score())
        return max_fitness

    def run_algorithm(self, cnf_statement, debug_stmt = False):
        '''
            Function that performs the Genetic Algorithm on the CNF statement
            Arguments: cnf_statement -> CNF statement whose solution has to be generated
                        debug_stmt -> If True, prints a more verbose info about the algo run
        '''
        start_time = time.time()
        max_fitness = 0
        population = self.init_population()
        epoch = 0
        time_taken = 0.0
        prev_fitness = 0.0
        stagnate_cnt = 0

        while(max_fitness<100.0):
            weights = self.Weights(population)
            population2 = []

            for i in range(self.population_size):
                parent1, parent2 = np.random.choice(population,2, p=weights)
                child = self.reproduce(parent1, parent2)
                child.fitness_eval(cnf_statement)

                if(np.random.random() < self.mutate_p):
                    self.Mutate(child)
                
                population2.append(child)
            
            population = population2.copy()
            max_fitness = self.Max_fitness(population)
            self.max_fitness_scores.append(max_fitness)
            epoch+=1

            if(epoch%1000 == 1 and debug_stmt):
                print(f"{epoch} epoch: Fitness score {max_fitness}%\n")

            if(abs(prev_fitness - max_fitness)<0.01):
                stagnate_cnt+=1
            else:
                stagnate_cnt =0
            
            prev_fitness = max_fitness
            
            time_taken = time.time() - start_time
            if (time_taken> time_limit-0.01):
                if (debug_stmt):
                    print("\nTime limit exceeded, couldn't find a solution\n")
                break
            if (stagnate_cnt==max_stagnate_cnt):
                if (debug_stmt):
                    print("\nFitness Score stagnated for too long\n")
                break
        for p in population:
            if p.get_fitness_score()==max_fitness:
                return p,time_taken
        return None,time_taken

def run_algo(cnf_statement):
    ga= Genetic_Algorithm(len(cnf_statement), pop_size, num_of_literals)
    best_child,time_taken = ga.run_algorithm(cnf_statement)
    return best_child.get_fitness_score(), best_child.get_truth_values(), time_taken

if __name__=='__main__':
    n = 50
    
    
    cnf_statement = CNF_Creator(n).ReadCNFfromCSVfile()
    num_of_clauses = len(cnf_statement)
    ga = Genetic_Algorithm(num_of_clauses,population_size=10, num_of_literals=n)
    best_child,time_taken = ga.run_algorithm(cnf_statement, True)
    if (best_child is None):
        print("Error")
    else:
        print(best_child.get_fitness_score())
        print(f"Time taken {time_taken}")
        print(f"\n CNF statement is: \n {cnf_statement} \n\n And CNF solution is \n{list(enumerate(best_child.truth_vals))}")
    
    
    '''
    Average_time = {}
    Average_accuracy = {}
    for m in range(100,140,20):

        avg_time = 0.0
        avg_acc = 0.0
        for k in range(2):
            cnf_statement = CNF_Creator(n).CreateRandomSentence(m)
            ga = Genetic_Algorithm(m,population_size=10, num_of_literals=50)
            start_time = time.time()
            best_child = ga.run_algorithm(cnf_statement, True)
            end_time = time.time() - start_time
            avg_acc += best_child.get_fitness_score()/2.0
            avg_time += end_time/2.0
        
        print(f"\n Average time taken by {m} clauses is {avg_time}s with Average accuracy {avg_acc}%\n--------------------\n")
        Average_time[m] = avg_time
        Average_accuracy[m] = avg_acc
        
    fig,axs = plt.subplots(1,2, figsize = (10,5))
    axs[0].plot(Average_time.keys(), Average_time.values())
    axs[1].plot(Average_accuracy.keys(), Average_accuracy.values())
    axs[0].set_title("Average Time taken")
    axs[1].set_title("Average Accuracy obtained")
    plt.show()
    '''
        
    




    


