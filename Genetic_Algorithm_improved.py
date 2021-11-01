from CNF_Creator import *
import numpy as np
import time
import timeit

#---Parameters-----------------------------------------------

num_of_literals = 50    # Number of literals
pop_size = 10           # Population size of each generation
time_limit = 45
p_mutate = 0.9          # Probability of Mutation
p_mutate_literal = 0.1  # Probability of Mutating each literal
p_tournament_sel = 0.9  # Initial probability that random child is chosen over a fit child
beta = 100.0              #Parameter in exponential decay
max_stagnate_cnt = 3000  #Maximum number of epochs for which max_fitness is allowed to stagnate

#=============================================================

class CNF_Model:
    '''
        Implements a single instance of Valuation that maps literals to True/False
    '''

    def __init__(self, num_of_literals, arr=None):
        '''
            Initializes the CNF model
            Arguments: num_of_literals -> Number of literals in the model
                       arr -> Inital values of the Valuation
                                if None: randomly initialized
        '''

        self.num_of_literals = num_of_literals
        if (arr is None):
            self.truth_vals = np.random.randint(0,2, size=num_of_literals)
        else:
            self.truth_vals = arr
        self.fitness_score = -1

    def fitness_eval(self, cnf_statement):
        '''
            Calculates the fitness score of the current valuation over a CNF_Statement
            Arguments: cnf_statement -> CNF statement over which fitness is evaluated
            Return: fitness_score -> Calculated Fitness score
        '''
        score = 0.0
        for row in cnf_statement:
            valid =  False
            for i in row:
                if ((i>0 and self.truth_vals[abs(i)-1]==1) or (i<0 and self.truth_vals[abs(i)-1]==0)):
                    valid = True
                    break
            if (valid):
                score+=1.0
        
        self.fitness_score = float(score)/float(len(cnf_statement))*100.0
        return self.fitness_score
    
    def get_fitness_score(self):
        '''
            Returns the last calculated fitness score of the model 
        '''
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
    '''
        Class that implements the Genetic Algorithm
    '''

    def __init__(self, num_of_clauses, population_size = 10, num_of_literals = 50):
        '''
            Initializes the algorithm parameters
            Arguments: num_of_clauses -> Number of clauses in the CNF statement
                       population_size -> Population size of each generation of models
                       num_of_literals -> Number of literals used in the CNF Statement 
        '''
        self.mutate_p = p_mutate
        self.max_fitness_scores = []
        self.num_of_clauses = num_of_clauses
        self.population_size = population_size
        self.num_of_clauses = num_of_clauses
        self.num_of_literals = num_of_literals

    def init_population(self, cnf_statement):
        '''
            Creates intial population of CNF Models that is used by the algorithm
            Arguments: cnf_statement -> CNF Statement beine evaluated by the class
            Returns: population -> Population of CNF Models
        '''
        population = []
        for i in range(self.population_size):
            population.append(CNF_Model(self.num_of_literals))
        for i in range(self.population_size):
            population[i].fitness_eval(cnf_statement)
        return population
    
    def Weights(self, models):
        '''
            Assigns a weight to each CNF model in the population that represent
            it's preference to be selected for reproduction by using fitness scores
            Arguments: models -> population of models
            Returns: weights -> An array of weights of models s
        '''
        weights = np.zeros(self.population_size)
        for i in range(self.population_size):
            weights[i] = models[i].get_fitness_score()
        
        sum = weights.sum()
        weights = weights/sum
        return weights

    def reproduce(self, parent_1, parent_2):
        '''
            Function to perform the Reproduction task by performing Crossover
            over a random pivot
            Arguments: parent_1, parent_2 -> parent models
            Returns:   child -> Child model
        '''
        length = self.num_of_literals
        pivot = np.random.randint(length)
        child_arr = parent_1.truth_vals[:pivot]
        child_arr = np.append(child_arr,parent_2.truth_vals[pivot:])
        child = CNF_Model(length, child_arr)
        return child

    def Mutate(self, child):
        '''
            Performs Mutation task
            Arguments: child -> CNF model to be mutated
        '''
        for i in range(self.num_of_literals):
            if (np.random.random() < p_mutate_literal):
                child.truth_vals[i] = 1-child.truth_vals[i]
        return

    def Tournament_Selection(self, population, pop_size, epoch):
        '''
            Performs Tournament Selection of the best fit models with a
            probability that increases with epoch
            Argument: population-> Population in which the best fit are chosen
                     pop_size-> Size of the final population after best fit
                     epoch -> Current epoch
            Returns: population2 -> Population generated after selection
        '''
        population2 = []
        population.sort(key = lambda x: x.fitness_score, reverse = True)
        p = float(p_tournament_sel)**(epoch/beta)

        for i in range(0,pop_size):
            if (np.random.random() > p ):
                population2.append(population[i])
            else:
                population2.append(population[np.random.randint(pop_size,len(population))])

        return population2

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
        population = self.init_population(cnf_statement)
        epoch = 0
        time_taken = 0.0
        prev_fitness = 0.0
        stagnate_cnt = 0

        while(max_fitness<100.0):
            weights = self.Weights(population)
            population2 = population.copy()

            for i in range(self.population_size):
                parent1, parent2 = np.random.choice(population,2, p=weights)
                child = self.reproduce(parent1, parent2)
                child.fitness_eval(cnf_statement)

                if(np.random.random() < self.mutate_p):
                    self.Mutate(child)
                
                population2.append(child)
            
            population = self.Tournament_Selection(population2, pop_size, epoch)
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



def main():
    cnfC = CNF_Creator(n=50) # n is number of symbols in the 3-CNF sentence
    #sentence = cnfC.CreateRandomSentence(m=120) # m is number of clauses in the 3-CNF sentence
    #print('Random sentence : ',sentence)

    sentence = cnfC.ReadCNFfromCSVfile()
    #print('\nSentence from CSV file : ',sentence)

    ga = Genetic_Algorithm(len(sentence))
    best_model,time_taken = ga.run_algorithm(sentence)

    print('\n\n')
    print('Roll No : 2019A7PS0033G')
    print('Number of clauses in CSV file : ',len(sentence))
    print('Best model : ', best_model.get_truth_values())
    print(f'Fitness value of best model : {best_model.get_fitness_score()}%')
    print(f'Time taken : {time_taken}')
    print('\n\n')
    
if __name__=='__main__':
    main()