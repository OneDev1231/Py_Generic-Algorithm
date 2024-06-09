#!/usr/bin/env python
# coding: utf-8

# ### MAkespan
# |

# In[ ]:


"""The algorithm starts with an empty set S and a set A of operations that need to be scheduled. 
In each iteration, it selects the operation with the earliest starting time if scheduled now, 
and assigns it to the machine that minimizes the makespan. 
The selected operation is then added to the set S and removed from the set A, 
along with any operations that are dependent on it. The algorithm continues to iterate until all
 operations have been scheduled.

In more detail:

Line 1: Initialize an empty set S.

Line 2: Initialize set A with all operations that need to be scheduled.

Line 3: For each operation oi in set A, find the earliest starting time if scheduled now.

Line 4: Select the operation ok from A with the earliest finishing time and assign it 
to the machine that minimizes the makespan.

Line 5: Set M* is the machine that is processing operation ok.

Line 6: Find all the operation that processing on machine M* and their starting
time is less than the finishing time of ok.

Line 7: Select the operation ot from B which has earliest starting time.

Line 8: Select operation o* from B that is the leftmost operation in the chromosome, 
and add it to S with the starting time st(o*).

Line 9: Remove the selected operation o* from set A and add any dependent operations to set A.

Line 10: Repeat the process until set A is empty.

Overall, the Hybrid Giffler and Thompson algorithm is used to schedule operations in a 
way that minimizes the makespan. It does this by selecting the operation 
with the earliest starting time and assigning it to the machine that minimizes the makespan. 
It then continues to iterate through the operations until all have been scheduled."""


# In[69]:


import copy
import random
import time
import sys
import plotly.figure_factory as ff
import datetime
import numpy as np
import random
from tqdm import tqdm


#this is the function to calculate Makespan
#If the chromosome representation is set as config, we willl call this function according to the Fig.4 in reverse.
def calculateMakespan(times, machines, config, n):
    time_table = []
    times = copy.deepcopy(times)
    machines = copy.deepcopy(machines)
    mn = len(machines[0])
    for i in range(mn):
        time_table.append([])

    current_times = [0]*n
    total_time = 0
    for j in config:
        job = j%n
        current_machine = machines[job].pop(0)
        current_time = current_times[job]
        machine_usage = time_table[current_machine]
        usage_time = times[job].pop(0)
        current_time, total_time = fillTimeSlot(machine_usage, current_time, usage_time, job, total_time)
        current_times[job] = current_time

    return total_time, time_table

#This is the main part of hybrid Giffler and Thompson

def fillTimeSlot(machine_usage, current_time, usage_time, job, total_time):

    #just look at 5.2 on the book
    # machine_usage = set M star in Algorithm 1

    if len(machine_usage) > 0:
        # Sort the machine usage by the start time of the scheduled job
        machine_usage.sort(key=lambda x: x[0])
        
        if current_time+usage_time <= machine_usage[0][0]:
            #if current machine schedule is new for machine
            machine_usage.append([current_time, current_time + usage_time, job])
            current_time += usage_time
            if current_time > total_time:
                total_time = current_time
            return current_time, total_time
        
        else:
            #that is while loop in Algorithm 1 
            for k in range(len(machine_usage) - 1):
                if machine_usage[k][1] >= current_time and machine_usage[k+1][0] >= (machine_usage[k][1] + usage_time):
                    machine_usage.append([machine_usage[k][1], machine_usage[k][1] + usage_time, job])
                    current_time = machine_usage[k][1] + usage_time
                    return current_time, total_time
            if machine_usage[-1][1] > current_time:
                #if there is no slot between current time and last usage of selected machine, then start work after finishing current machine usage
                current_time = machine_usage[-1][1] + usage_time
                if total_time < machine_usage[-1][1] + usage_time:
                    total_time = machine_usage[-1][1] + usage_time
                machine_usage.append([machine_usage[-1][1], machine_usage[-1][1] + usage_time, job])
                return current_time, total_time
                
            else:
                machine_usage.append([current_time, current_time + usage_time, job])
                if total_time < current_time + usage_time:
                    total_time = current_time + usage_time
                current_time += usage_time
                return current_time, total_time
    else:
        # If there are no scheduled jobs, assign the job to the current time
        machine_usage.append([current_time, current_time + usage_time, job])
        if total_time < current_time + usage_time:
            total_time = current_time + usage_time
        current_time += usage_time

    return current_time, total_time

#This is to read InputFile    
def readFilePairs(filepath):
    times_done = False
    times = []
    machines = []

    with open(filepath) as fp:
        line = fp.readline()
        n, mn = line.strip().split(' ')
        line = fp.readline()

        while line:
            parse_line = ' '.join(line.split())
            raw_line = parse_line.strip().split(' ')
            curr = []
            i = 0
            machine = []
            time = []
            while i < len(raw_line):
                m, t = raw_line[i], raw_line[i + 1]
                machine.append(int(m))
                time.append(int(t))
                i += 2

            times.append(time)
            machines.append(machine)
            line = fp.readline()

    return times, machines, int(n)

#This part is for mutation.
#Just look at Fig.12.
def swap_rnd(config):
    id1 = random.choice(range(len(config)))
    id2 = random.choice(range(len(config)))
    tmp = config[id1]
    config[id1] = config[id2]
    config[id2] = tmp
    return config


def fromPermutation(permutation, n):
    return list(map(lambda  x: x%n, permutation))
    

def printTable(table):
    i = 1
    print("TABLE: ")
    for row in table:
        print("M%s: %s" %(i, row))
        i += 1

#This part is for crossover operation
#Here the param parent is the receiver and list is undelined part in deonator in Fig.10. and Fg.11
def removeFromList(parent, list):
    seen = set()
    seen_add = seen.add
    return [x for x in parent if not (x in list or seen_add(x))]

#We can use this function to stop losing generalization for next population
def replaceWithRandomPopulation(population, q, n, mn):
    for i in range(q):
        population.pop()
    for i in range(q):
        addRandomIndividual(population, n, mn)

# def checkDiversity(population, diff, n, mn):
#     if diff < 0.7:
#         replaceWithRandomPopulation(population, int(n/5), n, mn)
#     if diff < 0.5:
#         replaceWithRandomPopulation(population, int(n/3), n, mn)
#     if diff < 0.3:
#         replaceWithRandomPopulation(population, int(n/3), n, mn)
#     if diff < 0.09:
#         replaceWithRandomPopulation(population, int(n/2), n, mn)
#     elif diff < 0.2:
#         replaceWithRandomPopulation(population, int(n/4), n, mn)

#Use this function to get the divergence between the best individual and the others in the current population
def checkDiversity(population, diff, n, mn):
    #The smaller the divergence is, the more random individuals are there.
    if diff < 0.03:
        replaceWithRandomPopulation(population, int(n/3), n, mn)
    if diff < 0.05:
        replaceWithRandomPopulation(population, int(n/5), n, mn)
    elif diff < 0.2:
        replaceWithRandomPopulation(population, int(n/10), n, mn)

def getFitness(population):
    prev = population[0][1]
    total = 0
    diffPercentage = 0.0
    for ind in population:
        curr = ind[1]
        total += curr
        diffPercentage += (curr/float(prev)) - 1
        prev = curr

    return total, diffPercentage

#Each individual is compose by a permutation(list from 0 to the job_number*machine_number)
#And a second parameter that is filled with the result of the makespan for the permutation
#We keep track of the result to not calculate multiple times the same result unnecesarily
#Is important to remove that number every time the permutation change
def addRandomIndividual(population, n, mn):
    ind = list(range(n*mn))
    random.shuffle(ind)
    population.append([ind, None])


#We generate the number of population
def generate_population(number, n, mn):
        population = []
        for i in range(number):
            addRandomIndividual(population, n, mn)
        return population

#During the crossover we select gens from the father from the start to the end index defined, we remove those from the mother
#Then we add them to the resultant in the same order that it was in the father origininally
# def crossover(father, mother, start_index, end_index):
#     father_gen = father[0][start_index:end_index]
#     fetus = removeFromList(mother[0], father_gen)
#     result = []
#     result.extend(fetus[:start_index])
#     result.extend(father_gen)
#     result.extend(fetus[start_index:])
#     return [result, None]

def crossover(father, mother, start_index, end_index, prob=0.75):
    if random.random() < prob:
        father_gen = father[0][start_index:end_index]
        fetus = removeFromList(mother[0], father_gen)
        result = []
        result.extend(fetus[:start_index])
        result.extend(father_gen)
        result.extend(fetus[start_index:])
        return [result, None]
    else:
        return mother


#mutate one member of the poupulation randomly excluding the first one(best individual)
#We just change the order of the permutation by one
def mutation(population, mutation_rate):
    if(random.random() < mutation_rate):
        candidate = random.choice(population[1:])
        swap_rnd(candidate[0])
        candidate[1] = None

def evolve(population, mutation_rate):
    #Important: the population should be sorted before evolve

    #We delete the worst individual of the population
    population.pop()

    #we choose a mother and father for the new individual
    father = random.choice(population)
    mother = random.choice(population)
    while(mother == father):
        mother = random.choice(population)
    
    indexes = range(len(father[0]))

    #we select wich part of the father will go to the mother
    start_index = random.choice(indexes)
    end_index = random.choice(indexes[start_index:])

    #we generate the baby with the crossover
    baby = crossover(father, mother, start_index, end_index)

    #we add the new member to the population
    population.append(baby)

    #we trigger the mutation for one of the population, depending on the mutation rate
    mutation(population, mutation_rate)
    return population
## PLots

def plotResult(table, maxValue):
    df = []
    mn = 0
    colors = []
    for row in table:
        mn += 1
        row.sort(key=lambda x: x[2])
        for slot in row:
            start_time=str(datetime.timedelta(seconds=slot[0]))
            end_time=str(datetime.timedelta(seconds=slot[1]))
            today = datetime.date.today()
            entry = dict(
                Task='Machine-{0}'.format(mn), 
                Start="{0} {1}".format(today, start_time), 
                Finish="{0} {1}".format(today, end_time),
                duration=slot[1] - slot[0],
                Resource='Job {0}'.format(slot[2] + 1)
                )
            df.append(entry)

            #Generate random colors
            #if(len(colors) < len(row)):
        a = min(255 - ( slot[2] * 10 ), 255)
        b = min(slot[2] * 10, 255)
        c = min(255, int(random.random() * 255))
        colors.append("rgb({0}, {1}, {2})".format(a, b, c))

    #In order to see the line ordered by integers and not by dates we need to generate the dateticks manually
    #we create 11 linespaced numbers between 0 and the maximum value
    num_tick_labels = np.linspace(start = 0, stop = maxValue, num = 11, dtype = int)
    date_ticks = ["{0} {1}".format(today, str(datetime.timedelta(seconds=int(x)))) for x in num_tick_labels]

    fig = ff.create_gantt(df,colors=colors, index_col='Resource', group_tasks=True, show_colorbar=True, showgrid_x=True, title='Job shop Schedule')
    fig.layout.xaxis.update({
        'tickvals' : date_ticks,
        'ticktext' : num_tick_labels
        })
    fig.show()


def printProgress(bestValue, iterations, timeElapsed):
    sys.stdout.write("\rIterations: {0} | Best result found {1} | Time elapsed: {2}s".format(iterations, bestValue, int(timeElapsed)))
    sys.stdout.flush()

def genetic(times, machines, n, population_number, iterations, rate, target):
    machine_number = len(machines[0])
    start_time = time.time()

    def sortAndGetBestIndividual(population):
        best_individual = None
        best_result = None
        for individual in population:
            result = None
            if not individual[1]: 
                result, table = calculateMakespan(times, machines, individual[0], n)
                individual[1] = result
            else: 
                result = individual[1]

            if not best_result or result < best_result:
                best_result = result
                best_individual = individual

        population.sort(key=lambda x: x[1])
        return best_individual, best_result

    population = generate_population(population_number, n, machine_number)
    global_best_ind, global_best = sortAndGetBestIndividual(population)
    
    ##if we don't define a target we set the number of iterations we want 
    if not target:
        for i in range(iterations):
            population = evolve(population, rate)
            best_ind, best_result = sortAndGetBestIndividual(population)
            total_fitness, diffPercentage = getFitness(population)

            if(not global_best or best_result < global_best):
                global_best = best_result
                global_best_ind = copy.deepcopy(best_ind)

            #printProgress(best_result, i, time.time() - start_time)
            checkDiversity(population, diffPercentage, n, machine_number)
    else:
        #If we define a target we iterate until the best result reach that target
        i = 0
        while(target < global_best):
            i += 1
            #in every iteration: 
            #We evolve the population
            population = evolve(population, rate)
            #We find the best individual 
            best_ind, best_result = sortAndGetBestIndividual(population)
            #We calculate the diversity % between the population and the total_fitness(sum of all the results)
            total_fitness, diffPercentage = getFitness(population)

            #if the result found is better than the global found we update the global
            if(not global_best or best_result < global_best):
                global_best = best_result
                global_best_ind = copy.deepcopy(best_ind)
            #We print the progress so far and the time elapsed
            #printProgress(best_result, i, time.time() - start_time)
            #We check the diversity, in case the diversity percentage is very low we delete a number of the population and we add randome members
            checkDiversity(population, diffPercentage, n, machine_number)

    
    best_result, best_table = calculateMakespan(times, machines, global_best_ind[0], n)           
    print("\nOVERALL RESULT")
    print("RESULT: %s" %best_result)                 
    print('the elapsed time:%ss'% (int(time.time() - start_time)))
    print("Permutation: ")
    print(fromPermutation(global_best_ind[0], n))
    printTable(best_table)
    plotResult(best_table, best_result)
    return best_result

# random.seed(42)
# target = 970
# population_size=int(input('Please input the size of population (default: 30): ') or 30)
# mutation_rate=float(input('Please input the size of Mutation Rate (default 0.2): ') or 0.2)
# iterations=int(input('Please input number of iteration (default 2000): ') or 2000)

# times, machines, n = readFilePairs("cases/mt10.txt")
# a=genetic(times, machines, n, population_size, iterations, mutation_rate, target)


# In[71]:


random.seed(42)
target = None
#population_size=int(input('Please input the size of population (default: 30): ') or 30)
# mutation_rate=float(input('Please input the size of Mutation Rate (default 0.2): ') or 0.2)
# iterations=int(input('Please input number of iteration (default 2000): ') or 2000)

population_size=50
mutation_rate=0.15
iterations=20000
times, machines, n = readFilePairs("cases/mt10.txt")

genetic(times, machines, n, population_size, iterations, mutation_rate, target)
    


# In[ ]:


# random.seed(42)
target = None

# population_size=int(input('Please input the size of population (default: 30): ') or 30)
# mutation_rate=float(input('Please input the size of Mutation Rate (default 0.2): ') or 0.2)
# iterations=int(input('Please input number of iteration (default 2000): ') or 2000)

random.seed(41542)
# population_size=int(input('Please input the size of population (default: 30): ') or 30)
# mutation_rate=float(input('Please input the size of Mutation Rate (default 0.2): ') or 0.2)
# iterations=int(input('Please input number of iteration (default 2000): ') or 2000)

population_size=42
mutation_rate=0.1
iterations=20000
times, machines, n = readFilePairs("cases/mt06.txt")
score=[]
for i in tqdm(range(30)):
    a=genetic(times, machines, n, population_size, iterations, mutation_rate, target)
    score.append(a)


# In[ ]:





# In[72]:


# random.seed(42)
target = None

# population_size=int(input('Please input the size of population (default: 30): ') or 30)
# mutation_rate=float(input('Please input the size of Mutation Rate (default 0.2): ') or 0.2)
# iterations=int(input('Please input number of iteration (default 2000): ') or 2000)

random.seed(41542)
# population_size=int(input('Please input the size of population (default: 30): ') or 30)
# mutation_rate=float(input('Please input the size of Mutation Rate (default 0.2): ') or 0.2)
# iterations=int(input('Please input number of iteration (default 2000): ') or 2000)

population_size=50
mutation_rate=0.15
iterations=20000
times, machines, n = readFilePairs("cases/mt10.txt")
score3=[]
for i in tqdm(range(30)):
    a=genetic(times, machines, n, population_size, iterations, mutation_rate, target)
    score3.append(a)


# In[74]:


import statistics

# Sample list of numbers
numbers = score3
# Calculate the mean
mean = statistics.mean(numbers)

# Calculate the minimum
minimum = min(numbers)

# Calculate the maximum
maximum = max(numbers)

# Calculate the spread (maximum - minimum)
spread = maximum - minimum

# Print the results in a table
print("Statistics for the list of numbers:")
print("Mean:", mean)
print("Minimum:", minimum)
print("Maximum:", maximum)
print("Spread:", spread)


# In[ ]:





# In[ ]:





# In[10]:





# In[11]:





# In[12]:





# In[13]:





# In[ ]:





# In[15]:





# In[ ]:




