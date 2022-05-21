from random import randint
import time
import sys
from numpy.random import choice
from Machine import Machine
from Job import Job
import math

# Constants

MAX_NUM_OF_JOBS = 100

NUM_OF_GEN = 40
NUM_OF_CHROMOZOMS = 100

file_times = (time.time() / 10000)

start_time = time.time()

debug_file = open("debugout.txt", "w")

# returns the total number of machines that will be in use , and a raw jobs data
def handleInput():
    global num_of_machines
    if input("Would you like to generate a new input file? y/n\n") == "y":
        num_of_machines = int(input("Please enter the number of machines: \n"))
        min_processing_time = int(input("Please enter the minimum processing time for a single job: \n"))
        max_processing_time = int(input("Please enter the maximum processing time for a single job: \n"))
        num_of_jobs = int(input("Please enter the number of jobs: \n"))

        print("max process time is :", max_processing_time)
        """
         Generate the soon-to-be input file
         input file format will be :

         NUMBER_OF_MACHINES
         JOB_INDEX JOB_SIZE

         notice that the total number of jobs will be indicated in the [n-1,0] cell
        """
        inpt = open("input.txt", 'w')

        inpt.write(str(num_of_machines))
        inpt.write("\n")

        # # Generate random number of jobs
        print("number of jobs generated: ", num_of_jobs)
        jobs = []
        for index in range(0, num_of_jobs):
            j = []
            j.append(index)
            #job_size = input(j)
            job_size = randint(int(min_processing_time), int(max_processing_time))
            j.append(job_size)
            inpt.write(str(index))
            inpt.write(" ")
            inpt.write(str(job_size))
            inpt.write(" ")
            inpt.write("\n")
            jobs.append(j)

        inpt.close()


    else:
        inpt = open("input.txt", 'r')
        jobs = []
        for index, line in enumerate(inpt):
            if index == 0:
                num_of_machines = int(line)
                print("The number of machines loaded : ", line, "\n")
            else:
                jobs.append(line.split())

        inpt.close()

    return num_of_machines, jobs


# handle input
num_of_machines, raw_jobs = handleInput()
num_of_jobs = len(raw_jobs)

# output file and first prints
out_file = open("output_" + str(NUM_OF_CHROMOZOMS) + "chromozoms_" + str(NUM_OF_GEN) + "generations_" + str(
    num_of_machines) + "machines_" + str(num_of_jobs) + "jobs_" + str(file_times) + ".txt", "w")

print("Number of Machines:", num_of_machines, file=out_file)
print(num_of_jobs, "jobs:", NUM_OF_CHROMOZOMS, "chromozoms", NUM_OF_GEN, "generations", file=out_file)
for job in raw_jobs:
    print(job, file=out_file)

print("---------------------------------", file=out_file)


# Creates and returns a machines list
def createMachines():
    machines = []
    for i in range(0, num_of_machines):
        cur_machine = Machine(i)
        machines.append(cur_machine)
    return machines


# Create and returns a list of jobs objects
def createJobs():
    jobs_list = []
    for job in raw_jobs:
        cur_job = Job(int(job[0]), int(job[1]))
        print("Created job: index:", cur_job.index, "processing time: ", cur_job.process_time, file=debug_file)
        jobs_list.append(cur_job)
    print("-----------------FINISHED CREATING JOB OBJECTS----------------------\n\n", file=debug_file)
    return jobs_list


# Creating objects
machines_list = createMachines()
jobs_list = createJobs()


# creating a chromosome - returning a list of size num_of_jobs , each index is job number, value is the assigned machine
def createChrom():
    ch = [0] * num_of_jobs
    for i in range(num_of_jobs):
        legal = False
        while not legal:
            machine_rand = randint(0, num_of_machines - 1)
            ch[i] = machine_rand
            machines_list[machine_rand].addJob(jobs_list[i])
            legal = True

    removeAllJobs()
    return ch

# creating a population - returning a list (of lists) of NUM_OF_CHROMOZOMS chromosomes
def createPop():
    global chrom, eval
    pop = []
    for i in range(NUM_OF_CHROMOZOMS):
        curr = []
        legal = False

        while not legal:
            chrom = createChrom()
            eval = evaluateOne(chrom)

            if eval > 0:
                legal = True

        curr.append(chrom)
        curr.append(eval)
        pop.append(curr)

    return pop


# returns current minimum makespan
def makeSpan():
    max_span = 0
    for machine in machines_list:
        if machine.span > max_span:
            max_span = machine.span

    return max_span


# removing all jobs at current state
def removeAllJobs():
    for machine in machines_list:
        cur_jobs = dict(machine.assigned_jobs)
        for key, job in cur_jobs.items():
            if key != job.index:
                print("SOMETHING WENT WRONG")
            num = job.index
            machine.removeJob(num)


# evalutation : at the moment is just the makespan of a single chromosome
def evaluateOne(chromosome: list):
    for i in range(len(chromosome)):
        machines_list[chromosome[i]].addJob(jobs_list[i])

    span = makeSpan()
    removeAllJobs()
    return span


# current fitness function = the difference between chromosome's makespan and the worst chromosome's makespan
# def updateFitness(chormosome,worst):
#    fitness = (worst-chormosome[1])+1
#
#    chormosome.append(fitness)
#    return fitness

#def updateFitness(chormosome, worst):
#    fitness = 1 / (chormosome[1])
#    chormosome.append(fitness)
#    return fitness


#using 1/squared distance
def updateFitness(chromosome,worst):
    for i in range(len(chromosome[0])):
        machines_list[chromosome[0][i]].addJob(jobs_list[i])
    machines_span = []
    makespan = chromosome[1]
    for i in range(len(machines_list)):
        machines_span.append((makespan - machines_list[i].span)**2)
    removeAllJobs()
    squared_distance = math.sqrt(sum(machines_span))
    chromosome.append(1/squared_distance)
    return 1/squared_distance


# update actual probability according to the fitness function
def updateProb(chromosome, sum):
    prob = chromosome[2] / (sum)
    chromosome.append(prob)
    return prob


# go over popluation and calculate each one's fitness
def evaluateAll(population: list):
    worst = 0
    best = sys.maxsize
    sum = 0
    probabilities = []
    for i in range(len(population)):
        eval = population[i][1]
        if eval > worst:
            worst = eval
        if eval < best:
            best = eval
    for j in range(len(population)):
        fitness = updateFitness(population[j], worst)
        sum += fitness
    for k in range(len(population)):
        prob = updateProb(population[k], sum)
        probabilities.append(prob)

    print("worst chromosome makespan:", worst, "best chromosome makespan:", best, file=out_file)
    return probabilities


# prints a given population
def printPop(population: list):
    for p in population:
        print("".join((map,p[0])),", ", p[1], ", ", p[3])

# selections of parent according to a given probabilities
def selection(probs):
    # pick 2 parents out of this distribution
    t = [i for i in range(len(probs))]
    draw = choice(t, 2, p=probs, replace=False)

    return draw


"""
 crossover operator for 2 parents , producing 2 children
 getting 2 lists, mom and dad, and slices ==-> how many slice do we want to crossover (2 slices = 1 cross point etc.)
 also returns the makespan of each child
"""


def xo(mom: list, eval_mom, dad: list, eval_dad, slices):
    global eval_daughter, eval_son
    legal = False
    legal_son = False
    legal_daughter = False
    son = []
    daughter = []
    point_track = set()
    while not legal:

        slice_point = randint(0, len(dad) - 1)
        if slice_point in point_track:
            continue
        point_track.add(slice_point)
        if len(point_track) == len(dad):  # exhausted all possible points with no success
            if legal_son is True:  # and legal_daughter is False
                return son, eval_son, mom, eval_mom
            if legal_daughter is True:  # and legal_son is False
                return dad, eval_dad, daughter, eval_daughter
        if legal_son is False:
            son = dad[:slice_point] + mom[slice_point:]
            eval_son = evaluateOne(son)
            if eval_son > -1:
                legal_son = True
        if legal_daughter is False:
            daughter = mom[:slice_point] + dad[slice_point:]
            eval_daughter = evaluateOne(daughter)
            if eval_daughter > -1:
                legal_daughter = True

        legal = legal_son and legal_daughter

    return son, eval_son, daughter, eval_daughter


# reproducing procedure -  at the moment only 1% of new generation are getting mutated
def reproduce(population: list):
    new_gen = []
    probs = []
    for p in population:
        probs.append(p[3])
    while len(new_gen) != len(probs):
        parents = selection(probs)
        son, eval_son, daughter, eval_daughter = xo(population[parents[0]][0], population[parents[0]][1],
                                                    population[parents[1]][0], population[parents[1]][1], 2)
        new_gen.append([son, eval_son])
        new_gen.append([daughter, eval_daughter])

    # mutation
    # lets say 2.5% of the population gets mutated
    how_many_to_mutate = int(NUM_OF_CHROMOZOMS * (1 / 100))
    t = [i for i in range(NUM_OF_CHROMOZOMS)]
    # choose percent of the population randomly, uniformly
    indices_to_mutate = choice(t, how_many_to_mutate, replace=False)
    for i in range(len(indices_to_mutate)):
        mutate(new_gen[indices_to_mutate[i]])

    evaluateAll(new_gen)

    return new_gen


# mutating a chromosome in N genes , at index 0 - chromosome itself, at index 1 - the makespan
def mutate(chromosome: list):

    t = [i for i in range(len(chromosome[0]))]

    # 3 this is the maximum indices to mutate
    indices_to_mutate = choice(t, 3, replace=False)

    # now needs to simulate as if the whole chromosome is assigned and check changes
    # assigning all
    for i in range(len(chromosome[0])):
        machines_list[chromosome[0][i]].addJob(jobs_list[i])

    indices_track = set()
    for i in range(len(indices_to_mutate)):

        # remove old (and good) index
        machines_list[chromosome[0][indices_to_mutate[i]]].removeJob(indices_to_mutate[i])
        legal = False
        while not legal:
            machine_rand = randint(0, num_of_machines - 1)
            # check if not already mutated this
            if machine_rand in indices_track:
                continue
            # check if another mutation is possible
            if len(indices_track) == len(chromosome):
                break

            indices_track.add(machine_rand)

            # add a new one instead
            machines_list[machine_rand].addJob(jobs_list[indices_to_mutate[i]])
            chromosome[0][indices_to_mutate[i]] = randint(0, num_of_machines - 1)
            #printChromQual(chromosome[0])
            legal = True

    span = makeSpan()
    chromosome[1] = span
    removeAllJobs()

    return span


# prints stats to file
def printMachineStatOut():
    print("---------------MACHINES STATS --------------------------\n", file=out_file)
    for machine in machines_list:
        cur_job_list = machine.retrieveJobsList()
        print("machine number ", machine.number, "assigned jobs [process_time,index]:", file=out_file)
        l = []
        for job_number, job in cur_job_list.items():
            l.append(job)
        print("".join(str(l)), file=out_file)

        print("Completion time : ", machine.span, file=out_file)
        print("\n", file=out_file)
    print("Max makespan is : ", makeSpan(), file=out_file)


# prints the chromosome's stat
def printChromQual(chromosome: list):
    sum = 0
    for i in range(len(chromosome)):
        machines_list[chromosome[i]].addJob(jobs_list[i])

        sum += jobs_list[i].process_time
        print("Makespan is :", sum)
        printMachineStatOut()
    print("Optimal solution (sum/num_of_jobs) could be :", sum / num_of_machines, file=out_file)
    print("------------------------------------------------\n", file=out_file)


# main function
def genetic():
    print("Number of jobs:", len(jobs_list), file=out_file)
    print("Number of machines:", len(machines_list), file=out_file)
    print("Number of chromosomes:", NUM_OF_CHROMOZOMS, file=out_file)
    print("Number of generations to be created:", NUM_OF_GEN, file=out_file)
    print("First population:", file=out_file)
    pop = createPop()
    #printPop(pop)
    best = 999999999999
    best_chromosome = []
    probs = evaluateAll(pop)
    for p in pop:
        #print(p, file=out_file)
        print("[", "".join(map(str, p[0])), ",", p[1], ", ", p[3], "]", file=out_file)

        if p[1] < best:
            best = p[1]
            best_chromosome = p

    print("###############")

    for i in range(NUM_OF_GEN):
        new_gg = reproduce(pop)
        pop = new_gg

        # print every 5 generations
        if (i % 5 == 0):
            print("New generation, number:", i, file=out_file)

        for p in pop:
            #if (i % 10 == 0):
            if (i == 1):
                #print(p, file=out_file)
                print("[", "".join(map(str, p[0])), ",", p[1], ", ", p[3], "]", file=out_file)
            if p[1] < best:
                best = p[1]
                best_chromosome = p
        print("###############", file=out_file)
    print("###############", file=out_file)
    # do a console output also , to get a progress track
    print("Best chromosome is :", "".join(map(str,best_chromosome[0])), "with makespan of:",
          best_chromosome[1], file=out_file)
    #print("Best chromosome is :", best_chromosome[0], "with makespan of: ", best_chromosome[1], file=out_file)
    printChromQual(best_chromosome[0])
    #printMachineStatOut()



genetic()
print("---Finished in %s seconds ---" % (time.time() - start_time),file=out_file)
debug_file.close()
out_file.close()
