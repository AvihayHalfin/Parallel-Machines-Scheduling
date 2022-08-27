import collections
from random import randint
import math
import time
import copy
from Machine import Machine
from Job import Job
# Constants
import networkx as nx

MAX_NUM_OF_JOBS = 1000
MIN_NUM_OF_JOBS = 1

debug_file = open("debugout.txt", "w")

file_times = (time.time() / 10000)


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

        # Generate random number of jobs
        print("number of jobs generated: ", num_of_jobs)
        jobs = []
        for index in range(0, num_of_jobs):
            j = []
            j.append(index)
            job_size = randint(min_processing_time, int(max_processing_time))
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


# get input and handle it
num_of_machines, raw_jobs = handleInput()
num_of_jobs = len(raw_jobs)

# output file
out_file = open(
    "output" + str(file_times) + "_" + str(num_of_machines) + "machines_" + str(
        num_of_jobs) + "jobs_" + ".txt", "w")

print("Number of Machines:", num_of_machines, file=out_file)
print(num_of_jobs, "jobs:", file=out_file)
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
        print("Created job: index:", cur_job.index, "Length:", cur_job.processing_time, file=debug_file)
        jobs_list.append(cur_job)
    print("-----------------FINISHED CREATING JOB OBJECTS----------------------\n\n", file=debug_file)
    return jobs_list


# Creating objects
machines_list = createMachines()
jobs_list = createJobs()


# removes all jobs from a state ( a list of machines )
def removeAllJobs(m_list):
    for machine in m_list:
        cur_jobs = dict(machine.assigned_jobs)
        for key, job in cur_jobs.items():
            if key != job.index:
                print("SOMETHING WENT WRONG")
            num = job.index
            machine.removeJob(num)
            print("REMOVED  -- machine#: ", machine.number, "assigned jobs: ", job)


# returns the minumum loaded machine in a given state
def findMinLoadMachine(m_list):
    prev_min_load = m_list[0].span
    min_load_index = 0
    for i in range(1, len(m_list)):
        if m_list[i].span < prev_min_load:
            prev_min_load = m_list[i].span
            min_load_index = i
    return min_load_index


# finds the minumum loaded machine in a state
def findMinLoadMachineLegaly(m_list):
    m_list_sorted = sorted(m_list, key=lambda x: x.span)
    return m_list_sorted


# The same LPT algorithm , but making sure the returned state is legal. If no legal state is possible - returns an
# empty list
def legalLpt(jobs, m_list):
    job_list_sorted_by_length = sorted(jobs, key=lambda x: x.processing_time, reverse=True)
    new_machines_list = copy.deepcopy(m_list)
    for i in range(len(job_list_sorted_by_length)):
        legal = False
        # check assignment for next min loaded machine that is legal
        for j in range(len(new_machines_list)):
            assign_to_machines = findMinLoadMachineLegaly(new_machines_list)
            new_machines_list[assign_to_machines[j].number].addJob(job_list_sorted_by_length[i])
            if new_machines_list[assign_to_machines[j].number].isLegal():
                legal = True
                break
            else:  # revert
                new_machines_list[assign_to_machines[j].number].removeJob(job_list_sorted_by_length[i].index)
        if not legal:
            return []

    return new_machines_list


# return the makespan of a give state
def makeSpan(m_list: list):
    max_span = 0
    for machine in m_list:
        if machine.span > max_span:
            max_span = machine.span
    return max_span


# assigning a new job to a state , returning a copy of the original ,so it can be reverted in case of illegal state
def simulateState(cur_job, cur_machine, cur_state):
    new_state = copy.deepcopy(cur_state)
    new_state[cur_machine].addJob(cur_job)
    return new_state

# printing the current state status
def printMachineStatOut(m_list):
    print("---------------MACHINES STATS--------------------------\n", file=out_file)
    for machine in m_list:
        cur_job_list = machine.retrieveJobsList()
        print("machine number ", machine.number, "assigned jobs [processing time,index]:", file=out_file)
        l = []
        for job_number, job in cur_job_list.items():
            l.append(job)
        print("".join(str(l)), file=out_file)

        print("Makespan : ", machine.span, file=out_file)


# check if a state is legal so far
def checkLegalState(state: list):
    for machine in state:
        if machine.isLegal() is False:
            return False
    return True

def sumOfJobsLeft(processing_time,jobs):
    sum = 0
    for job in jobs:
        if job.processing_time == processing_time:
            sum += job.processing_time
    return sum

# calculating the lower bound
def lowerBound(state, jobs):
    args = []
    args.append(avg_job)
    args.append(max_job)
    for machine in state:
        args.append(machine.span)
    machine_possibilities = findLegalPossibilities(state, jobs[0].processing_time)
    possibilities_sum = 0
    for i in machine_possibilities:
        possibilities_sum += state[i].span

    delta = sumOfJobsLeft(jobs[0].processing_time, jobs[1:])
    args.append((possibilities_sum + delta) / (len(machine_possibilities)) + 1)
    return max(args)


# gets a job type with a current state (m_list) and returns a list of machines(numbers) that can accept this job type
def findLegalPossibilities(m_list, job):
    machines = []
    for machine in m_list:
        jobs = machine.assigned_jobs
        if job in enumerate(jobs):
            machines.append(machine.number)
        elif len(jobs) != 0:
            machines.append(machine.number)
    if len(machines) == 0:
        print()
    return machines

# returns the minumum loaded LEGAL machine (by number of jobs), with the least types. If fails to find a legal machine, returns -1
def findMinJobLoadedMachine(m_list, job_process, job):
    legal_possibilities_numbers = findLegalPossibilities(m_list, job)
    if len(legal_possibilities_numbers) == 0:
        print()
    legal_machines = [m_list[i] for i in legal_possibilities_numbers]
    legal_machines_sorted_lenght = sorted(legal_machines, key=lambda x: (len(x.assigned_jobs), x.assigned_jobs[job.processing_time]),
                                               reverse=False)
    if len(legal_machines_sorted_lenght) == 0:
        return -1
    return legal_machines_sorted_lenght[0].number


def upperBoundAlg(jobs, m_list):
    # job_list_sorted_by_type_by_length = sorted(jobs, key=lambda x: (x.type,x.length), reverse=True)
    assigned_process_time = set()
    assigned_jobs_indices = []
    new_machines_list = copy.deepcopy(m_list)
    new_jobs = copy.deepcopy(jobs)

    # check which type already have a machine yet
    time_check = set()
    for i in range(len(new_machines_list)):
        for process_time in new_machines_list[i].assigned_jobs:
            time_check.add(process_time)

    if len(time_check) < len(jobs_list):
        # need to add the missing types so each type will have at least one machine
        for i in range(len(new_jobs)):
            if len(assigned_process_time) == len(jobs_list) - len(time_check):
                break
            assigned = False
            if new_jobs[i].processing_time in assigned_process_time or new_jobs[i].processing_time in time_check:
                continue
            else:  # first time seen this type
                for j in range(len(new_machines_list)):
                    new_machines_list[j].addJob(new_jobs[i])
                    if new_machines_list[j].isLegal():
                        assigned_process_time.add(new_jobs[i].processing_time)
                        assigned_jobs_indices.append(i)
                        assigned = True
                    else:
                        # revert
                        new_machines_list[j].removeJob(new_jobs[i].index)
                    if assigned:
                        break

    # fix the job list that left after first assign
    for i in sorted(assigned_jobs_indices, reverse=True):
        del new_jobs[i]

    for i in range(len(new_jobs)):
        legal = False
        job_process = new_jobs[i].processing_time
        # check assignment for next min loaded machine that is legal
        for j in range(len(new_machines_list)):
            assign_to_machine = findMinJobLoadedMachine(new_machines_list, job_process, new_jobs[i])
            if assign_to_machine == -1:  # meaning there's no legal machine to assign to
                return []
            new_machines_list[assign_to_machine].addJob(new_jobs[i])
            if new_machines_list[assign_to_machine].isLegal():
                legal = True
                break
            else:  # revert
                new_machines_list[assign_to_machine].removeJob(new_jobs[i].index)
        if not legal:
            return []

    return new_machines_list


# main branch and bound function
def bnb(state, jobs):
    global best_state, best_state_makespan, level_count
    if len(jobs) == 0:
        return

    # track levels
    level_count[jobs[0].index] += 1

    if best_state_makespan == math.ceil(avg_job):
        return

    for i in range(len(machines_list)):
        new_state = simulateState(jobs[0], i, state)
        is_legal_state = checkLegalState(new_state)
        lower_bound = lowerBound(new_state, jobs)

        if is_legal_state is True:
            # remember that doing lpt is just for upper bound calculation , so there might be no need in getting the after_lpt
            # print("now doing lpt for the rest", file=out_file)
            after_lpt = legalLpt(jobs[1:], new_state)
            # print("legal state,after lpt:", checkLegalState(after_lpt), file=out_file)
            if len(after_lpt) == 0:
                # meaning legalLPT has failed - need the other algorithm
                after_lpt = upperBoundAlg(jobs[1:], new_state)
                if len(after_lpt) == 0:  # upperBoundAlg can't find legal bound
                    upper_bound = 9999999999
                else:  # upperBoundAlg succeeded
                    upper_bound = makeSpan(after_lpt)

            else:  # lpt succeeded
                upper_bound = makeSpan(after_lpt)

            if lower_bound == upper_bound:
                if best_state_makespan > upper_bound:
                    best_state_makespan = upper_bound
                    # print only if there's new best solution
                    printMachineStatOut(after_lpt)
                    best_state = after_lpt
            else:
                if lower_bound < upper_bound and lower_bound < best_state_makespan:
                    bnb(new_state, jobs[1:])


# do a kind of sort/reordring of the jobs to make sure that the all the types has representatives - OPTIONAL
def initialSort():
    numbers = set()
    first_in_line = {}
    for index, job in enumerate(jobs_list):
        if job.processing_time not in numbers:
            numbers.add(job.index)
            first_in_line[index] = job

    for i, job in collections.OrderedDict(sorted(first_in_line.items())).items():
        del jobs_list[i]
        jobs_list.insert(0, job)

    # update jobs numbers
    for i in range(len(jobs_list)):
        jobs_list[i].index = i


# If initial sort is wanted , uncomment the next line
initialSort()


max_job = max(x.processing_time for x in jobs_list)
avg_job = sum(x.processing_time for x in jobs_list) / num_of_machines

best_state = legalLpt(jobs_list, machines_list)
if len(best_state) != 0:
    best_state_makespan = makeSpan(best_state)
else:
    best_state = upperBoundAlg(jobs_list, machines_list)
    best_state_makespan = makeSpan(best_state)

level_count = [0] * num_of_jobs
start_time = time.time()
bnb(machines_list, jobs_list)

print("***************************************************", file=out_file)
print("***************************************************", file=out_file)
print("BEST STATE IS", file=out_file)
printMachineStatOut(best_state)
print("***************************************************", file=out_file)
print("---Finished in %s seconds ---" % (time.time() - start_time), file=out_file)
print("***************************************************", file=out_file)
print("***************************************************", file=out_file)
for i in range(len(level_count)):
    print("Number of nodes in level", i, ":", level_count[i], file=out_file)

out_file.close()

gf = nx.erdos_renyi_graph(100)
print(gf)