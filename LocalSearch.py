import copy
from random import *
import time
import math
from Job import Job
from Machine import Machine
import numpy as np


# Constants

MAX_NUM_OF_JOBS = 1000000
MIN_NUM_OF_JOBS = 1

file_times = (time.time() / 100000)
start_time = time.time()

debug_file = open("debugout.txt", "w")

final = open("Final Scheduling.txt",'w')

jobs = [None]


# returns the total number of machines that will be in use , and a raw jobs data
def handleInput():
    if input("Would you like to generate a new input file? Y/N\n") == "Y":
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


num_of_machines, raw_jobs = handleInput()
num_of_jobs = len(raw_jobs)

# output file and first prints
out_file = open("output_" + str(num_of_machines) + "machines_" + str(num_of_jobs) + "jobs_" + str(file_times) + ".txt",
                "w")

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
        print("Created job: index:", cur_job.index, "processing_time:", cur_job.processing_time, file=debug_file)
        jobs_list.append(cur_job)
    print("-----------------FINISHED CREATING JOB OBJECTS----------------------\n\n", file=debug_file)
    return jobs_list


machines_list = createMachines()
jobs_list = createJobs()


# initialization - every job at the first machine
def initialAssign():
    for j in jobs_list:
        machines_list[0].add_job(j)


# returns the makespan
def finalMakeSpan():
    max_span = 0
    for machine in machines_list:
        if machine.span > max_span:
            max_span = machine.span
    return max_span


# Print machines' stats
def printMachineStat():
    print("---------------MACHINES STATS--------------------------\n", file=debug_file)
    for machine in machines_list:
        cur_job_list = machine.retrieve_jobs_list()
        print("machine # ", machine.number, "assigned jobs #:", file=debug_file)
        l = []
        for job in cur_job_list:
            l.append(job)
        print("".join(str(l)), file=debug_file)

        print("Makespan : ", machine.span, file=debug_file)
    print("Max makespan is : ", finalMakeSpan(), file=debug_file)
    print("------------------------------------------------\n", file=debug_file)


# Print machines' stats to file
def printMachineStatOut(action):
    print("---------------MACHINES STATS # %s %s--------------------------\n" % (
        printMachineStatOut.out_stat_counter, action), file=out_file)
    for machine in machines_list:
        cur_job_list = machine.retrieve_jobs_list()
        print("machine number ", machine.number, "assigned jobs [processing_time,index]:", file=out_file)
        l = []
        for job_number, job in cur_job_list.items():
            l.append(job)
        print("".join(str(l)), file=out_file)

        print("Makespan : ", machine.span, file=out_file)
        print("\n", file=out_file)
    print("Max makespan is : ", finalMakeSpan(), file=out_file)
    print("------------------------------------------------\n", file=out_file)
    printMachineStatOut.out_stat_counter = printMachineStatOut.out_stat_counter + 1

printMachineStatOut.out_stat_counter = 0

def printFinalMachineStatOut(action):
    print("---------------MACHINES STATS # %s %s--------------------------\n" % (
        printMachineStatOut.out_stat_counter, action), file=final)
    for machine in machines_list:
        cur_job_list = machine.retrieve_jobs_list()
        print("machine number ", machine.number, "assigned jobs [processing_time,index]:", file=final)
        l = []
        for job_number, job in cur_job_list.items():
            l.append(job)
        print("".join(str(l)), file=final)

        print("Makespan : ", machine.span, file=final)
        print("\n", file=final)
    print("Max makespan is : ", finalMakeSpan(), file=final)
    print("------------------------------------------------\n", file=final)
    printMachineStatOut.out_stat_counter = printMachineStatOut.out_stat_counter + 1


printMachineStatOut.out_stat_counter = 0

# Print the stats of the machines
def printMachineStatConsole():
    print("---------------MACHINES STATS--------------------------\n")
    for machine in machines_list:
        cur_job_list = machine.retrieve_jobs_list()
        print("machine # ", machine.number, "assigned jobs #:")
        l = []
        for job in cur_job_list:
            l.append(job)
        print("".join(str(l)))

        print("Makespan : ", machine.span)
    print("Max makespan is : ", finalMakeSpan())
    print("------------------------------------------------\n")


def removeAllJobs():
    for machine in machines_list:
        cur_jobs = dict(machine.assigned_jobs)
        for key, job in cur_jobs.items():
            if key != job.index:
                print("SOMETHING WENT WRONG")
            num = job.index
            machine.remove_job(num)
            print("REMOVED  -- machine#: ", machine.number, "assigned jobs: ", job)

    print("---------------MACHINES' REMAINING JOB LISTS-----------------------\n")

    for machine in machines_list:
        cur_jobs = dict(machine.assigned_jobs)
        for key, job in cur_jobs.items():
            if key != job.index:
                print("SOMETHING WENT WRONG")
            num = job.index
            print("LEFT  -- machine#: ", machine.number, "assigned jobs: ", job)


# A method for moving a job
# parameters: origin machine , a single job to move , a target machine
# returns : True if successful , else False

def moveJob(origin_machine: Machine, target_machine: Machine, job_to_move: Job):
    if target_machine.span + job_to_move < origin_machine.span:  # move job if it is useful
        cur_job = origin_machine.retrieve_job(job_to_move)
        origin_machine.remove_job(job_to_move)
        target_machine.add_job(cur_job)
        return True
    else:
        return False


# Swap between 2 jobs from origin to target
def swapJobs(origin_machine: Machine, target_machine: Machine, origin_job, target_job):
    if target_machine.span <= origin_machine.span:  # swap two jobs even if the machines have the same span
        temp = origin_machine.retrieve_job(origin_job)
        origin_machine.remove_job(origin_job)
        target_machine.add_job(temp)
        temp = target_machine.retrieve_job(target_job)
        target_machine.remove_job(target_job)
        origin_machine.add_job(temp)
        return True
    else:
        return False


# Check if we should do the swap
def checkSwapSpan(origin_machine: Machine, target_machine: Machine, origin_job, target_job):
    cur_span = finalMakeSpan()
    origin_span = origin_machine.span
    target_span = target_machine.span
    local_max_span = max(origin_span, target_span)
    origin_job_span = jobs_list[origin_job].processing_time
    target_job_span = jobs_list[target_job].processing_time
    new_origin_span = origin_span - origin_job_span + target_job_span
    new_target_span = target_span - target_job_span + origin_job_span
    new_local_max_span = max(new_origin_span,new_target_span)  # find the max between the machines after the swap
    if new_local_max_span < cur_span:  # by swapping the jobs we won't exceed the current makespan
        if new_local_max_span < local_max_span:
            return True
        else:
            return False
    else:
        return False


# Check if a move is at least as good as current state .
def checkMoveSpan(origin_machine: Machine, target_machine: Machine, job_to_move : Job):
    cur_span = finalMakeSpan()
    origin_span = origin_machine.span
    target_span = target_machine.span
    local_max_span = max(origin_span, target_span)
    job_process_time = jobs_list[job_to_move].processing_time
    new_local_max_span = max(origin_span - job_process_time, target_span + job_process_time)
    if cur_span == target_span:
        #print(cur_span)
        return False  # assuming job processing_time is at least 1 , it won't be good to move to that machine, which is already at max span
    elif cur_span > target_span + job_process_time:  # by moving the job we won't exceed the current max span
        if new_local_max_span <= local_max_span:  # if still making an improvement
            return True
        else:
            return False
    else:
        return False

def uniquePairs(source):
    result = []
    for p1 in range(len(source)):
        for p2 in range(p1 + 1, len(source)):
            result.append([source[p1], source[p2]])
    return result

# Check if a certain 2-1 swap is legal
def isLegalTwoSwap(origin_machine: Machine, target_machine: Machine, pair1: list, pair2: list):
    if pair1[0] not in origin_machine.assigned_jobs:
        print("d here")
    try:
        first = origin_machine.assigned_jobs[pair1[0]]
        second = origin_machine.assigned_jobs[pair1[1]]
        third = target_machine.assigned_jobs[pair2[0]]

        origin_count = origin_machine.span
        target_count = target_machine.span

        max_count = max(origin_count, target_count)

        new_origin_count = origin_count + third.processing_time - first.processing_time - second.processing_time
        new_target_count = target_count + first.processing_time + second.processing_time - third.processing_time

        new_max_count = max(new_origin_count, new_target_count)

        if new_max_count < max_count:  # if the new maximum is smaller
            return True
        elif new_origin_count > origin_count:
            if new_target_count < target_count:
               return True
        else:
           return False
        #else:
        #    return False
    except Exception as error:
        print('Caught this error: ' + repr(error))


# Check if we should do 2-1 swap
def checkTwoSwapSpan(origin_machine: Machine, target_machine: Machine, pair1: list, pair2: list):
    first = (origin_machine.assigned_jobs[pair1[0]])
    second = (origin_machine.assigned_jobs[pair1[1]])
    third = (target_machine.assigned_jobs[pair2[0]])

    cur_span = finalMakeSpan()
    origin_span = origin_machine.span
    target_span = target_machine.span
    local_max_span = max(origin_span, target_span)

    new_origin_span = origin_span + third.processing_time - first.processing_time - second.processing_time
    new_target_span = target_span + first.processing_time + second.processing_time - third.processing_time

    new_local_max_span = max(new_origin_span, new_target_span)

    if new_local_max_span < cur_span:  # by swapping the jobs we won't exceed the current makespan
        #print(cur_span)
        if new_local_max_span < local_max_span:
            print(new_local_max_span)
            return True
        elif origin_span < new_origin_span:
            if target_span > new_target_span:
                return True
        else:
            return False
    else:
        return False


# Swap 2-1 jobs between 2 machines
def swapTwoJobs(origin_machine: Machine, target_machine: Machine, pair1: list, pair2: list):
    first_move = swapJobs(origin_machine, target_machine, pair1[0], pair2[0])
    second_move = moveJob(origin_machine, target_machine, pair1[1])

    if first_move and second_move:
        return True
    else:
        return False


# if all done return True, else return False
def isDone(d_list):
    return all(item is False for item in d_list)


def oneJobRoutine():
    done = False
    while not done:
        prev_makespan = finalMakeSpan()

        done_list = [
                        False] * num_of_machines  # for checking if at least one job has moved in the last machine iteration
        for number, machine in enumerate(machines_list):
            for job_number, job in machine.assigned_jobs.copy().items():
                for i in range(1, num_of_machines):
                        move_or_not_to_move = checkMoveSpan(machine,
                                                            machines_list[(machine.number + i) % num_of_machines],
                                                            job_number)
                        if move_or_not_to_move is True:
                            moved = moveJob(machine, machines_list[(machine.number + i) % num_of_machines], job_number)
                            target_machine = machines_list[(machine.number + i) % num_of_machines]
                            if moved is True:
                                print("Moved jobs : ", job_number, "from machine: ", machine.number, "and job ",
                                      job_number, " from machine "
                                      , target_machine.number, file=out_file)
                                print("Moved jobs : ", job_number, "from machine: ", machine.number, "and job ",
                                      job_number, " from machine "
                                      , target_machine.number, file=debug_file)
                                if done_list[machine.number] is False:
                                    done_list[machine.number] = True
                            break

            if num_of_jobs <= 500:
                printMachineStatOut("Moving one job")
            if prev_makespan > finalMakeSpan():
                print("makespan: ", finalMakeSpan(), file=out_file)
                prev_makespan = finalMakeSpan()

            if isDone(done_list):
                done = True
                break


def oneByOneSwapRoutine():
    done = False
    while not done:
        prev_makespan = finalMakeSpan()
        no_swap_count = len(jobs_list)
        done_list = [
                        False] * num_of_machines  # for checking if at least one job has moved in the last machine iteration
        for number, machine in enumerate(machines_list):  # origin machine
            for job_number, job in machine.assigned_jobs.copy().items():  # origin job
                move_at_least_once = False
                break_flag = False
                for i in range(1, num_of_machines):
                    target_machine = machines_list[(machine.number + i) % num_of_machines]
                    for target_job_number, target_job in target_machine.assigned_jobs.copy().items():
                        moved = False

                        move_or_not_to_move = checkSwapSpan(machine,
                                                                target_machine,
                                                                job_number, target_job_number)

                        if move_or_not_to_move is True:
                                moved = swapJobs(machine, target_machine, job_number, target_job_number)

                                move_at_least_once = True
                                if moved is True:
                                    print("Swapped jobs : ", job_number, "from machine: ", machine.number, "and job ",
                                          target_job_number, " from machine "
                                          , target_machine.number, file=debug_file)
                                    print("Swapped jobs : ", job_number, "from machine: ", machine.number, "and job ",
                                          target_job_number, " from machine "
                                          , target_machine.number, file=out_file)
                                    break_flag = True
                                    break
                    if break_flag is True:
                        break

                if move_at_least_once is False:
                    no_swap_count = no_swap_count - 1

            if num_of_jobs <= 500000:
                printMachineStatOut("Swapping jobs 1 by 1 with 2 machine")
                print("makespan: ", finalMakeSpan(), file=out_file)
                prev_makespan = finalMakeSpan()

            if prev_makespan > finalMakeSpan():
                print("makespan: ", finalMakeSpan(), file=out_file)
                prev_makespan = finalMakeSpan()

        if no_swap_count == 0:
            done = True
            break


def twoRoutineHelper(machine: Machine):
    origin_pairs = uniquePairs(list((machine.assigned_jobs.copy().keys())))

    for pair1 in origin_pairs:

        for i in range(1, num_of_machines):
            target_machine = machines_list[(machine.number + i) % num_of_machines]

            target_pairs = uniquePairs(list(target_machine.assigned_jobs.copy().keys()))

            for pair2 in target_pairs:

                if isLegalTwoSwap(machine, target_machine, pair1,
                                  pair2):  # check if origin machine can accept target job and if target machine can accept origin job

                    move_or_not_to_move = checkTwoSwapSpan(machine, target_machine, pair1, pair2)

                    if move_or_not_to_move is True:

                        swapped = swapTwoJobs(machine, target_machine, pair1, pair2)

                        if swapped is True:
                            print("Swapped jobs are: job ", pair1[0], "and job ", pair1[1], "from machine number ",
                                  machine.number, " with job number ",
                                  pair2[0], "from machine number ", target_machine.number, file=out_file)
                            print("Swapped jobs are: job ", pair1[0], "and job ", pair1[1], "from machine number ",
                                  machine.number, " with job number ",
                                  pair2[0], "from machine number ", target_machine.number, file=debug_file)
                            swapped_at_least_once = True
                            return True

    return False


def twoByOneSwapRoutine():
    done = False
    machine_one_counter = 0

    while not done:

        prev_makespan = finalMakeSpan()
        done_list = [False] * num_of_machines
        # iterate over the machine - 1st machine is passed only if all the jobs in this machine cant be swapped
        for number, machine in enumerate(machines_list):  # 1st machine
            swapped_at_least_once = False
            if machine.number == 0:
                machine_one_counter += 1
            print("hii")

            # generate all unique jobs pairs in the machine
            swapped = True
            print("im in machine", machine.number, "final makespan= ", finalMakeSpan(), file=debug_file)

            while swapped is True:
                swapped = twoRoutineHelper(machine)

            if num_of_jobs <= 500:
                printMachineStatOut("Swapping jobs 2 by 1 with 2 machine")
            if prev_makespan > finalMakeSpan():
                print("makespan: ", finalMakeSpan(), file=out_file)
                prev_makespan = finalMakeSpan()

        if isDone(done_list):
            done = True
            break

        if machine_one_counter == 2:
            return

# finds the minumum loaded machine in a state
def findMinLoadMachineLegaly(m_list):
    m_list_sorted = sorted(m_list, key=lambda x: x.span)
    return m_list_sorted

# the LPT algorithm , but making sure the returned state is okey. If no legal state is possible - returns an empty list
def Lpt(jobs,m_list):
    job_list_sorted_by_length = sorted(jobs, key=lambda x: x.processing_time, reverse=True)
    new_machines_list = copy.deepcopy(m_list)
    for i in range(len(job_list_sorted_by_length)):
        legal = False
        # check assignment for next min loaded machine that is legal
        for j in range(len(new_machines_list)):
            assign_to_machines = findMinLoadMachineLegaly(new_machines_list)
            new_machines_list[assign_to_machines[j].number].add_job(job_list_sorted_by_length[i])
            if new_machines_list[assign_to_machines[j].number].get_number_of_jobs() != 0:
                legal = True
                break
            else:   # revert
                new_machines_list[assign_to_machines[j].number].removeJob(job_list_sorted_by_length[i].index)
        if not legal:
            return []

    return new_machines_list


# Main routine
def localSearch():
    printMachineStatOut("Initial state")

    prev = finalMakeSpan()
    done = False
    while not done:
        done_list = [None] * num_of_machines
        oneJobRoutine()
        oneByOneSwapRoutine()
        twoByOneSwapRoutine()
        oneJobRoutine()
        done = isDone(done_list)

        if done is True:
            break
        if finalMakeSpan() < prev:
            prev = finalMakeSpan()
        else:
            break


sum_of_jobs = sum(x.processing_time for x in jobs_list)
avg_job = sum_of_jobs / num_of_machines

if num_of_jobs > 100:
    machines_list = Lpt(jobs_list, machines_list)
else:
    initialAssign()

printMachineStat()
localSearch()


printMachineStatOut("Final state")

print("-------------- %s seconds ---------------------" % (time.time() - start_time), file=out_file)

final.close()
debug_file.close()
out_file.close()

