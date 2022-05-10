
class Machine(object):
    def __init__(self, num):
        self.assigned_jobs = {}
        self.number = num  # Machine serial #
        self.span = 0  # Initial makespan

    def __str__(self):
        ret = ""
        for key, val in self.assigned_jobs.items():
            ret.join(val.getIndex()).join(", ")
        return "Jobs indices : %s" % ret

    def __repr__(self):
        ret = ""
        for a in self.assigned_jobs:
            ret.join(a.getIndex()).join(", ")
        return "Jobs indices : %s" % ret

    def __iter__(self):
        return iter(self)

    def retrieve_jobs_list(self):
        return self.assigned_jobs

    def add_job(self, job):
        self.assigned_jobs[job.get_index()] = job
        self.span += job.get_processing_time()
        job.in_machine = self.number

    def retrieve_job(self, job_number):
        return self.assigned_jobs[job_number]

    # removing job from the machine by job number
    def remove_job(self, job_number):
        job = self.retrieve_job(job_number)
        del (self.assigned_jobs[job_number])
        self.span -= job.get_processing_time()
        job.in_machine = -1

    # Check how many jobs do I have
    def get_number_of_jobs(self):
        return len(self.assigned_jobs)
