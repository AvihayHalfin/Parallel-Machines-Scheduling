

class Job(object):
    def __init__(self, ind, process_time):
        self.index = ind
        self.processing_time = process_time
        self.in_machine = -1

    def __iter__(self):
        return iter(self)

    def __str__(self):
        return "[%s, %s]" % (self.index, self.processing_time)

    def __repr__(self):
        return "[%s, %s]" % (self.index, self.processing_time)

    def __len__(self):
        return self.processing_time

    def __eq__(self, other):
        if self.index != other.index:
            return False
        else:
            return True

    def get_index(self):
        return self.index

    def get_processing_time(self):
        return self.processing_time
