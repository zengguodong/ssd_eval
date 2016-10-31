import time

class Timer(object):
    def __init__(self, topic=''):
        self.start_time = None
        self.end_time = None
        self.topic = topic

    def set_topic(self, topic=''):
        self.topic = topic

    def tic(self):
        self.start_time = time.time()

    def toc(self):
        self.end_time = time.time()
        print '{} cost time: {}'.format(self.topic, self.end_time - self.start_time)
