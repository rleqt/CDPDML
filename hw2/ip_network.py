from network import *
import multiprocessing
from preprocessor import Worker
import my_queue
from os import environ

class IPNeuralNetwork(NeuralNetwork):

    def fit(self, training_data, validation_data=None):
        '''
        Override this function to create and destroy workers
        '''
        # 1. Create Workers
        # (Call Worker() with self.mini_batch_size as the batch_size)
        jobs_queue = multiprocessing.JoinableQueue()
        result_queue = my_queue.MyQueue()
        self.jobs = jobs_queue
        self.result = result_queue
        workers = min(int(environ['SLURM_CPUS_PER_TASK'])*2, self.number_of_batches)
        for i in range(workers):
            tmp = Worker(jobs_queue, result_queue, training_data, self.mini_batch_size)
            tmp.start()
        # 2. Set jobs
        # Call the parent's fit. Notice how create_batches is called inside super.fit().
        super().fit(training_data, validation_data)
        # 3. Stop Workers
        for iteration in range(workers):
            jobs_queue.put(None)
        self.jobs.join()

    def create_batches(self, data, labels, batch_size):
        '''
        Override this function to return self.number_of_batches batches created by workers
		Hint: you can either generate (i.e sample randomly from the training data) the image batches here OR in Worker.run()
        '''
        for i in range(self.number_of_batches):
            pair = ([], [])
            for iterations in range(batch_size):
                datum = random.randint(0, data.shape[0]-1)
                pair[0].append(data[datum])
                pair[1].append(labels[datum])
            self.jobs.put(pair)

        return [self.result.get() for i in range(self.number_of_batches)]
