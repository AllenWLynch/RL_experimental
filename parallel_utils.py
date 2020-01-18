import threading
import queue

#this parallel actor will sit on one process/CPU core, but will thread multiple agents, is assisted in exploration and diversification with noisy policy
class ParallelActor():

    def __init__(self, state_manager, memory_addition_function, buffer_size = 64):
        
        self.manager = state_manager
        self.buffer_size = 64
        self.buffer = queue.Queue()
        self.memory_addition_function = memory_addition_function

    def process_batch(self):
        if not self.buffer.empty():
            next_batch = self.buffer.get()
            td_errors = self.manager.agent.estimate_batch_priorities(next_batch)
            self.memory_addition_function(next_batch, td_errors)
            #this is where it would get sent to memeory

    def start(self):

        iterator = iter(self.manager.run())
        while True:
            batch = [
                next(iterator) for j in range(self.buffer_size)
            ]
            self.buffer.put(batch)
            try:
                t1 = threading.Thread(target = self.process_batch)
                t1.start()
            except Exception as err:
                raise err








