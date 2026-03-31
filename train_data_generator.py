class TrainDataGenerator(object):
    def __init__(self, gen_f, gen_s):
        self.gen_f = gen_f
        self.gen_s = gen_s
        
        # Combine the total batch count so progress bars/loops work correctly
        self._nb_total_batches = gen_f.get_total_batches_in_data() + gen_s.get_total_batches_in_data()

    def get_data_sizes(self):
        # Returns the size from the first generator (assuming both datasets use the same shape)
        return self.gen_f.get_data_sizes()

    def get_total_batches_in_data(self):
        return self._nb_total_batches

    def get_nb_classes(self):
        return self.gen_f.get_nb_classes()

    def generate(self):
        # Iterate through the first dataset
        for batch in self.gen_f.generate():
            yield batch
        
        # Iterate through the second dataset
        for batch in self.gen_s.generate():
            yield batch

    # Forward other common SELD-net methods to the underlying generator
    def nb_frames_1s(self):
        return self.gen_f.nb_frames_1s()

    def get_hop_len_sec(self):
        return self.gen_f.get_hop_len_sec()
    
    def get_nb_frames(self):
        return self.gen_f.get_nb_frames()