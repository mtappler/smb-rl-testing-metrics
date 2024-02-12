class CoverageInfo:
    def __init__(self, coverage_bit_vectors):
        self.coverage_bit_vectors = coverage_bit_vectors
       
        

    def combine(self, other):
        if len(self.coverage_bit_vectors) == 0:
            return other
        combined = []
        for i,bv in enumerate(self.coverage_bit_vectors):
            combined.append(bv | other.coverage_bit_vectors[i])
        return CoverageInfo(combined)

    def compute_coverage(self):
        
        nr_activated = [sum(bv).item() for bv in self.coverage_bit_vectors]
        nr_neurons = [len(bv) for bv in self.coverage_bit_vectors]
        summed_act = 0
        summed_nr = 0
        coverage_ratios = []
        for act,nr_all in zip(nr_activated,nr_neurons):
            coverage_ratios.append(act / nr_all)
            summed_act += act
            summed_nr += nr_all
        return (summed_act/summed_nr,coverage_ratios)
    
    def print_coverage(self):
        nr_activated = [sum(bv).item() for bv in self.coverage_bit_vectors]
        nr_neurons = [len(bv) for bv in self.coverage_bit_vectors]
        print("*"*80)
        print("Coverage info:")
        summed_act = 0
        summed_nr = 0
        for act,nr_all in zip(nr_activated,nr_neurons):
            print(act / nr_all)
            summed_act += act
            summed_nr += nr_all
        print(f"All layers: {summed_act/summed_nr}")
