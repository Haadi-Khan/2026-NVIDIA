import unittest
import numpy as np
import pandas as pd
import os
from mts import LabsEnergyCounter, merit_factor, memetic_tabu_search

def hex_to_sequence(hex_str, length):
    """
    Converts a hexadecimal string to a sequence of +/-1.
    The hex string represents a binary sequence where 0 is -1 and 1 is +1.
    """
    # hex_str is typically something like '0D' or '019A' or '0002F4A9B1B28671'
    # We need to handle potential leading zeros and ensure the length is correct.
    
    val = int(hex_str, 16)
    # Convert hex to bits, ensuring we keep all leading zeros from the hex
    # by using the length of the hex string * 4.
    full_bin_str = bin(val)[2:].zfill(len(hex_str) * 4)
    # Take the LAST length bits
    bin_str = full_bin_str[-length:]

    seq = []
    for bit in bin_str:
        if bit == '0':
            seq.append(-1)
        else:
            seq.append(1)
    return np.array(seq), bin_str

class TestLabsEnergy(unittest.TestCase):
    def setUp(self):
        self.energy_counter = LabsEnergyCounter()
        self.data_path = 'data82.csv'

    def test_energy_against_data82(self):
        """
        Verify the energy function against merit factors in data82.csv for N <= 20.
        """
        if not os.path.exists(self.data_path):
            self.skipTest(f"{self.data_path} not found")

        df = pd.read_csv(self.data_path)
        # Use MF_MF and Optimal_or_Best_MF columns
        for _, row in df.iterrows():
            N = int(row['Len'])
            if N > 50:
                continue

            target_mf = float(row['MF_MF'])
            hex_seq = str(row['Optimal_or_Best_MF'])
            
            seq, bin_str = hex_to_sequence(hex_seq, N)
            energy = self.energy_counter(seq)
            mf = merit_factor(seq, energy)
            
            print(f"Testing N={N}, target_mf={target_mf}, calculated_mf={mf:.3f}")
            
            try:
                self.assertAlmostEqual(mf, target_mf, delta=0.1, 
                                       msg=f"N={N} failed: calculated MF={mf}, target MF={target_mf}")
            except AssertionError:
                # Try reversed sequence
                seq_rev = seq[::-1]
                energy_rev = self.energy_counter(seq_rev)
                mf_rev = merit_factor(seq_rev, energy_rev)
                if abs(mf_rev - target_mf) < 0.1:
                    continue
                
                print(f"DEBUG: N={N}, hex={hex_seq}, calculated MF={mf}, target MF={target_mf}")
                raise

class TestMTS(unittest.TestCase):
    def setUp(self):
        self.energy_counter = LabsEnergyCounter()
        self.data_path = 'data82.csv'

    def test_mts_against_data82(self):
        """
        Verify that MTS finds the optimal merit factors in data82.csv for N <= 20.
        """
        if not os.path.exists(self.data_path):
            self.skipTest(f"{self.data_path} not found")

        df = pd.read_csv(self.data_path)
        
        # MTS parameters from notebook
        pop_size = 20
        max_generations = 1000
        
        # Initialize LabsEnergyCounter without warning for tests
        self.energy_counter = LabsEnergyCounter(warn_at=None)
        
        # We test N <= 20
        for _, row in df.iterrows():
            N = int(row['Len'])
            if N > 20:
                continue

            target_mf = float(row['MF_MF'])
            
            # Reset counter and seed for reproducibility in each N
            self.energy_counter.reset()
            np.random.seed(42) 

            best_s, best_energy, population, energies = memetic_tabu_search(
                N=N,
                energy_func=self.energy_counter,
                pop_size=pop_size,
                max_generations=max_generations,
                p_mut=0.1,
                p_combine=0.5,
                tabu_max_iter=100,
                tabu_tenure=7,
                verbose=False,
                target_energy=int(round(N**2 / (2 * target_mf)))
            )
            
            mf = merit_factor(best_s, best_energy)
            
            print(f"MTS Test N={N}: target_mf={target_mf}, found_mf={mf:.3f}, evals={self.energy_counter.count}")
            
            self.assertAlmostEqual(mf, target_mf, delta=0.1, 
                                   msg=f"MTS failed for N={N}: found MF={mf}, target MF={target_mf}")

if __name__ == '__main__':
    unittest.main()