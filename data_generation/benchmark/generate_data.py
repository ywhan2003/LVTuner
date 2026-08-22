import numpy as np
from utils import RangeHandler

ranges = RangeHandler.generate_fraction_ranges(20, 0, 10000, 17, rng=np.random.default_rng(42))
print(ranges)

print("Saving ranges to binary file...")
RangeHandler.save_ranges_to_bin(ranges, out_dir="output")
print("Done.")

print("Loading ranges from binary file...")
loaded_ranges = RangeHandler.load_ranges_from_bin("output/ranges.bin")
print(loaded_ranges)