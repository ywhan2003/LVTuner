import argparse
import logging
import os

from benchmark.utils import DataGenerator


def main():
	logging.basicConfig(level=logging.INFO)

	parser = argparse.ArgumentParser(description="Generate HDF5 data with a given fraction.")
	parser.add_argument("--fraction", type=int, required=True, help="Fraction parameter for range generation")
	parser.add_argument("--dataset", required=True, help="Dataset name without the .hdf5 suffix")
	parser.add_argument(
		"--data_root",
		default=None,
		help="Root directory containing the input dataset and generated output files",
	)
	args = parser.parse_args()

	root_dir = os.path.dirname(__file__)
	data_root = os.path.expanduser(args.data_root) if args.data_root else os.path.join(root_dir, "data")
	dataset = args.dataset
	filepath = os.path.join(data_root, f"{dataset}.hdf5")

	generator = DataGenerator(data_root)
	generator.run(dataset, filepath, fraction=args.fraction)


if __name__ == "__main__":
	main()
