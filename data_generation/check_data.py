import argparse

from benchmark.utils import HDF5Handler


def main():
	parser = argparse.ArgumentParser(description="Read an HDF5 file and show datasets.")
	parser.add_argument("file", help="Path to the HDF5 file")
	args = parser.parse_args()

	data = HDF5Handler.read_hdf5_file(args.file)
	for key in sorted(data.keys()):
		value = data[key]
		shape = getattr(value, "shape", None)
		dtype = getattr(value, "dtype", None)
		print(f"{key}: shape={shape}, dtype={dtype}")


if __name__ == "__main__":
	main()
