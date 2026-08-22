import argparse
import logging
import os

from benchmark.utils import HDF5Handler


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Convert a folder like msong (base/query/groundtruth files) into a "
            "single HDF5 file with train/test/neighbors datasets."
        )
    )
    parser.add_argument(
        "--input_dir",
        required=True,
        help="Input folder containing base/query/groundtruth files.",
    )
    parser.add_argument(
        "--output_hdf5",
        required=True,
        help="Output HDF5 path.",
    )
    parser.add_argument(
        "--base_file",
        default="",
        help="Optional explicit base filename/path. Default: auto match '*base.fvecs'.",
    )
    parser.add_argument(
        "--query_file",
        default="",
        help="Optional explicit query filename/path. Default: auto match '*query.fvecs'.",
    )
    parser.add_argument(
        "--groundtruth_file",
        default="",
        help=(
            "Optional explicit groundtruth filename/path. "
            "Default: auto match '*groundtruth.ivecs'."
        ),
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite output_hdf5 if it already exists.",
    )
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    args = parse_args()

    summary = HDF5Handler.convert_folder_to_hdf5(
        input_dir=os.path.abspath(args.input_dir),
        output_hdf5=os.path.abspath(args.output_hdf5),
        base_file=args.base_file or None,
        query_file=args.query_file or None,
        groundtruth_file=args.groundtruth_file or None,
        overwrite=args.overwrite,
    )
    logging.info("Folder to HDF5 completed: %s", summary)


if __name__ == "__main__":
    main()
