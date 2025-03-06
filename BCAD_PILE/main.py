"""
Main entry point for the BCAD_PILE package.

This module provides the command-line interface for the pile analysis program.
"""

import os
import sys
import argparse
from core.computation import analyze_pile_foundation, extract_visualization_data


def print_welcome():
    """Print welcome message."""
    print("\n" * 6)
    print("Welcome to use the BCAD_PILE Python program !!")
    print()
    print("  This program is aimed to execute spatial statical analysis of pile")
    print("foundations of bridge substructures. If you have any questions about")
    print("this program, please do not hesitate to write to :")
    print()
    print("CAD Research Group".rjust(50))
    print("Dept.of Bridge Engr.".rjust(50))
    print("Tongji University".rjust(50))
    print("1239 Sipin Road ".rjust(50))
    print("Shanghai 200092".rjust(50))
    print("P.R.of China".rjust(50))
    print()


def main():
    """Main function."""
    # Parse command-line arguments
    parser = argparse.ArgumentParser(
        description="BCAD_PILE: Spatial statical analysis of pile foundations"
    )

    parser.add_argument("input_file", nargs="?", help="Path to input file")

    parser.add_argument(
        "--visualize",
        "-v",
        action="store_true",
        help="Enable visualization after analysis",
    )

    args = parser.parse_args()

    # Print welcome message
    print_welcome()

    # Get input file
    input_file = args.input_file

    if input_file is None:
        input_file = input("Please enter data filename: ")

    # Check if file exists
    if not os.path.exists(input_file):
        # Try adding .dat extension if not specified
        if not input_file.lower().endswith(".dat"):
            test_file = input_file + ".dat"
            if os.path.exists(test_file):
                input_file = test_file
            else:
                print(f"Error: File '{input_file}' not found")
                return 1
        else:
            print(f"Error: File '{input_file}' not found")
            return 1

    # Run analysis
    try:
        results = analyze_pile_foundation(input_file)

        # Check if visualization was requested
        if args.visualize and results["jctr"] == 1:
            vis_data = extract_visualization_data(results)
            if vis_data:
                from visualization.plotter import plot_results

                plot_results(vis_data)

        return 0

    except Exception as e:
        print(f"Error during analysis: {str(e)}")
        import traceback

        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
