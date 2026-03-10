from __future__ import annotations

# Created: 2026-03-10
# Last Updated: 2026-03-10
# Author: Can Bagirgan

from io_api import simulate_from_files, simulation_result_to_dict


def simple_example() -> None:
    # By default reads JSON files from the same directory
    result = simulate_from_files("hardware_config.json", "graph.json")
    as_dict = simulation_result_to_dict(result)
    print(as_dict)


if __name__ == "__main__":
    simple_example()

