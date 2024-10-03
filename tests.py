import csv
import os

import numpy as np

from main import create_csv_file, calc_dists_with_min_and_max_values


def test_create_csv_file() -> None:
    n: int = 500
    m: int = 10
    create_csv_file(n, m)
    assert os.path.isfile("vectors.csv"), "The function `create_csv_file` should create  a file."
    with open("vectors.csv", "r") as csv_file:
        vectors: list = list(csv.reader(csv_file))
        assert len(vectors) == n, f"The output file should contain {n} lines."
        errors: list = []
        for ind, vector in enumerate(vectors, start=1):
            if len(vector) != m:
                errors.append(str(ind))
        assert len(errors) == 0, (
            f"The number of items in the lines must be {m}. Problem line numbers {' '.join(errors)}."
        )


def test_calc_dists_with_min_and_max_values() -> None:
    vectors: np.asarray = np.asarray([
        [0.7656655918260113, -0.795505323319804, -0.9415919475440568],
        [-0.47704340096942577, 0.500277617793691, -0.8591777771100182],
        [0.46114518940328897, 0.2683204385347655, -0.4607614864488139],
        [0.9828925661651142, -0.13758319608232417, 0.3724019787742545]
    ])
    example_dist: list = [
        1.79726769479221,
        1.20650564001757,
        1.48547269321608,
        1.04534085578797,
        2.01371997584461,
        1.06355039633095
    ]
    example_min_dist: float = 1.04534085578797
    example_max_dist: float = 2.01371997584461
    example_id_min: tuple = (1, 2)
    example_id_max: tuple = (1, 3)
    dists, min_dist, max_dist, id_min, id_max = calc_dists_with_min_and_max_values(vectors)
    assert example_dist == dists, "The distances do not match."
    assert example_min_dist == min_dist, "The minimum distance does not match."
    assert example_max_dist == max_dist, "The maximum distance does not match."
    assert example_id_min == id_min, "The IDs of the vectors with the minimum distance do not match."
    assert example_id_max == id_max, "The IDs of the vectors with the maximum distance do not match."


if __name__ == "__main__":
    test_create_csv_file()
    test_calc_dists_with_min_and_max_values()
