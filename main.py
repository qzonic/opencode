import itertools
import random

import numpy as np
import matplotlib.pyplot as plt


FILENAME: str = "vectors.csv"  # Output file name
MIN_NUMBER: int = -1  # Minimum random value
MAX_NUMBER: int = 1  # Maximum random value


def create_csv_file(n: int, m: int) -> None:
    """
    Generate csv file with random numbers.

    :param int n: the number of lines (500<n≤1000).
    :param int m: the number of random numbers in a line (10<m≤50).
    :return:
    """

    vectors: np.ndarray = np.asarray([[random.uniform(MIN_NUMBER, MAX_NUMBER) for j in range(m)] for i in range(n)])
    np.savetxt(FILENAME, vectors, delimiter=",")


def read_vectors_file() -> np.ndarray:
    """
    Reads the file and converts it to an array.

    :return: vectors obtained from the file.
    :rtype: np.ndarray
    """
    return np.genfromtxt(FILENAME, delimiter=",")


def calc_dists_with_min_and_max_values(vectors: np.ndarray) -> tuple[list, float, float, tuple, tuple]:
    """
    Calculates the distances between vectors, their maximum and minimum values, numbers  these vectors.

    :param np.ndarray vectors: array of vectors.
    :return: tuple with distances, minimum and maximum vectors, numbers of minimum and maximum vectors.
    :rtype: tuple.
    """

    min1: float = np.linalg.norm(vectors[0] - vectors[1])
    max1: float = np.linalg.norm(vectors[0] - vectors[1])
    id_min1: tuple = (0, 1)
    id_max1: tuple = (0, 1)
    dists: list = []
    for i in itertools.combinations(enumerate(vectors), 2):
        value: float = round(np.linalg.norm(i[0][1] - i[1][1]), 14)
        dists.append(value)
        if value < min1:
            min1 = value
            id_min1 = (i[0][0], i[1][0])
        elif value > max1:
            max1 = value
            id_max1 = (i[0][0], i[1][0])
    return dists, min1, max1, id_min1, id_max1


def draw_histograms(dists: list) -> None:
    """
    Shows a histogram of the distances.

    :param list dists: List of distances.
    :return:
    """
    plt.hist(dists, rwidth=0.75, weights=np.ones_like(dists) / len(dists))
    plt.title("Распределение евклидовых расстояний")
    plt.xlabel("Расстояние")
    plt.ylabel("Частота")
    plt.grid(axis="y", alpha=0.75)
    plt.show()


def main() -> None:
    create_csv_file(1000, 50)
    vectors: np.ndarray = read_vectors_file()
    dists, min_dist, max_dist, id_min, id_max = calc_dists_with_min_and_max_values(vectors)
    print(f"Номера векторов пары с минимальным расстоянием: {id_min}. Значение: {min_dist}")
    print(f"Номера векторов пары с максимальным расстоянием: {id_max}. Значение: {max_dist}")
    draw_histograms(dists)


if __name__ == "__main__":
    main()
