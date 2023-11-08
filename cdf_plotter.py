import csv
import numpy as np
import matplotlib.pyplot as plt


def draw_cdf(cdf_headers: list[str],
             cdf_data: list[list[float]],
             metric_name: str,
             mark_every: int = 1,
             title: str = ""):
    markers = ["*", "v", "+", "x", "d", "1"]
    marker_idx = 0

    plt.figure()

    plt.rcParams["font.family"] = "Times New Roman"
    plt.rcParams['font.size'] = 15

    for group_idx, group_data in enumerate(cdf_data):
        sorted_data = np.sort(group_data)
        cumulative_prob = np.arange(1, len(group_data) + 1) / len(group_data)
        plt.plot(
            sorted_data,
            cumulative_prob,
            # marker=markers[marker_idx],
            marker=None,
            # markersize=6,
            # markevery=mark_every,
            label=cdf_headers[group_idx])
        marker_idx += 1
        if marker_idx >= len(markers):
            marker_idx = 0
    plt.title('Cumulative Distribution Function (CDF)')
    if len(title) > 0:
        plt.title(title)
    plt.xlabel(metric_name)
    plt.ylabel('Cumulative Probability')

    plt.xticks(np.arange(0, 110, 10))
    plt.yticks(np.arange(0, 1.1, 0.1))

    plt.grid(True)
    if len(cdf_data) > 1:
        plt.legend()

    plt.show()


def read_csv(csv_file_name: str):
    with open(csv_file_name, 'r') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=",")

        # read headers and initialize data arrays
        cdf_headers = next(csv_reader)
        cdf_data = [[] for i in range(len(cdf_headers))]

        # read data
        for row in csv_reader:
            for idx, one_data in enumerate(row):
                cdf_data[idx].append(float(one_data))

    return cdf_headers, cdf_data


def main():
    cdf_headers, cdf_data = read_csv("data.csv")
    draw_cdf(cdf_headers, cdf_data, "Command transmission time (ms)")


if __name__ == "__main__":
    main()
