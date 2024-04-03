import csv
import unittest
import logging
import numpy as np
import matplotlib.pyplot as plt


class ImportantValues():

    def __init__(self, scenario_name: str, p90: float, p99: float,
                 p99dot9: float, p99dot99: float, p99dot999: float,
                 p100: float):
        self.scenario_name = scenario_name
        self.p90 = p90
        self.p99 = p99
        self.p99dot9 = p99dot9
        self.p99dot99 = p99dot99
        self.p99dot999 = p99dot999
        self.p100 = p100


class MultiImportantValues():

    def __init__(self, values: list[ImportantValues]):
        self.values = values

    def print(self):
        for row in range(-1, len(self.values)):
            if row == -1:
                print(
                    "Unit: millisecond,90th percentile,99th percentile,99.9th percentile,99.99th percentile,99.999th percentile,100th percentile"
                )
            else:
                print(  # , is the separator in csv, so we replace ', ' with '+'
                    f"{self.values[row].scenario_name.replace(', ', '+')},{self.values[row].p90},{self.values[row].p99},{self.values[row].p99dot9},{self.values[row].p99dot99},{self.values[row].p99dot999},{self.values[row].p100}"
                )


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


# Complementary Cumulative Distribution Function
def draw_ccdf(cdf_headers: list[str],
              cdf_data: list[list[float]],
              metric_name: str,
              mark_every: int = 1,
              title: str = "",
              xticks=None,
              yticks=None,
              x_rotation=0,
              y_rotation=0,
              xlim=None,
              ylim=None):
    markers = ["*", "v", "+", "x", "d", "1", ".", ",", "^", "2", "s"]
    marker_idx = 0

    plt.figure()

    plt.rcParams["font.family"] = "Times New Roman"
    plt.rcParams['font.size'] = 15

    # to print percentiles
    percentiles: list[ImportantValues] = []

    for group_idx, group_data in enumerate(cdf_data):
        sorted_data = np.sort(group_data)
        # P(X < x)
        cumulative_prob = np.arange(0, len(group_data)) / len(group_data)
        # P(X >= x)
        complementary_cumulative_prob = 1 - cumulative_prob

        # save percentile values
        true_cumulative_prob = np.arange(1,
                                         len(group_data) + 1) / len(group_data)
        percentiles.append(
            ImportantValues(
                cdf_headers[group_idx],
                np.interp(0.9, true_cumulative_prob, sorted_data),
                np.interp(0.99, true_cumulative_prob, sorted_data),
                np.interp(0.999, true_cumulative_prob, sorted_data),
                np.interp(0.9999, true_cumulative_prob, sorted_data),
                np.interp(0.99999, true_cumulative_prob, sorted_data),
                np.interp(1, true_cumulative_prob, sorted_data)))

        # Evenly spaced markers on a logarithmic scale
        num_markers = 10  # Number of markers
        mark_every_indices = np.geomspace(1,
                                          len(sorted_data) - 1,
                                          num_markers).astype(int)
        # equivalent to the above one
        # mark_every_indices = np.logspace(0, np.log10(sorted_data),num_markers).astype(int)

        # Now, the mark_every_indices is "tight at the front, loose at the back", such as:
        # [1 4 23 117 574 2810 13757 67346 329665 1613737]

        mark_every_indices = len(sorted_data) - mark_every_indices

        # Now, it becomes:
        # [1613737 1613734 1613715 1613621 1613164 1610928 1599981 1546392 1284073 1]

        # Reverse it.
        mark_every_indices = np.flip(mark_every_indices)
        # mark_every_indices = mark_every_indices[::-1] # equivalent to the above one

        # Now it becomes:
        # [1 1284073 1546392 1599981 1610928 1613164 1613621 1613715 1613734 1613737]
        # This is "loose at the front, tight at the back", which is what we need.

        plt.plot(
            sorted_data,
            complementary_cumulative_prob,
            marker=markers[marker_idx],
            # marker=None,
            # markersize=6,
            markevery=mark_every_indices,
            # markevery=mark_every,
            label=cdf_headers[group_idx])
        marker_idx += 1
        if marker_idx >= len(markers):
            marker_idx = 0
    # plt.title('Complementary CDF')
    # if len(title) > 0:
    #     plt.title(title)
    plt.xlabel(metric_name)
    plt.ylabel('CCDF')

    # convert y-axis to Logarithmic scale
    plt.yscale('log')
    # plt.xscale('log')

    if xticks is not None:
        plt.xticks(xticks, rotation=x_rotation)
    if yticks is not None:
        plt.yticks(yticks, rotation=y_rotation)

    if xlim is not None:
        plt.xlim(xlim)
    if ylim is not None:
        plt.ylim(ylim)

    plt.grid(True)
    # For Logarithmic y-axis, we need to turn on the minor grid, whose values are like:
    # major ticks: 0.1, 0.01
    # minor between the above 2 major ticks: 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09. There are 8 minor ticks between 2 major ticks.
    plt.grid(visible=True, which='minor', linestyle='--')
    if len(cdf_data) > 1:
        plt.legend()

    # print percentiles
    out_percentiles = MultiImportantValues(percentiles)
    out_percentiles.print()

    # save the figure in file
    # if len(title) > 0:
    #     plt.savefig(f'{OUT_FIG_PATH}/{title}, {metric_name}.png',
    #                 bbox_inches='tight')
    plt.show()
    plt.close()


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


class ChartsPlotters(unittest.TestCase):

    def setUp(self):
        # log configuration
        log_fmt = '%(asctime)s, %(levelname)s, %(filename)s:%(lineno)d, %(funcName)s(), %(message)s'
        logging.basicConfig(format=log_fmt, level=logging.INFO)

    # python -u -m unittest cdf_plotter.ChartsPlotters.cdf
    def cdf(self):
        cdf_headers, cdf_data = read_csv("data.csv")
        draw_cdf(cdf_headers, cdf_data, "Command transmission time (ms)")

    # python -u -m unittest cdf_plotter.ChartsPlotters.ccdf
    def ccdf(self):
        cdf_headers1, cdf_data1 = read_csv("data.csv")
        cdf_headers2, cdf_data2 = read_csv("data2.csv")
        draw_ccdf([cdf_headers1[0], cdf_headers2[0]],
                  [cdf_data1[0], cdf_data2[0]],
                  "Command transmission time (ms)")
