import math

# https://en.wikipedia.org/wiki/Exponential_smoothing#Double_exponential_smoothing
from collections import deque


class Estimator:

    def __init__(self):
        self.total_time_passed = 0

        self.data_smoothing_factor = 0.95  # 0 <= data_smoothing_factor <= 1
        self.trend_smoothing_factor = 0.05  # 0 <= trend_smoothing_factor <= 1

        self.raw_x_position_buffer = deque(maxlen=2)
        self.raw_y_position_buffer = deque(maxlen=2)
        self.x_smoothed_buffer = deque(maxlen=2)  # s_t
        self.x_trend_estimate_buffer = deque(maxlen=2)  # b_t

        self.position_buffer = deque(maxlen=2)
        self.smoothed_buffer = deque(maxlen=2)
        self.trend_estimate_buffer = deque(maxlen=2)

        self.initialized_for_calculations = False

    def initialize_estimator(self, initial_position: list, next_position: list) -> None:
        self.smoothed_buffer.append(initial_position)  # s_0

        x0, y0, width0, height0 = initial_position
        x1, y1, width1, height1 = next_position

        self.trend_estimate_buffer.append(([(x1 - x0) / self.total_time_passed, (y1 - y0) / self.total_time_passed,
                                            (width0 + width1) / 2, (height0 + height1) / 2]))  # b_0
        self.initialized_for_calculations = True

    def update_time(self) -> None:
        self.total_time_passed += 1

    def add_raw_data(self, position: list) -> None:
        """
        :param position: Rectangle surrounding detected ball area
        :return: None
        """
        self.position_buffer.append(position)
        if self.initialized_for_calculations is False and len(self.position_buffer) == self.position_buffer.maxlen:
            self.initialize_estimator(self.position_buffer[0], self.position_buffer[1])

    def predict(self):
        print("Entering predict:")
        if not self.initialized_for_calculations:
            return

        # should be called only if total time > 0
        print(f"\t Position buffer = {self.position_buffer[0]}")
        true_x = self.position_buffer[0][0]
        true_y = self.position_buffer[0][1]
        true_width = self.position_buffer[0][2]
        true_height = self.position_buffer[0][3]

        smoothed_previous = self.smoothed_buffer[-1]
        trend_previous = self.trend_estimate_buffer[-1]
        print(f"\t smoothed_previous = {smoothed_previous}")

        # new_smoothed = self.data_smoothing_factor * self.raw_x_position_buffer[-1] + (1 - self.data_smoothing_factor) * (
        #             smoothed_previous_x + trend_previous_x)

        smoothed_x = self.__calculate_smoothed_value(true_x, smoothed_previous[0], trend_previous[0])
        smoothed_y = self.__calculate_smoothed_value(true_y, smoothed_previous[1], trend_previous[1])

        print(f"\t smoothed_x = {smoothed_x}")
        print(f"\t smoothed_y = {smoothed_y}")

        trend_x = self.__calculate_trend_estimate(smoothed_x, smoothed_previous[0], trend_previous[0])
        trend_y = self.__calculate_trend_estimate(smoothed_y, smoothed_previous[1], trend_previous[1])

        print(f"\t trend_x = {trend_x}")

        prediction_x = smoothed_x + trend_x * 2#self.total_time_passed
        prediction_y = smoothed_y + trend_y * 2#self.total_time_passed

        print(f"\t prediction_x = {prediction_x}")
        print()

        return [prediction_x, prediction_y, true_width, true_height]

    def __calculate_smoothed_value(self, observed_true, previous_smoothed, previous_trend):
        return self.data_smoothing_factor * observed_true + \
               (1 - self.data_smoothing_factor) * (previous_smoothed + previous_trend)

    def __calculate_trend_estimate(self, current_smoothed, previous_smoothed, previous_trend):
        return self.trend_smoothing_factor * (current_smoothed - previous_smoothed) + \
               (1 - self.trend_smoothing_factor) * previous_trend
