# https://en.wikipedia.org/wiki/Exponential_smoothing#Double_exponential_smoothing
# https://www.itl.nist.gov/div898/handbook/pmc/section4/pmc433.htm
from collections import deque


class Estimator:
    """Provides an implementation for double exponential smoothing.
    """

    def __init__(self, initial_pos: list, next_pos: list):
        """Create and initialize the estimator for forecasting.
        :param initial_pos: Initial position of Rectangle [top-left x, top-left y, width, height].
        :param next_pos: Next position of Rectangle [top-left x, top-left y, width, height].
        """
        self.__data_smoothing_factor = 0.8  # 0 <= data_smoothing_factor <= 1
        self.__trend_smoothing_factor = 0.2  # 0 <= trend_smoothing_factor <= 1

        self.__position_buffer = deque([initial_pos, next_pos], maxlen=2)

        x0, y0, width0, height0 = self.__position_buffer[0]
        x1, y1, width1, height1 = self.__position_buffer[1]

        self.__previous_smoothed = (x0, y0)
        self.__previous_trend = (x1 - x0, y1 - y0)

    def add_data(self, position: list) -> None:
        """Add data to the position buffer.
        :param position: Bounding rectangle [top-left x, top-left y, width, height] of tracked object.
        :return: None
        """
        self.__position_buffer.append(position)

    def predict(self, t=1.0) -> list:
        """Forecasts a Rectangle [top-left x, top-left y, width, height] for time t=X.
        :param t: Time-step for which the forecast is made. Fractional time-steps are supported.
        :return: Predicted future bounding rectangle [top-left x, top-left y, width, height] of tracked object.
        """
        true_x, true_y, true_width, true_height = self.__position_buffer[-1]

        smoothed_previous_x, smoothed_previous_y = self.__previous_smoothed
        trend_previous_x, trend_previous_y = self.__previous_trend

        smoothed_x = self.__calculate_smoothed_value(true_x, smoothed_previous_x, trend_previous_x)
        smoothed_y = self.__calculate_smoothed_value(true_y, smoothed_previous_y, trend_previous_y)

        trend_x = self.__calculate_trend_estimate(smoothed_x, smoothed_previous_x, trend_previous_x)
        trend_y = self.__calculate_trend_estimate(smoothed_y, smoothed_previous_y, trend_previous_y)

        # The forecast follows an equation of the form: Prediction = b + tx,
        # Where t designates the time in the future for which the forecast is made.
        # i.e. t=1 means the forecast is for the next possible time-step.
        prediction_x = smoothed_x + t * trend_x
        prediction_y = smoothed_y + t * trend_y

        # Update the previous values of the estimates.
        self.__previous_smoothed = (smoothed_x, smoothed_y)
        self.__previous_trend = (trend_x, trend_y)

        # Return a "rectangle" [top-left x, top-left y, width, height]
        return [int(prediction_x), int(prediction_y), true_width, true_height]

    def __calculate_smoothed_value(self, observed_true: float, prev_smoothed: float, prev_trend: float) -> float:
        """Calculate the 'smoothed value' part of a double-exponential smoothing process.

        :param observed_true: Top-left corner of bounding rectangle which is assumed to be a true positive.
        :param prev_smoothed: Smoothed top-left corner of a bounding rectangle from previous time-step.
        :param prev_trend: Trend value from previous time-step.
        :return: Smoothed estimate of top-left corner of bounding rectangle.
        """
        return self.__data_smoothing_factor * observed_true + (1 - self.__data_smoothing_factor) * (
                prev_smoothed + prev_trend)

    def __calculate_trend_estimate(self, cur_smoothed: float, prev_smoothed: float, prev_trend: float) -> float:
        """Calculate the 'trend value' part of a double-exponential smoothing process.

        :param cur_smoothed: Smoothed value at current time-step.
        :param prev_smoothed: Smoothed value at previous time-step.
        :param prev_trend: Previous trend value.
        :return: New trend value.
        """
        return self.__trend_smoothing_factor * (cur_smoothed - prev_smoothed) + \
               (1 - self.__trend_smoothing_factor) * prev_trend
