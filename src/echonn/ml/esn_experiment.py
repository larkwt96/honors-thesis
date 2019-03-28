from .time_series_forecasting import TSData


class EchoStateExperiment:
    def __init__(self, ts_data):
        self.ts_data = ts_data

    def run(self, model, **model_params):
        """
        model - must have the time series api
        model_params - is a dictionary of arrays of parameters to pass to the
        model for cross training

        Returns a dictionary of results
        """
        pass
