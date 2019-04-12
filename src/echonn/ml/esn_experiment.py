from copy import deepcopy
import numpy as np
from ..sys import SystemSolver
from . import EchoStateNetwork


class ESNExperiment:
    def __init__(self, Model, lce_T, time_steps_per_lce_time=1000):
        """ Model is the dynamical system to test with """
        self.ModelClass = Model  # assign the class not initialize it
        self.time_steps_per_lce_time = time_steps_per_lce_time

    # TODO: visualization tools?

    def run(self):
        """
        build model, extract data and perform analysis

        Return
            system integration results: run
            LCE results: (lce, run)
            trained model: esn
            data set: ts_data
            param performance: {pair: RMSE vector per each network trial}
        """
        model_run, lce, lce_run = self.build_model()
        esn, ts_data, param_performance = self.build_esn(model_run)
        return model_run, (lce, lce_run), ts_data, esn, param_performance

    def build_model(self):
        """
        Generate data
            Build Model with default params
            Generate random IC
            Calc Max LCE from IC
            Calc Time is 20 times 1 / LCE
            Calc Step Size is 1 / LCE / 1000 (10,000?)
            Integrate model
        """
        model = self.ModelClass()
        y0 = model.get_rnd_ic()
        model_solver = SystemSolver(model)
        lce_T = model.get_best_lce_T()
        lce, lce_run = model_solver.get_lce(lce_T, y0)
        lce_time = 1 / lce
        Tf = lce_time * 50
        time_step = lce_time / self.time_steps_per_lce_time
        T_range = np.arange(0, Tf, time_step)
        model_run = model_solver.run([0, Tf], y0,
                                     dense_output=True,
                                     t_eval=T_range)
        return model_run, lce, lce_run

    def build_esn(self, model_run):
        """
        Analyze data
            Automate different ESN params and pairs of those params
            Initialize N models with a pair (recommended ranges from paper)
            Train all with train data
            Test with CV many runs
            performance is end RMSE (long term forecast)
            Collect AVG performance and best model
            Use param pair with best average performance
            Use model with best performance, retrain on train and cv
            Run model on testing set and calculate RMSE for every time point
        """
        ts_data = TSData(run=model_run)
        ts_analysis = TSAnalysis(ts_data)
        results = ts_analysis.run()
        return ts_data, results


class TSAnalysis:
    def __init__(self, ts_data, alpha=None, N=None, T0=None, trials=50):
        """
        ts_data is a TSDAta object
        """
        self.ts_data = ts_data
        self.alpha = alpha
        self.N = N
        self.T0 = T0
        self.trials = trials

    def run_params(self, alpha, N, T0):
        K = 0
        # N set from param
        L = self.ts_data.dim
        t_train, y_train = self.ts_data.train
        t_validation, y_validation = self.ts_data.validation
        esns = []
        rmses = []
        for _ in range(self.trials):
            esn = EchoStateNetwork(K, N, L, T0=T0, alpha=alpha)
            esn.fit(y_train)
            Tf = t_train.shape[0] + t_validation.shape[0]
            ys = esn.predict(y_train, Tf)
            ds = self.ts_data.y[:self.ts_data.cv_index_end]
            T = t_train.shape[0]
            (_, _, train_rmse), (_, _, cv_rmse) = esn.score(ds, ys, T=T)
            esns.append(esn)
            rmses.append((train_rmse, cv_rmse))
        return esns, rmses

    def run(self):
        """
        Analyze data
            Automate different ESN params and pairs of those params
            Initialize N models with a pair (recommended ranges from paper)
            Train all with train data
            Test with CV many runs
            performance is end RMSE (long term forecast)
            Collect AVG performance and best model
            Use param pair with best average performance
            Use model with best performance, retrain on train and cv
            Run model on testing set and calculate RMSE for every time point


            build lots of models with with params, train on train
            We want to find the best params with cv

            for each param:
                build lots of models
                train each with the param on train
                collect rmse with cv
                take avg of all
                take best model, avg rmse, std rmse

        TODO: with avg rmse, we can filter bad esns to get the more accurate
        rmse. We can do this since some esns aren't actually esns. This
        requires research on that statistical algorithms to do this.

        TODO: Add the noise param

        Return
            A dictionary of arrays where the index of each array is for a
            parameter (stored in 'params'). Use something like argsort to
            dissect the results.

            testing rmse is different since there is an rmse for every time
            value. Then, it still has a valuable rmse when it's long past its
            predictability limit.
            {
                'params': [ param pairs, ... ]
                'best model': [ best model for param (based on cv), ... ]
                'best model rmse': [ (train, cv, full_train, test), ... ]
                'avg rmse': [ (train, cv), ... ]
                'std dev rmse': [ (train, cv), ... ]
            }
        """
        params = []
        best_models = []
        best_model_rmses = []
        avg_rmses = []
        std_dev_rmses = []
        for pair in self.build_param_pairs():
            esns, rmses = self.run_params(*pair)
            cvs = [rmse[1] for rmse in rmses]  # cv for all models
            bmi = np.argmax(cvs)  # best model index by cv rmse
            best_model = esns[bmi]  # get best model
            bm_train, bm_cv = rmses[bmi]  # unpack train rmse and cv rmse
            # get full rmse fector
            full_rmse = self.get_rmse_vector(best_model, bm_train, bm_cv)
            best_model_rmses.append(full_rmse)
            # get mean train rmse and cv rmse
            avg_rmses.append(np.mean(rmses, axis=1))
            # get std dev train rmse and cv rmse
            std_dev_rmses.append(np.std(rmses, axis=1))
        return {
            'params': params,
            'best model': best_models,
            'best model rmse': best_model_rmses,
            'avg rmse': avg_rmses,
            'std dev rmse': std_dev_rmses,
        }

    def get_rmse_vector(self, best_model, bm_train, bm_cv):
        y = self.ts_data.y[:self.ts_data.cv_index_end]
        y_train = self.ts_data.full_train_y
        ys = best_model.fit(y_train)
        Tf = y.shape[0]
        ys = best_model.predict(y_train, Tf=Tf)
        bm_score = best_model.score(y, ys, y_train.shape[0])  # get rmse
        (_, _, bm_full_train), (test_d, test_y, _) = bm_score  # unpack
        bm_test = []
        for i in range(1, test_d.shape[0]+1):
            rmse = best_model.rmse(test_d[:i], test_y[:i])
            bm_test.append(rmse)
        return (bm_train, bm_cv, bm_full_train, bm_test)

    def build_param_pairs(self):
        # affects computation
        alpha = np.arange(0.7, 0.98, 0.02, dtype=np.float64)
        N = np.arange(5, 500, 10, dtype=int)
        T0 = np.arange(10, 500, 10, dtype=int)
        #complexity = alpha.shape[0] * N.shape[0] * T0.shape[0]
        return itertools.product(alpha, N, T0)


class TSData:
    def __init__(self, data=None, run=None, split=0.9):
        """
        data - is a tuple of t and y of the following format:
            t[time point]
            y[time point, dimension]
        run - is what's returned by the system solver (note that system
        solver has y[dimension, time point], so this is transposed)
        """
        if run is not None:
            res = run['results']
            data = res.t, res.y.T
        self.run = run
        self.t, self.y = data
        self.y = self.y
        self.N, self.dim = self.y.shape
        self.test_index = int(self.N * split)
        self.cv_index_end = self.test_index
        self.cv_index = int(self.N * split**2)
        self.train_index_end = self.cv_index

        self.train_t = self.t[:self.cv_index]
        self.train_y = self.y[:self.cv_index, :]

        self.validation_t = self.t[self.cv_index:self.test_index]
        self.validation_y = self.y[self.cv_index:self.test_index, :]

        self.full_train_t = self.t[:self.test_index]
        self.full_train_y = self.y[:self.test_index, :]

        self.test_t = self.t[self.test_index:]
        self.test_y = self.y[self.test_index:, :]

    @property
    def train(self):
        return self.train_t, self.train_y

    @property
    def validation(self):
        return self.validation_t, self.validation_y

    @property
    def full_train(self):
        return self.full_train_t, self.full_train_y

    @property
    def test(self):
        return self.test_t, self.test_y
