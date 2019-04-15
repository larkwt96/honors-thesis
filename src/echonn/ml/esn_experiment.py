from copy import deepcopy
import numpy as np
from ..sys import SystemSolver
from . import EchoStateNetwork
import itertools
import time
import matplotlib.pyplot as plt


class ESNExperiment:
    def __init__(self, model, data=None, params=None, trials=2, time_steps_per_lce_time=100):
        self.model = model
        self.data = data
        self.params = params
        self.trials = trials
        self.time_steps_per_lce_time = time_steps_per_lce_time

    # TODO: visualization tools?

    def run(self, verbose=False):
        """
        build model, extract data and perform analysis

        Return
            system integration results: run
            LCE results: (lce, run)
            data set: ts_data
            param performance results:
                {
                    'params': [ param pairs, ... ]
                    'best model': [ best model for param (based on cv), ... ]
                    'best model rmse': [ (train, cv, full_train, test), ... ]
                    'avg rmse': [ (train, cv), ... ]
                    'std dev rmse': [ (train, cv), ... ]
                }
        """
        self.verbose = verbose
        if self.data is None:
            model_run, lce, lce_run = self.build_model()
        else:
            model_run, lce, lce_run = self.data
        ts_data, results = self.build_esn(model_run)
        return model_run, (lce, lce_run), ts_data, results

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
        if self.verbose:
            print('Collecting data...')
        model = self.model
        y0 = model.get_rnd_ic()
        model_solver = SystemSolver(model)
        lce_T = model.best_lce_T
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
        ts_analysis = TSAnalysis(ts_data, self.params, self.trials)
        results = ts_analysis.run(self.verbose)
        return ts_data, results


class TSAnalysis:
    def __init__(self, ts_data, params, trials=2):
        """
        ts_data is a TSDAta object
        """
        self.ts_data = ts_data
        self.params = params
        self.trials = trials

    def run_params(self, alpha, N, T0):
        K = 0
        # N set from param
        L = self.ts_data.dim
        t_train, y_train = self.ts_data.train
        t_validation, y_validation = self.ts_data.validation
        esns = []
        rmses = []
        for i in range(self.trials):
            if self.verbose:
                print('\tRunning Trial {}/{} ... '.format(i+1, self.trials))
                trial_timer = time.time()
            esn = EchoStateNetwork(K, N, L, T0=T0, alpha=alpha)
            esn.fit(y_train)
            T = y_train.shape[0]
            Tf = T + t_validation.shape[0]
            # calc train rmse
            ys_train = esn.predict(y_train[:T0], Tf=T)[T0:]
            ds_train = y_train[T0:]
            rmse_train = esn.rmse(ds_train, ys_train)

            ys_cv = esn.predict(y_train, Tf=Tf)[T:]
            ds_cv = y_validation
            rmse_cv = esn.rmse(ds_cv, ys_cv)

            esns.append(esn)
            rmses.append((rmse_train, rmse_cv))
            if self.verbose:
                trial_timer = time.time() - trial_timer
                print('\t\tTrain RMSE: {:.5}'.format(rmse_train))
                print('\t\tValidation RMSE: {:.5}'.format(rmse_cv))
                print('\t\tTime: {:.5} s'.format(trial_timer))
        return esns, rmses

    def calc_appx_time(self):
        mean_time = np.mean(self.times)
        time_left = mean_time * self.num_pairs - np.sum(self.times)
        time_sign = 's'
        if time_left > 60:
            time_left = time_left / 60
            time_sign = 'm'
            if time_left > 60:
                time_left = time_left / 60
                time_sign = 'h'
                if time_left > 24:
                    time_left = time_left / 24
                    time_sign = 'day'
                    if time_left > 2:
                        time_sign += 's'
                    if time_left > 365.25:
                        time_left = time_left / 365.25
                        time_sign = 'YEARS'
        time_left_str = '{:.3} {}'.format(time_left, time_sign)
        return '(appx. {} remaining)'.format(time_left_str)

    def run(self, verbose=False):
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
                'best model rmse': [ (train, cv, full_train, (ds_test, ys_test, rmse_test)), ... ]
                'avg rmse': [ (train, cv), ... ]
                'std dev rmse': [ (train, cv), ... ]
            }
        """
        self.verbose = verbose
        params = self.build_param_pairs()
        best_models = []
        best_model_rmses = []
        avg_rmses = []
        std_dev_rmses = []
        if self.verbose:
            self.times = []
            self.num_pairs = len(params)
            print('Total Param Pairs:', self.num_pairs)
        for pair in params:
            if verbose:
                self.times.append(time.time())
                print('Testing Params:', pair)
            esns, rmses = self.run_params(*pair)
            cvs = [rmse[1] for rmse in rmses]  # cv for all models
            bmi = np.argmin(cvs)  # best model index by cv rmse
            best_model = esns[bmi]  # get best model
            bm_train, bm_cv = rmses[bmi]  # unpack train rmse and cv rmse
            # get full rmse fector
            if verbose:
                print('\tCalculating test rmse of best model...')
            full_rmse = self.get_rmse_vector(best_model, bm_train, bm_cv)
            if verbose:
                self.times[-1] = time.time() - self.times[-1]
                curr_time = 'Total Time: {:.3} s'.format(self.times[-1])
                appx_time = self.calc_appx_time()
                train, cv, full, _ = full_rmse
                test = full_rmse[-1][-1]
                print('\tTrain RMSE:', train)
                print('\tCV RMSE:', cv)
                print('\tFull Train RMSE:', full)
                print('\tFinal Test RMSE:', test)
                print('\t{} {}'.format(curr_time, appx_time))
            best_models.append(best_model)
            best_model_rmses.append(full_rmse)
            # get mean train rmse and cv rmse
            avg_rmses.append(np.mean(rmses, axis=1))
            # get std dev train rmse and cv rmse
            std_dev_rmses.append(np.std(rmses, axis=1))
        return {
            'params': params,
            'best model': [],  # best_models,
            'best model rmse': best_model_rmses,
            'avg rmse': avg_rmses,
            'std dev rmse': std_dev_rmses,
        }

    def get_rmse_vector(self, best_model, bm_train, bm_cv):
        y = self.ts_data.y
        y_train = self.ts_data.y[:self.ts_data.cv_index_end]
        T0 = best_model.T0
        T = y_train.shape[0]
        Tf = y.shape[0]

        ys_full_train = best_model.predict(y_train[:T0], Tf=T)[T0:]
        ds_full_train = y_train[T0:]
        bm_full_train = best_model.rmse(ds_full_train, ys_full_train)

        # calc test rmse vector
        ys_test = best_model.predict(y_train, Tf=Tf)[T:]
        ds_test = y[T:]
        rmse_test = best_model.rmse(ds_test, ys_test)
        bm_test = (ds_test, ys_test, rmse_test)

        return (bm_train, bm_cv, bm_full_train, bm_test)

    def build_param_pairs(self):
        # affects computation
        # if alpha is None:
            #alpha = [.7, .75, .8, .85, .9, .98]
            #alpha = [.7, .8, .9, .98]
            # alpha.reverse()

        # if N is None:
            #N = [10, 25, 50, 100, 200, 300, 500]
            #N = [10, 75, 300, 500]
            # N.reverse()

        # if T0 is None:
            #T0 = [10, 100, 200, 300, 500]
            #T0 = [100, 250, 500]
            # T0.reverse()

        if self.params is None:
            self.params = [
                (.7, 150, 300),
                (.8, 150, 300),
                (.9, 150, 300),
                (.98, 150, 300),
            ]
        else:
            self.params = list(self.params)

        # alpha_cut = 1
        # N_cut = 1
        # T0_cut = 1
        # return itertools.product(alpha[:alpha_cut], N[:N_cut], T0[:T0_cut])

        # complexity = alpha.shape[0] * N.shape[0] * T0.shape[0]
        return self.params


class TSData:
    def __init__(self, data=None, run=None, split=0.7):
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
