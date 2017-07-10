"""
RACOG
=====

Rapidy Converging Gibbs sampler for data oversampling

.. note::
    "B. Das, N. C. Krishnan and D. J. Cook,
    "RACOG and wRACOG: Two Probabilistic Oversampling Techniques,"
    in IEEE Transactions on Knowledge and Data Engineering,
    vol. 27, no. 1, pp. 222-234, Jan. 1 2015.
    doi: 10.1109/TKDE.2014.2324567"
    .. _a link: http://ieeexplore.ieee.org/document/6816044/

.. module:: racog
   :platform: Unix
   :synopsis: oversampling method

"""

import numpy as np
import pandas as pd

from pomegranate import BayesianNetwork

from imblearn.over_sampling.base import BaseOverSampler
from caimcaim import CAIMD
from mdlp import MDLP
from tqdm import *

from functools import partial
import multiprocessing


class RACOG(BaseOverSampler):
    """
    RACOG oversampling class

    Parameters
    ----------
    ratio : str, dict, or callable, optional (default='auto')
        Ratio to use for resampling the data set.

        - If ``str``, has to be one of: (i) ``'minority'``: resample the
          minority class; (ii) ``'majority'``: resample the majority class,
          (iii) ``'not minority'``: resample all classes apart of the minority
          class, (iv) ``'all'``: resample all classes, and (v) ``'auto'``:
          correspond to ``'all'`` with for over-sampling methods and ``'not
          minority'`` for under-sampling methods. The classes targeted will be
          over-sampled or under-sampled to achieve an equal number of sample
          with the majority or minority class.
        - If ``dict``, the keys correspond to the targeted classes. The values
          correspond to the desired number of samples.
        - If callable, function taking ``y`` and returns a ``dict``. The keys
          correspond to the targeted classes. The values correspond to the
          desired number of samples.

    random_state : int, RandomState instance or None, optional (default=None)
        If int, ``random_state`` is the seed used by the random number
        generator; If ``RandomState`` instance, random_state is the random
        number generator; If ``None``, the random number generator is the
        ``RandomState`` instance used by ``np.random``.

    discretization: 'caim' or 'mdlp'
        Method for discretization continuous variables

    categorical_features : 'auto' or 'all' or list/array of indices or list of labels
        Specify what features are treated as categorical (not using discretization).
        - 'auto' (default): Only those features whose number of unique values exceeds
                            the number of classes
                            of the target variable by 2 times or more
        - array of indices: array of categorical feature indices
        - list of labels: column labels of a pandas dataframe

    warmup_offset: int
        Warm up for gibbs sampler. It is number of sample generation iterations
        that are needed for the samples to reach a stationary distribution

    lag0: int
        Lag is the number of consecutive samples that are discarded from
        the Markov chain following each accepted sample to avoid
        autocorrelation between consecutive samples

    n_iter: int or 'auto'
        Total number of iteration per one run of Gibbs sampler

    ignition_step: int or 'auto'
        Gap between two samples of original data using for ignite Gibbs sampler

    root: int
        Index of the root feature using in Chow-Liu algorithm

    continous_distribution: 'normal' or 'laplace'
        The distribution using for sampling (reconstruct) continous variables
        after oversampling

    only_sampled: bool
        Concatenate or not original data with new samples. If True,
        only new samples will return

    alpha: float
        A threshold for continuous data reconstuction that allows to change
        between considering all instances within the interval or just those
        belonging to the same class [2]

    L: float
        The factor that defines a width within the distribution using to reconstruct
        continuous data [2]

    threshold: int
        If number of samples is needed for oversampling less than threshold,
        no oversamping will be made for that class

    eps: float
        A very small value to replace zero values of any probability

    verbose: int
        If greather than 0, enable verbose output

    shuffle: bool
        Shuffle or not the original data and a sampled array. If 'False',
        new rows will be stacked after original rows

    n_jobs: int
        The number of jobs to run in parallel for samplng

    References
    ----------
    [1] B. Das, N. C. Krishnan and D. J. Cook,
        "RACOG and wRACOG: Two Probabilistic Oversampling Techniques,"
        in IEEE Transactions on Knowledge and Data Engineering,
        vol. 27, no. 1, pp. 222-234, Jan. 1 2015.
        doi: 10.1109/TKDE.2014.2324567
        http://ieeexplore.ieee.org/document/6816044/

    [2] João Roberto Bertini Junior, Maria do Carmo Nicoletti, Liang Zhao,
        "An embedded imputation method via Attribute-based Decision Graphs",
        Expert Systems with Applications, Volume 57, 2016, Pages 159-177,
        ISSN 0957-4174, http://dx.doi.org/10.1016/j.eswa.2016.03.027.
        http://www.sciencedirect.com/science/article/pii/S0957417416301208

    Example
    ---------

    """

    def __init__(self, ratio='auto', random_state=None, discretization='caim', categorical_features='auto',
                 warmup_offset=100, lag0=20, n_iter='auto', ignition_step='auto', root=0,
                 continous_distribution='normal', only_sampled=False, alpha=0.6, L=0.5,
                 threshold=10, eps=10E-5, verbose=1, shuffle=False, n_jobs=1):

        super().__init__(ratio=ratio, random_state=random_state)
        self.type_disc = discretization
        self.categorical = categorical_features
        if not categorical_features:
            self.categorical = 'auto'
        self.i_categorical = []  # store only index of columns with categorical features

        if continous_distribution == 'normal':
            self.f_sample = np.random.normal
        elif continous_distribution == 'laplace':
            self.f_sample = np.random.laplace
        else:
            raise WrongDistributionException("'continous_distribution' must be 'normal' or 'laplace'")
        self.root = root
        self.only_sampled = only_sampled
        self.alpha = alpha
        self.L = L
        self.offset = warmup_offset
        self.lag = lag0
        self.n_iter = n_iter
        self.ign_step = ignition_step
        self.threshold = threshold
        self.eps = eps
        self.verbose = verbose
        self.shuffle = shuffle
        self.n_jobs = n_jobs
        if n_jobs == -1:
            self.n_jobs = multiprocessing.cpu_count()

    def fit(self, X, y):
        self.i_categorical = []
        self.pdflag = False
        if isinstance(X, pd.DataFrame):
            # self.indx = X.index
            # self.columns = X.columns
            if isinstance(self.categorical, list):
                self.i_categorical = [X.columns.get_loc(label) for label in self.categorical]
            X = X.values
            y = y.values
            self.pdflag = True
        X_di = X
        super().fit(X, y)

        if self.categorical != 'all':
            if self.categorical == 'auto':
                self.i_categorical = self.check_categorical(X, y)
            elif (isinstance(self.categorical, list)) or (isinstance(self.categorical, np.ndarray)):
                if not self.pdflag:
                    self.i_categorical = self.categorical[:]
        else:
            self.i_categorical = np.arange(X.shape[1]).tolist()

        continuous = self._get_countinuous(X_di)
        if continuous:
            if self.type_disc == 'caim':
                self.disc = CAIMD(categorical_features=self.i_categorical)
            elif self.type_disc == 'mdlp':
                self.disc = MDLP(categorical_features=self.i_categorical)
            X_di = self.disc.fit_transform(X_di, y)

        self.probs = {}
        self.priors = {}
        self.structure = {}
        # self.ptable = {}
        for class_sample, n_samples in self.ratio_.items():
            if n_samples < self.threshold:
                continue
            probs, priors, depend = self._tan(X_di, y, X_di[y == class_sample])
            self.probs[class_sample] = probs
            self.priors[class_sample] = priors
            self.structure[class_sample] = depend
        return self

    def sample(self, X, y):
        if isinstance(X, pd.DataFrame):
            self.pd_index = X.index.values
            self.pd_columns = X.columns
            self.pd_yname = y.name
            X = X.values
            y = y.values

        return super().sample(X, y)

    def _sample(self, X, y):
        n, m = X.shape
        dtype = X.dtype
        offset = self.offset
        n_jobs = self.n_jobs
        lag0 = self.lag
        n_iter = self.n_iter
        X_resampled = np.zeros((0, m), dtype=dtype)
        y_resampled = np.array([])

        X_di = X

        continuous = self._get_countinuous(X_di)
        if continuous:
            X_di = self.disc.transform(X)

        for class_sample, n_samples in self.ratio_.items():
            if n_samples < self.threshold:
                continue

            probs = [df.to_dict() for df in self.probs[class_sample]]
            priors = self.priors[class_sample]
            depend = self.structure[class_sample]

            step = self.ign_step

            X_class = X_di[y == class_sample]
            count = 0
            if n_iter == 'auto':
                n_iter = offset + ((X_di.shape[0] - 1.6 * X_class.shape[0]) / X_class.shape[0]) * lag0
                n_iter = int(np.round(n_iter))

            if self.verbose > 0:
                print('################################')
                print('Structure:', self.structure[class_sample])
                print('Number of original samples:', X_class.shape[0])
                print('Number of new samples:', n_samples)
                print('Number of iterations:', n_iter)

            val_list = self._get_vlist(priors)
            samples_per_gibbs = int((n_iter - offset) / lag0)
            n_gibbs = int(np.ceil(n_samples / samples_per_gibbs))
            max_spread = X_class.shape[0] // n_gibbs
            if max_spread < 1:
                max_spread = 1
                # ADD warning
            if (step == 'auto') or (step > max_spread):
                step = max_spread

            if self.verbose > 1:
                print('Samples from one Gibbs sampler:', samples_per_gibbs)
                print('Total number of Gibbs samplers:', n_gibbs)
                print('Spread:', max_spread)
            X_new = self._multi_run(X_class=X_class, vlist=val_list,
                                    depend=depend, probs=probs,
                                    priors=priors, n_iter=n_iter,
                                    n_gibbs=n_gibbs, spread=step)

            continuous = self._get_countinuous(X_new)
            if continuous:
                X_new = self._recon_continuous(X, y, X_di, X_new, class_sample, self.i_categorical,
                                               self.alpha, self.L)
            y_new = np.ones(X_new.shape[0]) * class_sample
            X_resampled = np.vstack((X_resampled,
                                     X_new[:n_samples]))

            y_resampled = np.hstack((y_resampled, y_new[:n_samples])) if y_resampled.size else y_new[:n_samples]
            count = count + 1

        if self.verbose > 0:
            print('Original shapes:', X.shape, y.shape)

        if self.only_sampled is False:
            if self.pdflag:
                index = self.pd_index[-1] + np.arange(X_resampled.shape[0])
                X_resampled, y_resampled = self.create_pandas(X_resampled, y_resampled, index=index)
                X, y = self.create_pandas(X, y)

                X_resampled = pd.concat((X, X_resampled), axis=0)
                y_resampled = pd.concat((y, y_resampled))
            else:
                X_resampled = np.vstack((X,
                                         X_resampled))

                y_resampled = np.hstack((y,
                                         y_resampled))
        else:
            if self.pdflag:
                print('flag', pdflag)
                index = self.pd_index[-1] + np.arange(X_resampled.shape[0])
                X_resampled, y_resampled = self.create_pandas(X_resampled, y_resampled, index=index)
            else:
                pass
        if self.verbose > 0:
            print('Resampled shapes:', X_resampled.shape, y_resampled.shape)

        if self.shuffle is True:
            X_resampled, y_resampled = self.shuffle_rows(X_resampled, y_resampled, num=5)

        return X_resampled, y_resampled

    def _recon_continuous(self, X, y, X_disc, X_sampled, class_sample, categorical, alpha=0.6, L=0.5):
        """
        Construct continuous features from categorical ones
        """
        if self.verbose > 0:
            print('')
            print('Restore continuous variable')
        X_recon = X_sampled.copy()
        for j in range(X_sampled.shape[1]):
            if j in categorical:
                continue

            for i in range(X_sampled.shape[0]):
                # xj = X[:, j]
                xj = np.take(X, j, axis=1)
                xj_di = np.take(X_disc, j, axis=1)

                mu0 = xj.mean()
                std0 = xj.std()

                val = X_sampled[i, j]
                idx_val = np.where(xj_di == val)[0]
                xjc = xj[idx_val]

                mu1 = xjc.mean()
                std1 = xjc.std()
                restor = None
                _mu_ = mu0
                _std_ = std0

                if mu1 / mu0 > alpha:
                    _mu_ = mu1
                    _std_ = std1

                if self.random_state is not None:
                    np.random.seed(self.random_state)
                restor = self.f_sample(_mu_, L * _std_)
                X_recon[i, j] = restor
        return X_recon

    def _multi_run(self, X_class, vlist, depend, probs, priors, n_iter, n_gibbs, spread):
        """
        Run Gibbs samplers in parallel
        """
        m = X_class.shape[1]
        n_jobs = self.n_jobs
        if self.verbose > 0:
            print('################################')
            print('Sampling...')
            print('n_jobs:', n_jobs)

        params = {'vlist': vlist,
                  'depend': depend,
                  'probs': probs,
                  'priors': priors,
                  'T': n_iter}
        X_new = np.zeros((0, m))
        f_gibbs = partial(self._gibbs_sampler, **params)
        zi = []
        p = multiprocessing.Pool(n_jobs)
        gibbs_count = n_gibbs
        for j in tqdm(range(0, X_class.shape[0], spread)):
            if gibbs_count == 0:
                continue
            zi.append(X_class[j].tolist())
            gibbs_count = gibbs_count - 1
            if (len(zi) == n_jobs) or (gibbs_count == 0):
                res = p.map(f_gibbs, zi)

                for s in res:
                    X_new = np.vstack((X_new, np.array(s)))
                zi = []

        return X_new

    def _all_keys(self, X, eps=0.00001):
        """
        Get all possible values for each feature
        """
        all_dicts = np.empty(X.shape[1], dtype='object')
        for j in range(X.shape[1]):
            keys = np.unique(X[:, j])
            n = keys.shape[0]
            fi = np.zeros(n) + eps
            all_dicts[j] = dict(zip(keys, fi))
        return all_dicts

    def _get_structure(self, X_plus, root=0):
        """
        Get the features dependency structure of the minority class
        """
        bayes = BayesianNetwork.from_samples(X_plus, algorithm='chow-liu', root=root)
        depend = []
        for i in bayes.structure:
            if i:
                depend.append(i[0])
            else:
                depend.append(-1)
        return depend

    def _fill_priors(self, X_plus, all_dicts):
        """
        Get priors of the minority class for each feature
        """
        N = X_plus.shape[0]
        for j in range(X_plus.shape[1]):
            keys, counts = np.unique(X_plus[:, j], return_counts=True)
            t = dict(zip(keys, counts / N))
            all_dicts[j].update(t)
        return all_dicts

    def _tan(self, X, y, X_plus):
        """
        Construct prior and dependency tables of the minority class from Chow-Liu dependence tree
        """
        allkeys = self._all_keys(X, eps=self.eps)
        priors = self._fill_priors(X_plus, allkeys)
        depend = self._get_structure(X_plus, root=self.root)

        probs = []  # np.empty(X.shape[1], dtype='object')
        for j in range(len(depend)):
            if depend[j] == -1:
                ival = list(priors[j].keys())
                dfp = pd.DataFrame(np.zeros((1, len(ival))), columns=ival, index=['root'])

                for k in ival:
                    dfp[k] = priors[j][k]
                probs.append(dfp.copy())
                continue
            num_parent = depend[j]
            ival = list(priors[j].keys())
            dval = list(priors[num_parent].keys())

            lend = len(dval)
            leni = len(ival)
            dfp = pd.DataFrame(np.zeros((lend, leni)), columns=ival, index=dval)
            dfp.index = dval
            dfp.columns = ival
            for row in dval:
                for col in ival:
                    numi = np.where((X_plus[:, j] == col) & (X_plus[:, num_parent] == row))[0].shape[0]
                    den = np.where(X_plus[:, num_parent] == row)[0].shape[0]
                    if den == 0:
                        dfp.loc[row, col] = self.eps
                    else:
                        dfp.loc[row, col] = numi / den
            probs.append(dfp.copy())
        return probs, priors, depend

    def _gibbs_sampler(self, zi, vlist, depend, probs, priors, T):
        """
        Gibbs sampler
        """
        lag0 = self.lag
        offset = self.offset
        Zi = zi[:]
        sample = []
        count = 0
        for t in range(T):
            for j in range(len(vlist)):
                ival = vlist[j]
                P = np.zeros(len(ival))
                # num_pr = self._numfactor_part(j, Zi, depend, probs)
                for i in range(len(ival)):
                    Zi[j] = ival[i]
                    # p = self._numfactor_n(j, Zi, depend, probs)
                    p = self._numfactor(Zi, depend, probs)
                    # P[i] = num_pr * p
                    P[i] = p
                    count = count + 1

                P = P / P.sum()
                if self.random_state is not None:
                    np.random.seed(self.random_state + count)
                Zi[j] = np.random.choice(ival, 1, p=P)[0]
            if t > offset and t % lag0 == 0:
                sample.append(Zi[:])

        return sample

    def _numfactor_part(self, i, zi, depend, probs):
        prob = 1
        for j in range(len(depend)):
            if j == i:
                continue
            cur_val = zi[j]
            dep = depend[j]
            if dep == -1:
                pr = probs[j][cur_val]['root']
            else:
                pr = 1
                dep_val = zi[dep]
                pr = probs[j][cur_val][dep_val]
            prob = prob * pr

        return prob

    def _numfactor_n(self, i, zi, depend, probs):
        prob = 1
        j = i

        cur_val = zi[j]
        dep = depend[j]
        if dep == -1:
            pr = probs[j][cur_val]['root']
        else:
            pr = 1
            dep_val = zi[dep]
            pr = probs[j][cur_val][dep_val]
        prob = prob * pr

        return prob

    def _numfactor(self, zi, depend, probs):
        """
        Calculate joint probability
        """
        prob = 1
        for j in range(len(depend)):
            cur_val = zi[j]
            dep = depend[j]
            if dep == -1:
                pr = probs[j][cur_val]['root']
            else:
                pr = 1
                dep_val = zi[dep]
                pr = probs[j][cur_val][dep_val]
            prob = prob * pr

        return prob

    def shuffle_rows(self, X, y, num=5, random_state=None):
        """
        Shuffle rows of input arrays
        """
        if isinstance(X, pd.DataFrame):
            new_index = X.index.values
            for i in range(num):
                if random_state is not None:
                    np.random.seed(random_state + i)
                new_index = np.random.permutation(new_index)
            X.reindex(new_index)
            y.reindex(new_index)
        else:
            for i in range(num):
                if random_state is not None:
                    np.random.seed(random_state + i)
                new_index = np.random.permutation(new_index)
                X = X[new_index]
                y = y[new_index]
        return X, y

    def _get_countinuous(self, X):
        t = np.arange(X.shape[1]).tolist()
        co = []
        for i in t:
            if i in self.i_categorical:
                continue
            co.append(i)
        return co

    def _get_vlist(self, priors):
        val_list = []
        eps = self.eps

        for j in range(len(priors)):
            pr = priors[j]
            k = list(pr.keys())
            one = [i for i in k if pr[i] > eps]
            val_list.append(one)
        return val_list

    def check_categorical(self, X, y):
        categorical = []
        ny2 = 2 * np.unique(y).shape[0]
        for j in range(X.shape[1]):
            xj = X[:, j]
            xj = xj[np.invert(np.isnan(xj))]
            if np.unique(xj).shape[0] < ny2:
                categorical.append(j)
        return categorical

    def create_pandas(self, X, y, index=None, columns=None, yname=None):
        """
        Create pandas dataframe
        """
        if index is None:
            index = self.pd_index
        if columns is None:
            columns = self.pd_columns
        if yname is None:
            yname = self.pd_yname
        X = pd.DataFrame(X, columns=columns, index=index)
        y = pd.Series(y, index=index, name=yname)
        return X, y


class WrongDistributionException(Exception):
    # Raise if wrong type of Distribution
    pass


class TooManyValues(Exception):
    # Raise if additional binning is needed
    pass
