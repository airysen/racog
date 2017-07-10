RACOG
=====

# A Python implementation of Rapidy Converging Gibbs sampler<sup>1,2</sup> for data oversampling with CAIM<sup>3</sup> and MDLP<sup>4</sup> discretization methods


Reference
----------
[1]*B. Das, N. C. Krishnan and D. J. Cook, "RACOG and wRACOG: Two Probabilistic Oversampling Techniques,"in IEEE Transactions on Knowledge and Data Engineering,vol. 27, no. 1, pp. 222-234, Jan. 1 2015. doi: 10.1109/TKDE.2014.2324567*
[http://ieeexplore.ieee.org/document/6816044/](http://ieeexplore.ieee.org/document/6816044/)

[2] [https://github.com/barnandas/DataSamplingTools](https://github.com/barnandas/DataSamplingTools)

[3][https://github.com/airysen/caimcaim](https://github.com/airysen/caimcaim)

[4][https://github.com/airysen/mdlp](https://github.com/airysen/mdlp)


Installation
-------------

Requirements:

 * [pomegranate](https://github.com/jmschrei/pomegranate)
 * [pandas](http://pandas.pydata.org/)
 * [imblearn](https://github.com/scikit-learn-contrib/imbalanced-learn)
 * [sklearn](scikit-learn.org)
 * [caimcaim](https://github.com/airysen/caimcaim)
 * [mdlp](https://github.com/airysen/mdlp)
 * [tqdm](https://pypi.python.org/pypi/tqdm)


 Example of usage
------------------

```python
>>> from racog import RACOG
>>> from sklearn.datasets import make_classification
>>> X, y = make_classification(n_samples=2000, n_features=7, n_redundant=2,
>>>                            n_informative=4, weights=[0.05, 0.95], n_classes=2)
                           
>>> racog = RACOG(discretization='caim', categorical_features='auto',
>>>               warmup_offset=100, lag0=20, n_iter='auto',
>>>               continous_distribution='normal', random_state=None,
>>>               alpha=0.6, L=0.5, threshold=10, eps=10E-5, verbose=2, n_jobs=1)

>>> X_res, y_res = racog.fit_sample(X, y)

```
