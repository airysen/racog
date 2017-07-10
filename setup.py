from setuptools import setup, find_packages

setup(name='racog',
      version='0.2',
      description='RACOG',
      long_description='Rapidy Converging Gibbs sampler for data oversampling',
      classifiers=[
          'Development Status :: 3 - Alpha',
          'License :: OSI Approved :: MIT License',
          'Programming Language :: Python :: 3.6',
          'Topic :: Scientific/Engineering :: Information Analysis',
          'Operating System :: OS Independent',
          'Intended Audience :: Science/Research',
      ],
      url='https://github.com/airysen/racog',
      author='Arseniy Kustov',
      author_email='me@airysen.co',
      license='MIT',
      packages=find_packages(),
      install_requires=['mdlp', 'caimcaim', 'tqdm', 'pandas', 'pomegranate', 'imblearn', 'numpy', 'sklearn'],
      include_package_data=True,
      zip_safe=False)
