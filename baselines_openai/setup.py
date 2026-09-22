import re
from setuptools import setup, find_packages
import sys

if sys.version_info.major != 3:
    print('This Python is only compatible with Python 3, but you are running '
          'Python {}. The installation will likely fail.'.format(sys.version_info.major))


extras = {
    'test': [
        'filelock',
        'pytest',
        'pytest-forked',
        'atari-py',
        'matplotlib',
        'pandas'
    ],
    'mpi': [
        'mpi4py'
    ]
}

all_deps = []
for group_name in extras:
    all_deps += extras[group_name]

extras['all'] = all_deps

setup(name='baselines',
      packages=[package for package in find_packages()
                if package.startswith('baselines')],
      # Trimmed for the RAISE fork: algorithm packages (ppo2, deepq, …) were
      # removed; only logger / bench / common.vec_env (+ transitive) remain.
      # opencv-python is optional (Atari wrappers only; CrowdSim never imports it).
      install_requires=[
          'gym>=0.15.4, <0.16.0',
          'scipy',
          'tqdm',
          'joblib',
          'cloudpickle',
      ],
      extras_require=extras,
      description='OpenAI baselines: high quality implementations of reinforcement learning algorithms',
      author='OpenAI',
      url='https://github.com/openai/baselines',
      author_email='gym@openai.com',
      version='0.1.6')


# ensure there is some tensorflow build with version above 1.4
# (importlib.metadata avoids pkg_resources, which is missing in modern
# isolated pip build envs / setuptools without setuptools' legacy extras)
try:
    from importlib.metadata import PackageNotFoundError, version as _pkg_version
except ImportError:  # pragma: no cover
    from importlib_metadata import PackageNotFoundError, version as _pkg_version

tf_pkg_version = None
for tf_pkg_name in ['tensorflow', 'tensorflow-gpu', 'tf-nightly', 'tf-nightly-gpu']:
    try:
        tf_pkg_version = _pkg_version(tf_pkg_name)
        break
    except PackageNotFoundError:
        pass
assert tf_pkg_version is not None, 'TensorFlow needed, of version above 1.4'
from distutils.version import LooseVersion
assert LooseVersion(re.sub(r'-?rc\d+$', '', tf_pkg_version)) >= LooseVersion('1.4.0')
