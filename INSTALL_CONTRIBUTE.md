# refnx - Installation and Development Instructions

refnx is a python package for analysis of neutron and X-ray reflectometry data.
It can also be used as a generalised curvefitting tool. It uses Markov Chain
Monte Carlo to obtain posterior distributions for curvefitting problems.

--------------
# Installation

*refnx* has been tested on Python 3.5, 3.6 and 3.7. It requires the *numpy,
scipy, cython, pandas, emcee* packages to work. Additional features require the
*pytest, h5py, xlrd, uncertainties, ptemcee, tqdm, matplotlib, pymc3* packages.
To build the bleeding edge code you will need to have access to a C-compiler to
build a couple of Python extensions. C-compilers should be installed on Linux.
On OSX you will need to install Xcode and the command line tools. On Windows you
will need to install the correct [Visual Studio compiler][Visual-studio-compiler]
for your Python version.

In the current version of *refnx* the *emcee* package is vendored by *refnx*. That
is, *refnx* possesses it's own private copy of the package, and there is no need to
install the *emcee* package separately.

## Installation into a *conda* environment

Perhaps the easiest way to create a scientific computing environment is to use the
[miniconda][miniconda] package manager. Once *conda* has been installed the first
step is to create a *conda* environment.

### Creating a conda environment
 
  1) In a shell window create a conda environment and install the dependencies. The **-n** flag indicates that the environment is called *refnx*.
  
  ```conda create -n refnx python=3.7 numpy scipy cython pandas h5py xlrd pytest tqdm```
  2) Activate the environment that we're going to be working in:
  
  ```
  # on OSX
  conda activate refnx

  # on windows  
  conda activate refnx
  ```
  3) Install the remaining dependencies:
  ```pip install uncertainties ptemcee```
 
### Installing into a conda environment from source

 The latest source code can be obtained from either [PyPi][PyPi] or [Github][github-refnx]. You can also build the package from within the refnx git repository (see later in this document).
  1) In a shell window navigate into the source directory and build the package. If you are on Windows you'll need to start a Visual Studio command window.
  ```
  python setup.py build
  python setup.py install
  ```
  2) Run the tests, they should all work.
  ```
  python setup.py test
  ```

### Installing into a conda environment from a released version

  1) There are pre-built versions on *conda-forge*, but they're not necessarily at the bleeding edge:
  
  ```conda install -c conda-forge refnx```
  2) Start up a Python interpreter and make sure the tests run:
  ```
  >>> import refnx
  >>> refnx.test()
  ```
 
-----------------------
## Development Workflow
 
These instructions outline the workflow for contributing to refnx development.
The refnx community welcomes all contributions that will improve the package.
The following instructions are based on use of a command line *git* client.
*Git* is a distributed version control program. An example of [how to contribute to the numpy project][numpy-contrib]
is a useful reference.

### Setting up a local git repository 
  1) Create an account on [github](https://github.com/).
  2) On the [refnx github][github-refnx] page fork the *refnx* repository to your own github account. Forking means that now you have your own personal repository of the *refnx* code.
  3) Now we will make a local copy of your personal repository on your local machine:
  ```
  # <username> is your github username
  git clone https://github.com/<username>/refnx.git
  ```
  4) Add the *refnx* remote repository, we're going to refer to the remote with the *upstream* name:
  ```
  git remote add upstream https://github.com/refnx/refnx.git
  ```
  5) List the remote repositories that your local repository knows about:
  ```
  git remote -v
  ```

### Keeping your local and remote repositories up to date
The main *refnx* repository may be a lot more advanced than your fork, or your local copy, of the git repository. 
  1) To update your repositories you need to fetch the changes from the main *refnx* repository:
  ```
  git fetch upstream
  ```
  2) Now update the local branch you're on by rebasing against the *refnx* master branch:
  ```
  git rebase upstream/master
  ```
  3) Push your updated local branch to the remote fork on github. You have to specify the remote branch you're pushing to. Here we push to the *master* branch:
  ```
  git push origin master
  ```

### Adding a feature
The git repository is automatically on the master branch to start with. However,
when developing features that you'd like to contribute to the *refnx* project
you'll need to do it on a feature branch.

  1) Create a feature branch and check it out:
  ```
  git branch my_feature_name
  git checkout my_feature_name
  ```
  2) Once you're happy with the changes you've made you should check that the tests still work:
  ```
  python setup.py test
  ```
  3) If the performance of what you've added/changed may be critical, then consider writing a benchmark. The benchmarks use
  the *asv* package and are run as:
  ```
  cd benchmarks
  pip install asv
  asv run
  asv publish
  asv preview
  ```
  For an example benchmark look at one of the files in the *benchmarks* directory.
  4) Now commit the changes. You'll have to supply a commit message that outlines the changes you made. The commit message should follow the [numpy guidelines][numpy-contib]
  ```
  git commit -a
  ```
  5) Now you need to push those changes on the *my_feature_branch* branch to *your* fork of the refnx repository on github:
  ```
  git push origin my_feature_branch
  ```
  6) On the main [refnx][github-refnx] repository you should be able to create a pull request (PR). The PR says that you'd like the *refnx* project to include the changes you made.
  7) Once the automated tests have passed, and the *refnx* maintainers are happy with the changes you've made then the PR is merged. You can then delete the feature branch on github, and delete your local feature branch:
  ```
  git branch -D my_feature_branch
  ```

   [PyPi]: <https://pypi.python.org/pypi/refnx>
   [github-refnx]: <https://github.com/refnx/refnx>
   [Visual-studio-compiler]: <https://wiki.python.org/moin/WindowsCompilers>
   [miniconda]: <https://conda.io/docs/install/quick.html>
   [numpy-contrib]: <https://docs.scipy.org/doc/numpy/dev/>