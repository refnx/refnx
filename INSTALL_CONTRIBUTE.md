# refnx - Installation and Development Instructions

refnx is a python package for analysis of neutron and X-ray reflectometry data.
It can also be used as a generalised curvefitting tool. It uses Markov Chain
Monte Carlo to obtain posterior distributions for curvefitting problems.

---

# Installation

*refnx* is tested on Python 3.11+ (see `requires-python` in `pyproject.toml`
for the exact floor). Core installation requires *numpy* and *scipy*.
Additional features (interactive plotting, motofit GUI, PyMC-based Bayesian
sampling, Jupyter widgets, etc.) require extra packages, all available via
the `all` optional-dependency group — see the `[project.optional-dependencies]`
table in `pyproject.toml` for the full list.

To build *refnx* from source you'll need a C compiler, because the build
compiles Cython extensions. C compilers are normally already installed on
Linux. On macOS you'll need Xcode and its command line tools. On Windows
you'll need a suitable [Visual Studio compiler](https://wiki.python.org/moin/WindowsCompilers)
for your Python version.

*refnx* vendors its own copies of the *emcee* and *ptemcee* packages
(under `refnx/_lib/`), so you don't need to install those separately.

## Installing a released version

The simplest way to install refnx is from PyPI:

```
pip install refnx
```

or from conda-forge:

```
conda install -c conda-forge refnx
```

To pull in all optional dependencies (GUI, plotting, Bayesian sampling, etc.):

```
pip install refnx[all]
```

Then check that the tests pass:

```python
>>> import refnx
>>> refnx.test()
```

## Building the bleeding-edge version from source

*refnx* uses a [Meson](https://mesonbuild.com/) build backend
(`meson-python`), configured via `meson.build` and `pyproject.toml`.

1. Clone the repository (or your fork, see "Setting up a local git
   repository" below):

   ```
   git clone https://github.com/refnx/refnx.git
   cd refnx
   ```

2. Create and activate a virtual environment (conda or venv both work). For
   example, with conda:

   ```
   conda create -n refnx python=3.12
   conda activate refnx
   ```

3. Install refnx in editable mode. `meson-python` supports an editable
   install that rebuilds Cython/C extensions automatically as you change
   source files, which is the recommended way to develop refnx:

   ```
   pip install --no-build-isolation --editable ".[all,test]"
   ```

   If you'd rather not use editable installs (e.g. for a one-off build), a
   regular install also works:

   ```
   pip install ".[all,test]"
   ```

   `pip` will pull in the build-time dependencies declared in
   `[build-system]` (`meson`, `meson-python`, `cython`, `numpy`) automatically.

4. Run the test suite with `pytest`:

   ```
   pytest --pyargs refnx
   ```

   or, from within the source checkout:

   ```
   pytest
   ```

## Code style and linting

The project uses [black](https://github.com/psf/black) (line length 79) and
[ruff](https://github.com/astral-sh/ruff) for formatting/linting, configured
in `pyproject.toml`. If a `.pre-commit-config.yaml` is present, install
[pre-commit](https://pre-commit.com/) so these run automatically on commit:

```
pip install pre-commit
pre-commit install
```

---

## Development Workflow

These instructions outline the workflow for contributing to refnx development.
The refnx community welcomes all contributions that will improve the package.
The following instructions are based on use of a command line *git* client. *Git* is a distributed version control program. An example of [how to contribute to the numpy project](https://numpy.org/doc/stable/dev/index.html) is a useful reference.

### Setting up a local git repository

1. Create an account on [github](https://github.com/).
2. On the [refnx github](https://github.com/refnx/refnx) page fork the *refnx* repository to your own github account. Forking means that now you have your own personal repository of the *refnx* code.
3. Now we will make a local copy of your personal repository on your local machine:

```
# <username> is your github username
git clone https://github.com/<username>/refnx.git
```

4. Add the *refnx* remote repository, we're going to refer to the remote with the *upstream* name:

```
git remote add upstream https://github.com/refnx/refnx.git
```

5. List the remote repositories that your local repository knows about:

```
git remote -v
```

### Keeping your local and remote repositories up to date

The main *refnx* repository may be a lot more advanced than your fork, or your local copy, of the git repository.

1. To update your repositories you need to fetch the changes from the main *refnx* repository:

```
git fetch upstream
```

2. Now update the local branch you're on by rebasing against the *refnx* main branch:

```
git rebase upstream/main
```

3. Push your updated local branch to the remote fork on github. You have to specify the remote branch you're pushing to. Here we push to the *main* branch:

```
git push origin main
```

### Adding a feature

The git repository is automatically on the main branch to start with. However,
when developing features that you'd like to contribute to the *refnx* project
you'll need to do it on a feature branch.

1. Create a feature branch and check it out:

```
git branch my_feature_name
git checkout my_feature_name
```

2. Once you're happy with the changes you've made you should check that the tests still work:

```
pytest
```

3. If the performance of what you've added/changed may be critical, then consider writing a benchmark. The benchmarks use
the *asv* package and are run as:

```
cd benchmarks
pip install asv
asv run
asv publish
asv preview
```

For an example benchmark look at one of the files in the *benchmarks* directory.

4. Now commit the changes. You'll have to supply a commit message that outlines the changes you made. The commit message should follow the [numpy guidelines](https://numpy.org/doc/stable/dev/development_workflow.html#writing-the-commit-message).

```
git commit -a
```

5. Now you need to push those changes on the *my_feature_branch* branch to *your* fork of the refnx repository on github:

```
git push origin my_feature_branch
```

6. On the main [refnx](https://github.com/refnx/refnx) repository you should be able to create a pull request (PR). The PR says that you'd like the *refnx* project to include the changes you made.
7. Once the automated tests have passed, and the *refnx* maintainers are happy with the changes you've made then the PR is merged. You can then delete the feature branch on github, and delete your local feature branch:

```
git branch -D my_feature_branch
```
