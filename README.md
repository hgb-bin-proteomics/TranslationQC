![Ruff](https://github.com/michabirklbauer/python-pkg_template/workflows/Ruff%20Lint%20and%20Format/badge.svg)
![Ty](https://github.com/michabirklbauer/python-pkg_template/workflows/Type-check%20with%20ty/badge.svg)
![Pyright](https://github.com/michabirklbauer/python-pkg_template/workflows/Type-check%20with%20Pyright/badge.svg)
![Pytest](https://github.com/michabirklbauer/python-pkg_template/workflows/Test%20with%20pytest/badge.svg)

# Template Repository for Python Packages

A template repository for modern python package development with [uv](https://docs.astral.sh/uv/)
using [Pydantic](https://pydantic.dev/docs/validation/latest/get-started/) and 
[Polars](https://pola.rs/).
Linted with [ruff](https://astral.sh/ruff),
type checked with [ty](https://docs.astral.sh/ty/) and [pyright](https://microsoft.github.io/pyright),
and tested with [pytest](https://docs.pytest.org/en/stable/) using
[GitHub Actions](https://docs.github.com/en/actions).

## Checklist

- [ ] Replace `YOURUSERNAME` and `IMAGENAME` in `.github/workflows/docker-image.yml` [or delete file].
- [ ] Replace test data in `data` with your own data [or delete if you don't have test data].
- [ ] Adjust `.gitignore` according to your needs.
- [ ] Setup your `CITATION.cff` according to your needs [or delete file].
- [ ] Update attribution in `Dockerfile` and write image instructions.
- [ ] Replace copyright name in `LICENSE`.
- [ ] Update attributions and package configuration in `pyproject.toml`.
- [ ] Update attributions and write your package in `src/name_of_your_package`.
- [ ] Update attributions and write tests in `tests/`.
- [ ] Add your requirements via `uv add`.
- [ ] Document your code using the [numpydoc style](https://numpydoc.readthedocs.io/en/latest/format.html) and [Sphinx](https://www.sphinx-doc.org/):
  - Adjust the configuration to your needs in `docs/conf.py`.
  - Automatically via GitHub Actions:
    - In the file `.github/workflows/gh-pages.yml` in line `55` replace `src/python_pkg_template` with `src/name_of_your_package`.
      For example, the full line should read as:
      ```bash
      uv run sphinx-apidoc -f -o docs src/name_of_your_package
      ```
    - In the repository go to `Settings` ➡️ `Pages` ➡️ `Build and deployment` ➡️ `Source` ➡️ `GitHub Actions`.
    - Select the `gh-pages.yml` / `Deploy Documentation to Pages` workflow.
  - Or build manually:
    - Install documentation dependencies via:
      ```bash
      uv sync --group docs
      ```
    - Build documentation with:
      ```bash
      uv run sphinx-apidoc -f -o docs src/python_pkg_template
      ```
      which eventually should be replaced with
      ```bash
      uv run sphinx-apidoc -f -o docs src/name_of_your_package
      ```
      and then run
      ```bash
      uv run sphinx-build -b html docs html
      ```
    - Publish documentation [optional]!
    - Serving with GitHub pages needs the addition of an empty `.nojekyll` file to your `/html`.
- [ ] Decide on a type checker and delete the other!
- [ ] Adjust this `README.md` to your needs!

## Package Deployment and Workflows

Please consider reading the [Python Packaging User Guide](https://packaging.python.org/)!

> [!IMPORTANT]
> In order to publish your python package you need to create an account on [PyPI](https://pypi.org/) and
> ideally also [TestPyPI](https://test.pypi.org/)
> for [staging](https://en.wikipedia.org/wiki/Deployment_environment#Staging).

- [ ] Setup [trusted publishing](https://packaging.python.org/en/latest/guides/publishing-package-distribution-releases-using-github-actions-ci-cd-workflows/#configuring-trusted-publishing).
- [ ] Replace `<NAME-OF-YOUR-PACKAGE>` with the name of your package in
  - [ ] ...line `37` of `.github/workflows/publish-testpypi.yml`.
  - [ ] ...line `37` of `.github/workflows/publish-pypi.yml`.
  - [ ] ...line `28` of `.github/workflows/test-testpypi.yml`.
  - [ ] ...line `28` of `.github/workflows/test-pypi.yml`.

### Staging and Publishing

Whenever you have a new version of your package ready, go to the `Actions` tab in your GitHub repository and run the following workflows:

- [ ] Run `Publish Python Package to TestPyPI` (workflow file: `publish-testpypi.yml`).
- [ ] Check that the workflow successfully completes!
- [ ] Run `Run tests on package from TestPyPI` (workflow file: `test-testpypi.yml`).
- [ ] Check that the workflow successfully completes!
- [ ] Run `Publish Python Package to PyPI` (workflow file: `publish-pypi.yml`).
- [ ] Check that the workflow successfully completes!
- [ ] Run `Run tests on package from PyPI` (workflow file: `test-pypi.yml`).
- [ ] Check that the workflow successfully completes!
- [ ] Done! Your package was successfully published to _TestPyPI_ and _PyPI_!

> [!NOTE]
> You can omit TestPyPI but staging is generally good practice!

## Helpful Commands

- [uv](https://docs.astral.sh/uv/):
  - Add a dependency/package [`pkg`]:
    ```bash
    uv add pkg
    ```
  - Upgrade dependencies/packages:
    ```bash
    uv lock --upgrade
    ```
  - Update environment:
    ```bash
    uv sync
    ```
  - Run python:
    ```bash
    uv run python
    ```
  - Run a package script (with arguments):
    ```bash
    uv run battle -h
    ```
- [ruff](https://docs.astral.sh/ruff/):
  - Check and fix (fixable) errors:
    ```bash
    uv run ruff check --config ruff.toml --fix
    ```
  - Check (with explicit config file):
    ```bash
    uv run ruff check --config ruff.toml
    ```
  - Format code:
    ```bash
    uv run ruff format
    ```
  - Format (with explicit config file):
    ```bash
    uv run ruff format --config ruff.toml
    ```
- [ty](https://docs.astral.sh/ty/):
  ```bash
  uv run ty check --config-file ty.toml
  ```
- [pyright](https://microsoft.github.io/pyright):
  ```bash
  uv run pyright --warnings
  ```
- [pytest](https://docs.pytest.org/en/stable/):
  ```bash
  uv run pytest -c pytest.ini --durations=10 --durations-min=1.0 tests/
  ```

## Getting Help

- Help for this template:
  - [uv](https://docs.astral.sh/uv/): Python project and dependency management.
  - [ruff](https://docs.astral.sh/ruff/): Python linter and formatter.
  - [ty](https://docs.astral.sh/ty/): Python type checker.
  - [pyright](https://microsoft.github.io/pyright): Python type checker.
  - [pytest](https://docs.pytest.org/en/stable/): Python testing suit.
  - [GitHub Actions](https://docs.github.com/en/actions): Used for running the above automatically.
- Contact: [micha.birklbauer@gmail.com](mailto:micha.birklbauer@gmail.com)

> [!IMPORTANT]
> The below sections should be adjusted and updated by you!

## Known Issues

[List of known issues](https://github.com/michabirklbauer/python-pkg_template/issues)

## Citing

If you are using PLACEHOLDER please cite:
```
Very important title
Important Author, and Another Important Author
Journal of Cool Stuff 2023 12 (3), 4567-4589
DOI: 12.3456/cool-stuff
```

## License

- [MIT](https://github.com/michabirklbauer/python-pkg_template/blob/master/LICENSE)

## Contact

- [your.mail@mail.com](mailto:your.mail@mail.com)
