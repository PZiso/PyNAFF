# Release Process

PyPI currently has PyNAFF 1.1.7. This repository prepares version 1.2.0.

## One-time PyPI setup

In the PyPI PyNAFF project, add a Trusted Publisher with:

- Owner: `PZiso`
- Repository: `PyNAFF`
- Workflow: `publish.yml`
- Environment: `pypi`

In GitHub, create the `pypi` environment under repository settings.

## Release

Run the tests and build checks:

```bash
python -m unittest discover -s tests -v
python -m build
python -m twine check dist/*
```

Push the feature branch and merge it into `master` after CI passes:

```bash
git push -u origin feature/multi-bpm
```

After merging:

```bash
git switch master
git pull --ff-only origin master
git tag -a v1.2.0 -m "PyNAFF 1.2.0"
git push origin v1.2.0
```

Create a GitHub release from `v1.2.0`. Publishing that release triggers
`.github/workflows/publish.yml`, which builds and uploads to PyPI using Trusted
Publishing.
