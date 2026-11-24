# Release Process

Releases are published automatically when a version tag is pushed.

## One-time PyPI setup

In the PyPI PyNAFF project, add a Trusted Publisher with:

- Owner: `PZiso`
- Repository: `PyNAFF`
- Workflow: `publish.yml`
- Environment: `pypi`

In GitHub, create the `pypi` environment under repository settings.

## Release

1. Merge the feature branches into `master` through pull requests, after the
   `Tests` workflow passes.
2. On `master`, set the new version in `PyNAFF/_version.py` and rename the
   `## Unreleased` section of `CHANGELOG.md` to `## <version>`. Commit and push.
3. Tag and push:

   ```bash
   git switch master
   git pull --ff-only origin master
   git tag -a v1.3.0 -m "PyNAFF 1.3.0"
   git push origin v1.3.0
   ```

Pushing the tag starts `publish.yml`. It checks that the tag matches
`PyNAFF/_version.py` and that `CHANGELOG.md` has a section for the version,
runs the tests, uploads to PyPI, and then creates the GitHub release with the
notes of that version. If the upload fails, no release is created; fix the
cause and re-run the failed jobs from the Actions tab.

The workflow can also be started by hand from the Actions tab
(`Run workflow`). On a branch it uploads without creating a GitHub release.

## Local checks (optional)

```bash
python -m unittest discover -s tests -v
python -m build
python -m twine check dist/*
```
