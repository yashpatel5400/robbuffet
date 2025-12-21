# Publishing to PyPI

1. Bump the version in `pyproject.toml` and add a changelog entry.
2. Build artifacts from a clean environment:
   ```bash
   python -m pip install --upgrade build twine
   python -m build
   # dist/ now contains robbuffet-<ver>.tar.gz and robbuffet-<ver>-py3-none-any.whl
   ```
3. Upload with a PyPI token:
   ```bash
   export PYPI_TOKEN="pypi-XXXXXXXXXXXXXXXXXXXX"
   twine upload -u __token__ -p "$PYPI_TOKEN" dist/*
   ```
4. Verify the release:
   ```bash
   pip install --no-cache-dir robbuffet==<ver>
   python - <<'PY'
   import robbuffet
   print(robbuffet.__version__)
   PY
   ```

For TestPyPI, replace the upload command with:
```
twine upload --repository testpypi dist/*
```
