# Contributing to Avocet

Thanks for your interest in improving Avocet!

## How to contribute
- **Issues:** File an issue describing bugs or feature requests. Include repro steps, versions, and expected vs actual behavior.
- **Pull requests:** Fork, create a feature branch, and open a PR. Keep changes focused; add tests when possible. Note any breaking changes.
- **Code style:** Use clear, concise Python. Keep new dependencies minimal. Prefer numpy/torch/cvxpy idioms already used in the codebase.
- **Testing:** Run `pip install -e .[dev]` then `pytest`. Please ensure new functionality has coverage.
- **Docs/Examples:** If adding APIs, update README/examples where appropriate.

## Development setup
```
pip install -e .[dev]
pytest
```

## Communication
For questions, open an issue or start a discussion on the repo. PR reviews welcome!
