# Contributing to E-QFT

Thank you for your interest in contributing to E-QFT! This document provides guidelines for contributions.

## How to Contribute

1. **Fork the repository** and create your branch from `main`
2. **Write tests** for any new functionality
3. **Ensure tests pass** by running `python run_all_tests.py`
4. **Follow the code style** (use black for formatting)
5. **Submit a pull request** with a clear description

## Code Style

- Use Python 3.8+ features
- Follow PEP 8 guidelines
- Use type hints where appropriate
- Add docstrings to all functions/classes
- Use black for automatic formatting: `black .`

## Testing

All contributions must include appropriate tests:
- Unit tests for individual functions
- Integration tests for new features
- Performance benchmarks for optimizations

## Physics Contributions

If contributing new physics:
- Provide theoretical justification
- Include convergence tests
- Compare with known results where possible
- Document any approximations

## Reporting Issues

When reporting issues, please include:
- Python version and OS
- Complete error traceback
- Minimal reproducible example
- Expected vs actual behavior

## Questions?

Feel free to open an issue for questions or discussions.
