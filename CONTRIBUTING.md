# Contributing to DSS Pollution Extraction

We welcome contributions to the DSS Pollution Extraction project! This document provides guidelines for contributing.

## Getting Started

1. Fork the repository
2. Clone your fork locally
3. Set up the development environment:
   ```bash
   ./scripts/setup_dev.sh
   ```

## Development Workflow

1. Create a new branch for your feature or bugfix
2. Make your changes
3. Write or update tests as needed
4. Run the test suite:
   ```bash
   ./scripts/test.sh
   ```
5. Ensure code quality:
   ```bash
   black pollution_extraction/
   flake8 pollution_extraction/
   mypy pollution_extraction/
   ```
6. Commit your changes with a descriptive message
7. Push to your fork and create a pull request

## Code Standards

- Follow PEP 8 style guidelines
- Use type hints where possible
- Write docstrings for all public functions and classes
- Maintain test coverage above 80%

## Pull Request Process

1. Update documentation if needed
2. Add an entry to CHANGELOG.md
3. Ensure all tests pass
4. Request review from maintainers

## Reporting Issues

- Use the GitHub issue tracker
- Provide clear reproduction steps
- Include system information and error messages

Thank you for contributing!
