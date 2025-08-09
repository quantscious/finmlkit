# Contribution Guide

FinMLKit thrives on community involvement. We value **contribution**, **transparency**, **documentation**, and **thorough testing**. This guide explains how to get started and how to make your contribution count.

## Core Principles
- **Open Collaboration** – we welcome ideas, questions and improvements from everyone.
- **Transparency** – discussions, design decisions and code should be easy to follow.
- **Documentation First** – every feature or change should include clear docs and examples.
- **Test Rigorously** – changes must include tests that cover success and failure cases.

## Getting Started
1. Fork the repository and clone your fork.
2. Install the package in editable mode:
   ```bash
   pip install -e .
   ```
3. Install development dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Create a new branch for your work. *(Avoid pushing to `main` directly).* 

## How to Contribute
Choose the path that matches your idea:

### 🐞 Bug Fixes
- Check existing issues to avoid duplication.
- Write a failing test that reproduces the bug.
- Fix the bug and ensure tests pass.

### 🌱 New Features
- Open an issue to discuss the idea before coding.
- Provide minimal, well-documented APIs.
- Include usage examples in the documentation and tests.

### 🔧 Enhancements
- Propose improvements to existing functionality in an issue.
- Keep backward compatibility in mind.

### 📚 Documentation Improvements
- Improve docstrings, tutorials, or README sections.
- Follow the reStructuredText style used throughout the project.

### 🧪 Testing
- Add or expand tests for any change.
- Run the test suite from the project root. For a full run, disable JIT:
  `NUMBA_DISABLE_JIT=1 pytest -q`.
- Refer to [tests/README.md](tests/README.md) for details and helper scripts
  (`./local_test.sh` and `./local_test_nojit.sh`).

## Coding Standards
- Follow [PEP 8](https://peps.python.org/pep-0008/) and existing project style.
- Use descriptive commit messages and keep commits focused.
- Typing is encouraged; prefer explicit type hints.

## Pull Request Process
1. Ensure your branch is up to date with `main`.
2. Run tests locally: `NUMBA_DISABLE_JIT=1 pytest -q` (see [tests/README.md](tests/README.md)).
3. Update documentation and include examples as needed.
4. Fill out the PR template, describing the change and any breaking behaviour.
5. Link relevant issues in the PR description.

## Community
Questions or ideas? Open an issue or start a discussion. Respectful communication is expected at all times.

Thank you for helping make FinMLKit better!
