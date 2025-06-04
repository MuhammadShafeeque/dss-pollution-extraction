# 🚀 Automating PyPI Package Releases with `uv`, `bumpver`, and `twine`

This guide walks you through the full process of versioning, building, and publishing a Python package to PyPI, using modern tools like `uv` for dependency management and `bumpver` for version control.

---

## ✅ 1. Prerequisites

* Python package already initialized and pushed to GitHub
* Using `uv` for dependency and virtual environment management
* Package structured using `pyproject.toml` (no `setup.py`)
* You have a valid [PyPI API token](https://pypi.org/manage/account/token/)

---

## 📁 2. Project Structure

### Before Automation

```
your_project/
├── your_package/
│   └── __init__.py          # Contains: __version__ = "0.1.0"
├── pyproject.toml           # Contains: version = "0.1.0"
├── README.md
└── (other files...)
```

### After Automation

```
your_project/
├── your_package/
│   └── __init__.py
├── pyproject.toml
├── .bumpver.toml            # bumpver config
├── release.sh               # automation script
├── .gitignore
├── README.md
├── dist/                    # auto-generated (ignored)
└── *.egg-info/              # auto-generated (ignored)
```

---

## 🧰 3. Install Required Tools

Install tools globally in your Dev Container:

```bash
uv pip install --system bumpver build twine
```

Or install into a virtual environment:

```bash
uv venv .venv
source .venv/bin/activate
uv pip install bumpver build twine
```

---

## ⚙️ 4. Add `.bumpver.toml` Configuration

Create `.bumpver.toml` in your project root:

```toml
[bumpver]
current_version = "0.1.0"
version_pattern = "MAJOR.MINOR.PATCH"

[bumpver.file_patterns]
"pyproject.toml" = ['version = "{version}"']
"your_package/__init__.py" = ['__version__ = "{version}"']
```

> 🔁 Replace `your_package` with the actual package folder name (e.g., `pollution_extractor`).

---

## 🛠 5. Create `release.sh` Script

Create `release.sh` in the project root:

```bash
#!/bin/bash
set -e

echo "🔼 Bumping patch version..."
bumpver update --patch  # Change to --minor or --major as needed

echo "🧹 Cleaning old builds..."
rm -rf dist/ *.egg-info

echo "📦 Building package..."
python -m build

echo "🚀 Uploading to PyPI..."

# Option A: Using environment variables
# twine upload -u $TWINE_USERNAME -p $TWINE_PASSWORD dist/*

# Option B: Using .pypirc (uncomment if using this)
twine upload dist/*
```

Make it executable:

```bash
chmod +x release.sh
```

---

## 🔐 6. Authentication Options

### 🔸 Option A: Environment Variables (Recommended)

Set your PyPI credentials:

```bash
export TWINE_USERNAME=__token__
export TWINE_PASSWORD=pypi-<your-api-token>
```

Use this version in `release.sh`:

```bash
twine upload -u $TWINE_USERNAME -p $TWINE_PASSWORD dist/*
```

You can also add the exports to your `.env` file inside the dev container or shell profile for persistence.

---

### 🔸 Option B: `.pypirc` File (Alternative)

Create `~/.pypirc`:

```ini
[distutils]
index-servers =
    pypi

[pypi]
username = __token__
password = pypi-<your-api-token>
```

Use the simpler command in `release.sh`:

```bash
twine upload dist/*
```

Don't forget to ignore the file in Git:

```bash
# .gitignore
.pypirc
dist/
*.egg-info/
```

---

## 🔄 7. Versioning Commands

Use `bumpver` to manually bump versions when needed:

| Version Change | Command                  |
| -------------- | ------------------------ |
| Patch          | `bumpver update --patch` |
| Minor          | `bumpver update --minor` |
| Major          | `bumpver update --major` |

---

## 🚀 8. Release Your Package

Whenever you're ready to publish:

```bash
./release.sh
```

This will:

* Bump the version (default: patch)
* Update `pyproject.toml` and `__init__.py`
* Build `.whl` and `.tar.gz`
* Upload to PyPI

---

## 🎉 9. Success!

Your package should now be live on [https://pypi.org/](https://pypi.org/).
You can repeat this process after each change by rerunning `./release.sh`.

---

## 📝 Optional: What to Commit to Git

```gitignore
# Ignore local build artifacts and secrets
.pypirc
dist/
*.egg-info/
```

Commit the following automation files to Git:

* `.bumpver.toml`
* `release.sh`

---

## ✅ Example Package Reference

If your package is named `pollution_extractor`, your `.bumpver.toml` will look like this:

```toml
[bumpver]
current_version = "0.1.0"
version_pattern = "MAJOR.MINOR.PATCH"

[bumpver.file_patterns]
"pyproject.toml" = ['version = "{version}"']
"pollution_extractor/__init__.py" = ['__version__ = "{version}"']
```

---

Enjoy clean, automated Python packaging! 🐍📦
