# Prebuilt image for the surrogate "build" job.
#
# Bakes in the heavy Python dependencies (notably TensorFlow) so CI does not
# reinstall them on every run. The project source is deliberately NOT copied in
# here -- it changes constantly. At runtime the build job installs the checked-out
# source with `pip install --no-deps -e .`, which is instant because every
# dependency below is already present.
#
# Rebuild this image only when the dependency set changes (see
# .github/workflows/build-images.yml).

FROM python:3.11-slim

# Faster, quieter, no .pyc clutter.
ENV PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# git lets actions/checkout do a real clone instead of the tarball fallback.
RUN apt-get update && \
    apt-get install -y --no-install-recommends git && \
    rm -rf /var/lib/apt/lists/*

# Runtime dependencies from pyproject.toml, plus TensorFlow (primary training
# backend) and pytest (dev). Kept in sync with pyproject.toml [project] and
# [project.optional-dependencies].
RUN python -m pip install --upgrade pip && \
    pip install \
        "numpy>=1.24" \
        "pandas>=2.0" \
        "scikit-learn>=1.3" \
        "openpyxl>=3.1" \
        "pydantic>=2.5" \
        "PyYAML>=6.0" \
        "tensorflow>=2.13" \
        "pytest>=7.4"

WORKDIR /workspace
