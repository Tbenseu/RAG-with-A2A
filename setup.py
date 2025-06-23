from setuptools import setup, find_packages

setup(
    name="rag-with-a-2-a",  # Use hyphens, not underscores
    version="0.1.0",
    package_dir={"": "src"},
    packages=find_packages(where="src"),  # Finds `my_package` in `src/`
    install_requires=["requests"],
    python_requires=">=3.8",  # Relax Python version (or use your actual version)
)