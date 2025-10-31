from setuptools import setup, find_packages

setup(
    name="utils",
    version="0.1.0",
    author="Mahmoud Adel",
    description="Custom utility package",
    packages=find_packages(where="src", include=["utils", "utils.*"]),
    package_dir={"": "src"},
    install_requires=[
        "matplotlib",
        "numpy",
        "pandas",
        "seaborn",
        "scikit-learn",
    ],
    python_requires=">=3.8",
)
