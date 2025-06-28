from setuptools import setup, find_packages

setup(
    name="bounce_loan_prediction",
    version="0.1.0",
    description="Bounce Loan Repayment Prediction Package",
    author="Data Science Team",
    author_email="data@bounce.com",
    packages=find_packages(),
    install_requires=[
        "python-dotenv",
        "pandas",
        "numpy",
        "scikit-learn",
        "sqlalchemy",
        "snowflake-sqlalchemy",
        "jupyterlab",
        "matplotlib",
        "seaborn",
        "pytest",
    ],
    python_requires=">=3.8",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
) 