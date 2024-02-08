from setuptools import setup, find_packages

setup(
    name="lyzr-text-to-sql",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "openai==1.3.4",
        "requests==2.31.0",
        "asyncio==3.4.3",
        "chromadb",
        "psycopg2-binary", 
        "db-dtypes",
        "plotly",
        "tabulate",
    ],
    author="lyzr",
    description="low-code multi-agent automation framework",
    long_description=open("README.md").read(),
    long_description_content_type="check Readme file",
    url="https://github.com/lyzrcore/lyzr-text-to-sql",
    license="MIT",
        classifiers=[
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8.1, <3.12",
)