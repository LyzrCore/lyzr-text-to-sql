# Lyzr Text to Sql

![version](https://img.shields.io/badge/version-0.1.0-blue.svg) [![Discord](https://img.shields.io/badge/Discord-join%20now-blue.svg?style=flat&logo=Discord)](https://discord.gg/dXmgggHYUz)

`lyzr_text_to_sql` is a Advanecd Python package that bridges natural language processing with SQL database interactions. It utilizes LLM's models to convert natural language queries into executable SQL commands. This tool aims to simplify database querying, making it more accessible and intuitive for users without extensive SQL knowledge.

## Features

- Seamless integration with PostgreSQL databases.
- Natural language to SQL translation.
- Direct execution of SQL commands with results processing.
- Support for model training with custom database schemas.
- Data visualization capabilities.

## Installation

Install `lyzr_text_to_sql` directly from PyPI:

```sh
pip install lyzr_text_to_sql
```

## Getting Started

Here's a quick example to get you started with `lyzr_text_to_sql`:

```python
from lyzr_text_to_sql import Data_Analyzer

# Initialize the Data Analyzer with your API key and model configuration
da = Data_Analyzer(
    config={
        "api_key": "your_api_key_here",
        "model": "gpt-4-1106-preview",
    }
)

# Connect to a PostgreSQL database
da.connect_to_postgres(dbname="postgres", user="postgres", password="", host="", port="")

# Query the database schema, create a training plan, and train the model
schema = da.run_sql("SELECT * FROM INFORMATION_SCHEMA.COLUMNS")
training_plan = da.get_training_plan_generic(schema)
da.train(plan=training_plan)

# Use natural language to query your database
results, sql_query, df, figure = da.ask(question="List all employees hired in 2020")
```

## Usage

### Database Connection

Use `connect_to_postgres` to establish a connection with your PostgreSQL database before executing queries.

### Training the Model

To improve translation accuracy, generate a training plan based on your database schema using `get_training_plan_generic` and train the model with `train`.

### Querying with Natural Language

Once trained, you can input questions in natural language using the `ask` method to get SQL translations, executed results, and visualizations.

## Contributing

We welcome contributions! If you'd like to contribute, please fork the repository and use a pull request for your contributions. For major changes, please open an issue first to discuss what you would like to change.

## Reporting Issues

Found a bug or have a suggestion? Please use the GitHub issues to report them.

## Contact
For queries, reach us at contact@lyzr.ai

[![Discord](https://img.shields.io/badge/Discord-join%20now-blue.svg?style=flat&logo=Discord)](https://discord.gg/dXmgggHYUz)

## License

`lyzr_text_to_sql` is licensed under the MIT License. See the LICENSE file for more details.
 