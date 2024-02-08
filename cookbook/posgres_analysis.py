from lyzr_text_to_sql import Data_Analyzer

# Initialize the Data Analyzer with your API key and model configuration
da = Data_Analyzer(
    config={
        "api_key": "sk-",
        "model": "gpt-4-1106-preview",
    }
)

# Connect to a PostgreSQL database
da.connect_to_postgres(  
    dbname="",
    user="",
    password="",
    host="",
    port="",
)

# Query the database schema, create a training plan, and train the model
schema = da.run_sql("SELECT * FROM INFORMATION_SCHEMA.COLUMNS")
training_plan = da.get_training_plan_generic(schema)
da.train(plan=training_plan)

# Use natural language to query your database
results, sql_query, df, figure = da.ask(question="List all employees hired in 2020")
print(results)
