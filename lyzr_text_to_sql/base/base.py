import json
import os
import re
import sqlite3
import traceback
from abc import ABC, abstractmethod
from typing import List, Tuple, Union
from urllib.parse import urlparse

import pandas as pd
import plotly
import plotly.express as px
import plotly.graph_objects as go
import requests

from ..exceptions import DependencyError, ImproperlyConfigured, ValidationError
from ..types import TrainingPlan, TrainingPlanItem
from ..utils import validate_config_path


class LyzrBase(ABC):
    def __init__(self, config=None):
        self.config = config
        self.run_sql_is_set = False

    def log(self, message: str):
        print(message)

    def generate_sql(self, question: str, **kwargs) -> str:
        question_sql_list = self.get_similar_question_sql(question, **kwargs)
        ddl_list = self.get_related_ddl(question, **kwargs)
        doc_list = self.get_related_documentation(question, **kwargs)
        prompt = self.get_sql_prompt(
            question=question,
            question_sql_list=question_sql_list,
            ddl_list=ddl_list,
            doc_list=doc_list,
            **kwargs,
        )

        # print("This is the prompt", prompt)
        llm_response = self.submit_prompt(prompt, **kwargs)
        # print("This is the llm_response", llm_response)
        return self.extract_sql(llm_response)

    def extract_sql(self, llm_response: str) -> str:
        # If the llm_response contains a markdown code block, with or without the sql tag, extract the sql from it
        sql = re.search(r"```sql\n(.*)```", llm_response, re.DOTALL)
        if sql:
            self.log(f"Output from LLM: {llm_response} \nExtracted SQL: {sql.group(1)}")
            return sql.group(1)

        sql = re.search(r"```(.*)```", llm_response, re.DOTALL)
        if sql:
            self.log(f"Output from LLM: {llm_response} \nExtracted SQL: {sql.group(1)}")
            return sql.group(1)

        return llm_response

    def is_sql_valid(self, sql: str) -> bool:
        # This is a check to see the SQL is valid and should be run
        # This simple function just checks if the SQL contains a SELECT statement

        if "SELECT" in sql.upper():
            return True
        else:
            return False

    def generate_followup_questions(self, question: str, **kwargs) -> str:
        question_sql_list = self.get_similar_question_sql(question, **kwargs)
        ddl_list = self.get_related_ddl(question, **kwargs)
        doc_list = self.get_related_documentation(question, **kwargs)
        prompt = self.get_followup_questions_prompt(
            question=question,
            question_sql_list=question_sql_list,
            ddl_list=ddl_list,
            doc_list=doc_list,
            **kwargs,
        )
        llm_response = self.submit_prompt(prompt, **kwargs)

        numbers_removed = re.sub(r"^\d+\.\s*", "", llm_response, flags=re.MULTILINE)
        return numbers_removed.split("\n")

    def generate_questions(self, **kwargs) -> List[str]:
        """
        **Example:**
        ```python
        vn.generate_questions()
        ```

        Generate a list of questions that you can ask Lyzr.AI.
        """
        question_sql = self.get_similar_question_sql(question="", **kwargs)

        return [q["question"] for q in question_sql]

    # ----------------- Use Any Embeddings API ----------------- #
    @abstractmethod
    def generate_embedding(self, data: str, **kwargs) -> List[float]:
        pass

    # ----------------- Use Any Database to Store and Retrieve Context ----------------- #
    @abstractmethod
    def get_similar_question_sql(self, question: str, **kwargs) -> list:
        pass

    @abstractmethod
    def get_related_ddl(self, question: str, **kwargs) -> list:
        pass

    @abstractmethod
    def get_related_documentation(self, question: str, **kwargs) -> list:
        pass

    @abstractmethod
    def add_question_sql(self, question: str, sql: str, **kwargs) -> str:
        pass

    @abstractmethod
    def add_ddl(self, ddl: str, **kwargs) -> str:
        pass

    @abstractmethod
    def add_documentation(self, documentation: str, **kwargs) -> str:
        pass

    @abstractmethod
    def get_training_data(self, **kwargs) -> pd.DataFrame:
        pass

    @abstractmethod
    def remove_training_data(id: str, **kwargs) -> bool:
        pass

    # ----------------- Use Any Language Model API ----------------- #
    
    @abstractmethod
    def get_sql_prompt(
        self,
        question: str,
        question_sql_list: list,
        ddl_list: list,
        doc_list: list,
        **kwargs,
    ):
        pass

    @abstractmethod
    def get_followup_questions_prompt(
        self,
        question: str,
        question_sql_list: list,
        ddl_list: list,
        doc_list: list,
        **kwargs,
    ):
        pass

    @abstractmethod
    def submit_prompt(self, prompt, **kwargs) -> str:
        pass

    @abstractmethod
    def generate_question(self, sql: str, **kwargs) -> str:
        pass

    @abstractmethod
    def generate_plotly_code(
        self, question: str = None, sql: str = None, df_metadata: str = None, **kwargs
    ) -> str:
        pass
    
    @abstractmethod
    def get_results_prompt(
        self,
        question: str,
        sql: str,
        df: pd.DataFrame, 
        **kwargs,
    ):
        pass

    # ----------------- Connect to Any Database to run the Generated SQL ----------------- #
   
    def connect_to_postgres(
        self,
        host: str = None,
        dbname: str = None,
        user: str = None,
        password: str = None,
        port: int = None,
    ):
        """
        Connect to postgres using the psycopg2 connector. This is just a helper function to set [`vn.run_sql`][Lyzr.run_sql]
        **Example:**
        ```python
        vn.connect_to_postgres(
            host="myhost",
            dbname="mydatabase",
            user="myuser",
            password="mypassword",
            port=5432
        )
        ```
        Args:
            host (str): The postgres host.
            dbname (str): The postgres database name.
            user (str): The postgres user.
            password (str): The postgres password.
            port (int): The postgres Port.
        """

        try:
            import psycopg2
            import psycopg2.extras
        except ImportError:
            raise DependencyError(
                "You need to install required dependencies to execute this method,"
                " run command: \npip install Lyzr[postgres]"
            )

        if not host:
            host = os.getenv("HOST")

        if not host:
            raise ImproperlyConfigured("Please set your postgres host")

        if not dbname:
            dbname = os.getenv("DATABASE")

        if not dbname:
            raise ImproperlyConfigured("Please set your postgres database")

        if not user:
            user = os.getenv("PG_USER")

        if not user:
            raise ImproperlyConfigured("Please set your postgres user")

        if not password:
            password = os.getenv("PASSWORD")

        if not password:
            raise ImproperlyConfigured("Please set your postgres password")

        if not port:
            port = os.getenv("PORT")

        if not port:
            raise ImproperlyConfigured("Please set your postgres port")

        conn = None

        try:
            conn = psycopg2.connect(
                host=host,
                dbname=dbname,
                user=user,
                password=password,
                port=port,
            )
        except psycopg2.Error as e:
            raise ValidationError(e)

        def run_sql_postgres(sql: str) -> Union[pd.DataFrame, None]:
            if conn:
                try:
                    cs = conn.cursor()
                    cs.execute(sql)
                    results = cs.fetchall()

                    # Create a pandas dataframe from the results
                    df = pd.DataFrame(
                        results, columns=[desc[0] for desc in cs.description]
                    )
                    return df

                except psycopg2.Error as e:
                    conn.rollback()
                    raise ValidationError(e)

        self.run_sql_is_set = True
        self.run_sql = run_sql_postgres

    def run_sql(sql: str, **kwargs) -> pd.DataFrame:
        raise NotImplementedError(
            "You need to connect_to_posgres or other database first."
        )
        
    def generate_results(self, question: str, sql: str, df: pd.DataFrame,**kwargs):
        prompt = self.get_results_prompt(question,sql,df)
        llm_response = self.submit_prompt(prompt, **kwargs)
        return  llm_response


    def ask(
        self,
        question: Union[str, None] = None,
        print_sql: bool = True,
        auto_train: bool = True,
        visualize: bool = True,
        print_results: bool = True
    ) -> Tuple[
        Union[str, None],  # results or None
        Union[str, None],  # sql_query or None
        Union[pd.DataFrame, None],  # df or None
        Union[plotly.graph_objs.Figure, None],  # figure or None
    ]:
        if question is None:
            question = input("Enter a question: ")

        sql_query = None
        df = None
        figure = None
        results = None

        # SQL Generation
        try:
            sql_query = self.generate_sql(question=question)
            if print_sql:
                try:
                    Code = __import__("IPython.display", fromList=["Code"]).Code
                    display = __import__("IPython.display", fromlist=["display"]).display
                    display(Code(sql_query))
                except Exception:
                    print(sql_query)
        except Exception as e:
            print(f"SQL generation error: {e}")

        # SQL Execution
        if sql_query and self.run_sql_is_set:
            try:
                df = self.run_sql(sql_query)
                if auto_train and df is not None and len(df) > 0:
                    self.add_question_sql(question=question, sql=sql_query)
            except Exception as e:
                print(f"SQL execution error: {e}")

        # Results Generation and Printing
        if print_results and df is not None:
            try:
                results = self.generate_results(question, sql_query, df)
                print(results)
            except Exception as e:
                print(f"Results generation error: {e}")

        # Visualization
        if visualize and df is not None and len(df) > 0:
            try:
                plotly_code = self.generate_plotly_code(question=question, sql=sql_query, df_metadata=f"Running df.dtypes gives:\n {df.dtypes}")
                figure = self.get_plotly_figure(plotly_code=plotly_code, df=df)
                display = __import__(
                                    "IPython.display", fromlist=["display"]
                                ).display
                Image = __import__(
                                    "IPython.display", fromlist=["Image"]
                                ).Image
                img_bytes = figure.to_image(format="png", scale=2)
                display(Image(img_bytes))
            except Exception as e:
                print(f"Visualization error: {e}")

        return results, sql_query, df, figure

    
    def train(
        self,
        question: str = None,
        sql: str = None,
        ddl: str = None,
        documentation: str = None,
        plan: TrainingPlan = None,
    ) -> str:
        """
        **Example:**
        ```python
        vn.train()
        ```

        Train Lyzr.AI on a question and its corresponding SQL query.
        If you call it with no arguments, it will check if you connected to a database and it will attempt to train on the metadata of that database.
        If you call it with the sql argument, it's equivalent to [`add_sql()`][Lyzr.add_sql].
        If you call it with the ddl argument, it's equivalent to [`add_ddl()`][Lyzr.add_ddl].
        If you call it with the documentation argument, it's equivalent to [`add_documentation()`][Lyzr.add_documentation].
        Additionally, you can pass a [`TrainingPlan`][Lyzr.TrainingPlan] object. Get a training plan with [`vn.get_training_plan_experimental()`][Lyzr.get_training_plan_experimental].

        Args:
            question (str): The question to train on.
            sql (str): The SQL query to train on.
            ddl (str):  The DDL statement.
            documentation (str): The documentation to train on.
            plan (TrainingPlan): The training plan to train on.
        """

        if question and not sql:
            raise ValidationError(f"Please also provide a SQL query")

        if documentation:
            print("Adding documentation....")
            return self.add_documentation(documentation)

        if sql:
            if question is None:
                question = self.generate_question(sql)
                print("Question generated with sql:", question, "\nAdding SQL...")
            return self.add_question_sql(question=question, sql=sql)

        if ddl:
            print("Adding ddl:", ddl)
            return self.add_ddl(ddl)

        if plan:
            for item in plan._plan:
                if item.item_type == TrainingPlanItem.ITEM_TYPE_DDL:
                    self.add_ddl(item.item_value)
                elif item.item_type == TrainingPlanItem.ITEM_TYPE_IS:
                    self.add_documentation(item.item_value)
                elif item.item_type == TrainingPlanItem.ITEM_TYPE_SQL:
                    self.add_question_sql(question=item.item_name, sql=item.item_value)

    def _get_databases(self) -> List[str]:
        try:
            print("Trying INFORMATION_SCHEMA.DATABASES")
            df_databases = self.run_sql("SELECT * FROM INFORMATION_SCHEMA.DATABASES")
        except Exception as e:
            print(e)
            try:
                print("Trying SHOW DATABASES")
                df_databases = self.run_sql("SHOW DATABASES")
            except Exception as e:
                print(e)
                return []

        return df_databases["DATABASE_NAME"].unique().tolist()

    def _get_information_schema_tables(self, database: str) -> pd.DataFrame:
        df_tables = self.run_sql(f"SELECT * FROM {database}.INFORMATION_SCHEMA.TABLES")

        return df_tables

    def get_training_plan_generic(self, df) -> TrainingPlan:
        # For each of the following, we look at the df columns to see if there's a match:
        database_column = df.columns[
            df.columns.str.lower().str.contains("database")
            | df.columns.str.lower().str.contains("table_catalog")
        ].to_list()[0]
        schema_column = df.columns[
            df.columns.str.lower().str.contains("table_schema")
        ].to_list()[0]
        table_column = df.columns[
            df.columns.str.lower().str.contains("table_name")
        ].to_list()[0]
        column_column = df.columns[
            df.columns.str.lower().str.contains("column_name")
        ].to_list()[0]
        data_type_column = df.columns[
            df.columns.str.lower().str.contains("data_type")
        ].to_list()[0]

        plan = TrainingPlan([])

        for database in df[database_column].unique().tolist():
            for schema in (
                df.query(f'{database_column} == "{database}"')[schema_column]
                .unique()
                .tolist()
            ):
                for table in (
                    df.query(
                        f'{database_column} == "{database}" and {schema_column} == "{schema}"'
                    )[table_column]
                    .unique()
                    .tolist()
                ):
                    df_columns_filtered_to_table = df.query(
                        f'{database_column} == "{database}" and {schema_column} == "{schema}" and {table_column} == "{table}"'
                    )
                    doc = f"The following columns are in the {table} table in the {database} database:\n\n"
                    doc += df_columns_filtered_to_table[
                        [
                            database_column,
                            schema_column,
                            table_column,
                            column_column,
                            data_type_column,
                        ]
                    ].to_markdown()

                    plan._plan.append(
                        TrainingPlanItem(
                            item_type=TrainingPlanItem.ITEM_TYPE_IS,
                            item_group=f"{database}.{schema}",
                            item_name=table,
                            item_value=doc,
                        )
                    )

        return plan

    def get_plotly_figure(
        self, plotly_code: str, df: pd.DataFrame, dark_mode: bool = True
    ) -> plotly.graph_objs.Figure:
        """
        **Example:**
        ```python
        fig = vn.get_plotly_figure(
            plotly_code="fig = px.bar(df, x='name', y='salary')",
            df=df
        )
        fig.show()
        ```
        Get a Plotly figure from a dataframe and Plotly code.

        Args:
            df (pd.DataFrame): The dataframe to use.
            plotly_code (str): The Plotly code to use.

        Returns:
            plotly.graph_objs.Figure: The Plotly figure.
        """
        ldict = {"df": df, "px": px, "go": go}
        try:
            exec(plotly_code, globals(), ldict)

            fig = ldict.get("fig", None)
        except Exception as e:
            # Inspect data types
            numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
            categorical_cols = df.select_dtypes(
                include=["object", "category"]
            ).columns.tolist()

            # Decision-making for plot type
            if len(numeric_cols) >= 2:
                # Use the first two numeric columns for a scatter plot
                fig = px.scatter(df, x=numeric_cols[0], y=numeric_cols[1])
            elif len(numeric_cols) == 1 and len(categorical_cols) >= 1:
                # Use a bar plot if there's one numeric and one categorical column
                fig = px.bar(df, x=categorical_cols[0], y=numeric_cols[0])
            elif len(categorical_cols) >= 1 and df[categorical_cols[0]].nunique() < 10:
                # Use a pie chart for categorical data with fewer unique values
                fig = px.pie(df, names=categorical_cols[0])
            else:
                # Default to a simple line plot if above conditions are not met
                fig = px.line(df)

        if fig is None:
            return None

        if dark_mode:
            fig.update_layout(template="plotly_dark")

        return fig


class SplitStorage(LyzrBase):
    def __init__(self, config=None):
        LyzrBase.__init__(self, config=config)

    def get_similar_question_sql(self, embedding: str, **kwargs) -> list:
        question_sql_ids = self.get_similar_question_sql_ids(embedding, **kwargs)
        question_sql_list = self.get_question_sql(question_sql_ids, **kwargs)
        return question_sql_list

    def get_related_ddl(self, embedding: str, **kwargs) -> list:
        ddl_ids = self.get_related_ddl_ids(embedding, **kwargs)
        ddl_list = self.get_ddl(ddl_ids, **kwargs)
        return ddl_list

    def get_related_documentation(self, embedding: str, **kwargs) -> list:
        doc_ids = self.get_related_documentation_ids(embedding, **kwargs)
        doc_list = self.get_documentation(doc_ids, **kwargs)
        return doc_list

    # ----------------- Use Any Vector Database to Store and Lookup Embedding Similarity ----------------- #
    @abstractmethod
    def store_question_sql_embedding(self, embedding: str, **kwargs) -> str:
        pass

    @abstractmethod
    def store_ddl_embedding(self, embedding: str, **kwargs) -> str:
        pass

    @abstractmethod
    def store_documentation_embedding(self, embedding: str, **kwargs) -> str:
        pass

    @abstractmethod
    def get_similar_question_sql_ids(self, embedding: str, **kwargs) -> list:
        pass

    @abstractmethod
    def get_related_ddl_ids(self, embedding: str, **kwargs) -> list:
        pass

    @abstractmethod
    def get_related_documentation_ids(self, embedding: str, **kwargs) -> list:
        pass

    # ----------------- Use Database to Retrieve the Documents from ID Lists ----------------- #
    @abstractmethod
    def get_question_sql(self, question_sql_ids: list, **kwargs) -> list:
        pass

    @abstractmethod
    def get_documentation(self, doc_ids: list, **kwargs) -> list:
        pass

    @abstractmethod
    def get_ddl(self, ddl_ids: list, **kwargs) -> list:
        pass
