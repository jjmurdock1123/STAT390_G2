# flake8: noqa
import pandas as pd, numpy as np, os, re, glob
pd.set_option('display.float_format', lambda x: '%.2f' % x)

import uuid, json, base64
from tools.display_dataframe import display_dataframe
from functools import lru_cache
from langchain.tools import BaseTool
from typing import Any, Dict, Optional, Sequence, Type, Union
# from langchain_community.utilities import SQLDatabase  # Commented out - SQL dependency
from pydantic import BaseModel, Field, ConfigDict
from langchain_core.tools import BaseTool

from langchain_core.tools import InjectedToolArg
from typing_extensions import Annotated

from func_timeout import func_timeout

# from func.gcp import upload_file_to_gcp  # Commented out - GCP dependency


class BaseDataFrameTool(BaseModel):
    """
    Adapted from BaseSQLDatabaseTool in sql.py
    Instead of db_dict containing SQLDatabase objects, df_dict contains DataFrame metadata
    """
    df_dict: Dict[str, Dict] = Field(exclude=True)  # Changed from SQLDatabase to Dict
    model_config = ConfigDict(
        arbitrary_types_allowed=True,
    )


class DataFrameListTableInput(BaseModel):
    """Mirrors SQLListTableInput from sql.py"""
    query: str = Field(default="", description="An empty string")


class DataFrameListTableTool(BaseDataFrameTool, BaseTool):
    """
    Adapted from SQLListTableTool in sql.py
    Lists all available DataFrames (CSV/Parquet files) in the data/input directory
    """
    name: str = "df_list_table"
    description: str = """
    Function: List all available DataFrames in the local data directory.
    Input: An empty string.
    Output: The names and brief descriptions of all DataFrames loaded from files.
    """
    response_format: str = "content_and_artifact"
    args_schema: Type[BaseModel] = DataFrameListTableInput

    data_source: str = "LocalDataFrames"  # Changed from db_name
    display_mode: str = "markdown"
    data_path: str = "data/input/"  # New field for local data path

    def _run(self, query: str):
        """
        Adapted from SQLListTableTool._run() in sql.py
        Instead of querying database metadata, scans local files
        """
        response = {}
        try:
            # Scan for CSV and Parquet files in data directory
            csv_files = glob.glob(os.path.join(self.data_path, "*.csv"))
            parquet_files = glob.glob(os.path.join(self.data_path, "*.parquet"))
            
            table_dict = []
            
            # Process CSV files
            for file_path in csv_files:
                table_name = os.path.splitext(os.path.basename(file_path))[0]
                file_size = os.path.getsize(file_path)
                table_dict.append({
                    "TableName": table_name,
                    "TableDescription": f"CSV file ({file_size} bytes) - {file_path}"
                })
            
            # Process Parquet files  
            for file_path in parquet_files:
                table_name = os.path.splitext(os.path.basename(file_path))[0]
                file_size = os.path.getsize(file_path)
                table_dict.append({
                    "TableName": table_name,
                    "TableDescription": f"Parquet file ({file_size} bytes) - {file_path}"
                })

            if self.display_mode == "markdown":
                output = pd.DataFrame(table_dict).to_markdown(index=False)
            else:
                output = json.dumps(table_dict, indent=4)

            response['response'] = output

        except Exception as e:
            response['response'] = "{}: {}".format(type(e).__name__, str(e))
        
        return response, response


@lru_cache(maxsize=32)
def cached_get_dataframe_info(data_path, table_name):
    """
    Adapted from cached_get_table_info() in sql.py
    Gets DataFrame column info and dtypes instead of SQL table schema
    """
    file_path = None
    
    # Try to find CSV or Parquet file
    csv_path = os.path.join(data_path, f"{table_name}.csv")
    parquet_path = os.path.join(data_path, f"{table_name}.parquet")
    
    if os.path.exists(csv_path):
        file_path = csv_path
        # Read just the first few rows to get column info
        df_sample = pd.read_csv(file_path, nrows=5)
    elif os.path.exists(parquet_path):
        file_path = parquet_path
        # Read just the first few rows to get column info
        df_sample = pd.read_parquet(file_path).head(5)
    else:
        raise FileNotFoundError(f"No CSV or Parquet file found for table: {table_name}")
    
    # Build schema info similar to SQL format
    schema_info = f"CREATE TABLE {table_name} (\n"
    for col in df_sample.columns:
        dtype = str(df_sample[col].dtype)
        schema_info += f"    {col} {dtype},\n"
    schema_info = schema_info.rstrip(",\n") + "\n)"
    
    return schema_info

def clear_dataframe_info_cache():
    """Adapted from clear_table_info_cache() in sql.py"""
    cached_get_dataframe_info.cache_clear()

def read_dataframe_chunked(file_path: str, chunksize: int = 1000, timeout: int = 120):
    """
    Adapted from read_sql() in sql.py
    Reads DataFrame from file with chunking support to avoid memory issues
    """
    def _read_file():
        if file_path.endswith('.csv'):
            if chunksize:
                # Return chunked reader for large files
                return pd.read_csv(file_path, chunksize=chunksize)
            else:
                return pd.read_csv(file_path)
        elif file_path.endswith('.parquet'):
            # Parquet doesn't support chunksize directly, so read all and chunk if needed
            df = pd.read_parquet(file_path)
            if chunksize and len(df) > chunksize:
                # Convert to chunked iterator
                return [df[i:i+chunksize] for i in range(0, len(df), chunksize)]
            return df
        else:
            raise ValueError(f"Unsupported file format: {file_path}")
    
    result = func_timeout(timeout, _read_file)
    
    # Handle chunked vs non-chunked results similar to original
    if isinstance(result, pd.DataFrame):
        df = result
    elif hasattr(result, '__iter__') and not isinstance(result, pd.DataFrame):
        # Concatenate chunks
        df = pd.concat([chunk for chunk in result], ignore_index=True)
    else:
        df = result
    
    # Handle list columns similar to original
    for col in df.columns:
        if df[col].dtype == object:
            # Check if any values are lists and convert to numpy arrays
            if any(isinstance(x, list) for x in df[col].dropna()):
                df[col] = df[col].apply(lambda x: np.array(x) if isinstance(x, list) else x)
    
    return df


class DataFrameGetSchemaInput(BaseModel):
    """Mirrors SQLGetSchemaInput from sql.py"""
    query: str = Field(default="", description="A list of table names separated by commas. For example, `table1, table2, table3`.")


class DataFrameGetSchemaTool(BaseDataFrameTool, BaseTool):
    """
    Adapted from SQLGetSchemaTool in sql.py
    Retrieves DataFrame schema information and sample rows from local files
    """
    name: str = "df_get_schema"
    description: str = """
    Function: Retrieves detailed schema information and sample rows for specified DataFrames.
    Input: A comma-separated list of table names. If left empty, retrieves information for all available DataFrames.
    Output:
    For each specified DataFrame:
    1. Detailed column information (names, data types, descriptions)
    2. Sample rows to illustrate the data structure
    Dependencies:
    1. Use `df_list_table` to get a list of all available DataFrames.
    """
    response_format: str = "content_and_artifact"
    args_schema: Type[BaseModel] = DataFrameGetSchemaInput

    sample_rows: int = 3
    data_source: str = "LocalDataFrames"  # Changed from db_name
    data_path: str = "data/input/"

    def _run(self, query: str):
        """
        Adapted from SQLGetSchemaTool._run() in sql.py
        Gets schema info from local files instead of SQL database
        """
        response = {}
        try:
            if query == "":
                # Get all available table names from files
                csv_files = glob.glob(os.path.join(self.data_path, "*.csv"))
                parquet_files = glob.glob(os.path.join(self.data_path, "*.parquet"))
                table_names = []
                table_names.extend([os.path.splitext(os.path.basename(f))[0] for f in csv_files])
                table_names.extend([os.path.splitext(os.path.basename(f))[0] for f in parquet_files])
            else:
                table_names = [i.strip() for i in query.split(",")]

            # Get schema info for each table
            table_info_list = []
            for table_name in table_names:
                # Get cached schema info
                schema_info = cached_get_dataframe_info(self.data_path, table_name)
                table_info_list.append(schema_info)
                
                # Get sample rows
                file_path = None
                csv_path = os.path.join(self.data_path, f"{table_name}.csv")
                parquet_path = os.path.join(self.data_path, f"{table_name}.parquet")
                
                if os.path.exists(csv_path):
                    df = pd.read_csv(csv_path, nrows=1000).sample(n=min(self.sample_rows, 1000))
                elif os.path.exists(parquet_path):
                    df_full = pd.read_parquet(parquet_path)
                    df = df_full.sample(n=min(self.sample_rows, len(df_full)))
                else:
                    continue
                
                df_string = display_dataframe(df, mode="string", display_rows=100, decimal_precision=4)
                table_info_list.append(f"\n/*\n{self.sample_rows} rows from {table_name} table:\n{df_string}\n*/\n")

            response["response"] = "\n".join(table_info_list)
        except Exception as e:
            response["response"] = "{}: {}".format(type(e).__name__, str(e))
        return response, response


# from prompts.prompts import sql_query_tool_description_2  # Keep original import
from typing import Literal

class DataFrameQueryInput(BaseModel):
    """Mirrors SQLQueryInput from sql.py"""
    query: str = Field(..., description="A pandas query or operation description (not SQL - use pandas syntax/operations).")


class DataFrameQueryTool(BaseDataFrameTool, BaseTool):
    """
    Executes Python (pandas) operations on local DataFrames.
    Input: A Python code block (as a string) that assigns the final result to a variable called `result`.
    Output: The content of `result`, rendered as a table if possible.
    Note: All DataFrames in the data_path are available as variables named after the CSV/Parquet file (without extension).
    """

    name: str = "df_query"
    description: str = (
        "Function: Execute Python (pandas) operations on local DataFrames loaded from CSV/Parquet files. "
        "Input: Python code (as string). Output: Result of the operation. "
        "You must assign the final output to a variable called `result`. "
        "If you don't, the tool will attempt to automatically assign the last expression to `result`."
    )
    response_format: str = "content_and_artifact"
    args_schema: Type[BaseModel] = DataFrameQueryInput

    data_source: str = "LocalDataFrames"
    session_id: str = "test"
    data_path: str = "data/input/"
    workspace: str = "data/output"
    display_mode: str = "markdown"
    display_rows_preview: int = 10
    display_rows_complete: int = 200
    decimal_precision: int = 4
    filename: str = None  # For optional file naming

    @staticmethod
    def auto_assign_result(query: str) -> str:
        import ast
        lines = query.strip().splitlines()
        if not lines:
            return query
        # Find last meaningful line (not blank or comment)
        for i in range(len(lines)-1, -1, -1):
            last_line = lines[i].strip()
            if last_line and not last_line.startswith("#"):
                break
        else:
            return query
        try:
            expr = ast.parse(last_line, mode='eval')
            # It's a pure expression, rewrite as assignment
            lines[i] = f"result = {last_line}"
            return "\n".join(lines)
        except SyntaxError:
            return query  # Not a pure expression

    def _run(self, query: str, display_rows: int = 10, display_mode: Literal["preview", "complete"] = "preview"):
        response = {}
        os.makedirs(self.workspace, exist_ok=True)

        # 1. Load all DataFrames in data_path as variables
        local_vars = {}
        for file_path in glob.glob(os.path.join(self.data_path, "*.csv")) + glob.glob(os.path.join(self.data_path, "*.parquet")):
            varname = os.path.splitext(os.path.basename(file_path))[0]
            try:
                if file_path.endswith('.csv'):
                    local_vars[varname] = pd.read_csv(file_path)
                elif file_path.endswith('.parquet'):
                    local_vars[varname] = pd.read_parquet(file_path)
            except Exception as read_err:
                response['response'] = f"Failed to load {file_path}: {read_err}"
                return response, response

        # 2. Auto-assign last expression to result if needed
        query = self.auto_assign_result(query)

        # 3. Execute the user-supplied Python code in this local_vars context
        try:
            exec(query, {}, local_vars)
        except Exception as exec_err:
            response['response'] = f"Execution error: {exec_err}"
            return response, response

        # 4. Grab the result
        result = local_vars.get("result", None)
        if result is None:
            response['response'] = "No variable named `result` was found in your code."
            return response, response

        # 5. Display as table if possible, otherwise as string
        if isinstance(result, pd.DataFrame):
            response['response'] = result.head(display_rows).to_markdown(index=False)
            # Save result as a file if it's a DataFrame
            file_id = str(uuid.uuid4())
            file_name = self.filename if self.filename else f"{file_id}.parquet"
            file_path = os.path.join(self.workspace, file_name)
            result.to_parquet(file_path, index=False)
            response["files"] = [{
                "name": file_name,
                "id": file_id,
                "download_link": file_path,
                "file_path": file_path,
                "mime_type": "application/parquet",
            }]
        elif isinstance(result, pd.Series):
            response['response'] = result.to_frame().head(display_rows).to_markdown(index=False)
        else:
            response['response'] = str(result)

        return response, response
        
# Helper function to initialize DataFrame tools (mirrors sql.py initialization pattern)
def initialize_dataframe_tools(data_path: str = "data/input/", workspace: str = "data/output/"):
    """
    Initialize DataFrame tools similar to how sql.py initializes database connections
    """
    # Create dummy df_dict structure to match sql.py's db_dict pattern
    df_dict = {"LocalDataFrames": {"data_path": data_path}}
    
    tools = {
        "list_table": DataFrameListTableTool(df_dict=df_dict, data_path=data_path),
        "get_schema": DataFrameGetSchemaTool(df_dict=df_dict, data_path=data_path),
        "query": DataFrameQueryTool(df_dict=df_dict, data_path=data_path, workspace=workspace)
    }
    
    return tools