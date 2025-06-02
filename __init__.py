##### Data Extraction Tools
# Commented out SQL tools - replaced with local DataFrame tools
# from tools.sql import SQLListTableTool, SQLGetSchemaTool, SQLQueryTool
# from langchain_community.utilities import SQLDatabase

# New local DataFrame tools
from tools.local_df import DataFrameListTableTool, DataFrameGetSchemaTool, DataFrameQueryTool
from tools.local_df import initialize_dataframe_tools

from tools.gensci_literature import GenSciLiteratureRetrievalTool
import os

from dotenv import load_dotenv
load_dotenv()

# Commented out BigQuery setup - no longer needed for local DataFrames
# bigquery_uri = os.getenv("GOOGLE_BIGQUERY_URI")
# db_name = bigquery_uri.split("/")[-1]
# db_dict = {
# 	db_name: SQLDatabase.from_uri(
# 		database_uri=bigquery_uri, 
# 		sample_rows_in_table_info=0, 
# 	)
# }

# Initialize local DataFrame tools
data_path = os.getenv("LOCAL_DATA_PATH", "data/input/")  # Allow env override
workspace_path = os.getenv("LOCAL_WORKSPACE_PATH", "data/output/")  # Allow env override

# Create DataFrame tools using the helper function
df_tools = initialize_dataframe_tools(
    data_path=data_path,
    workspace=workspace_path
)

# Extract individual tools with same variable names for compatibility
df_list_table_tool = df_tools["list_table"]  # DataFrameListTableTool instance
df_get_schema_tool = df_tools["get_schema"]  # DataFrameGetSchemaTool instance  
df_query_tool = df_tools["query"]          # DataFrameQueryTool instance

# Note: GenSciLiteratureRetrievalTool may need updating if it depends on SQL db_dict
# For now, creating a dummy df_dict structure to maintain compatibility
df_dict = {"LocalDataFrames": {"data_path": data_path}}  # Dummy structure for compatibility

# Check if GenSciLiteratureRetrievalTool can work with df_dict or needs modification
try:
    gen_sci_literature_tool = GenSciLiteratureRetrievalTool(db_dict=df_dict)
except Exception as e:
    print(f"Warning: GenSciLiteratureRetrievalTool may need updating for local DataFrame usage: {e}")
    # You may need to create a local version or modify this tool
    gen_sci_literature_tool = None

# from tools.neo4j import neo4j_get_schema_tool, neo4j_query_tool
# from tools.name import search_name_tool

##### Data Analysis Tools
from tools.sandbox import python_shell_tool #, r_shell_tool

##### Literature Review
# from tools.literature import search_policy_vanilla_tool, search_policy_advanced_tool

##### External Data
from tools.arxiv import arxiv_query_tool
# from tools.google_search import google_search_tool
from tools.openalex import openalex_query_tool

# Tools list - same variable names, but now using DataFrame tools instead of SQL tools
tools = [
    df_list_table_tool, df_get_schema_tool, df_query_tool,  # Now DataFrame tools
	# neo4j_get_schema_tool, neo4j_query_tool, 
    #search_name_tool, 
	python_shell_tool, #r_shell_tool, 
    #search_policy_vanilla_tool, search_policy_advanced_tool, 
    gen_sci_literature_tool if gen_sci_literature_tool else None  # Include if compatible
	# arxiv_query_tool, google_search_tool, openalex_query_tool,
]

# Filter out None values in case gen_sci_literature_tool failed to initialize
tools = [tool for tool in tools if tool is not None]

enabled_tools = [
    df_list_table_tool, df_get_schema_tool, df_query_tool,  # Now DataFrame tools
    python_shell_tool, #search_name_tool, #r_shell_tool, #search_name_tool, 
    # neo4j_get_schema_tool, neo4j_query_tool, 
    #search_policy_advanced_tool, 
    gen_sci_literature_tool if gen_sci_literature_tool else None  # Include if compatible
]

# Filter out None values
enabled_tools = [tool for tool in enabled_tools if tool is not None]

# Configuration summary for debugging
print("=== Tool Configuration Summary ===")
print(f"Data path: {data_path}")
print(f"Workspace path: {workspace_path}")
print(f"List tool type: {type(df_list_table_tool).__name__}")
print(f"Schema tool type: {type(df_get_schema_tool).__name__}")
print(f"Query tool type: {type(df_query_tool).__name__}")
print(f"Total tools loaded: {len(tools)}")
print(f"Total enabled tools: {len(enabled_tools)}")
if gen_sci_literature_tool is None:
    print("Warning: gen_sci_literature_tool failed to initialize - may need local DataFrame adaptation")
print("=====================================")