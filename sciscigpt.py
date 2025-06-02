from langchain_core.messages import AnyMessage, HumanMessage, ToolMessage, AIMessage
from typing import Annotated, Dict, Any, Literal, TypedDict
from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import ToolNode
from langchain.hub import pull
from functools import partial
import re, json

from agents.nodes import call_research_manager, call_specialist, call_tool, AgentState



def select_next(state: AgentState) -> Literal["research_manager", "database_specialist", "analytics_specialist", "literature_specialist", "evaluation_specialist", "specialistset", "toolset", END]:
	if "task_list" not in state: # If SciSciGPT does not require any subtasks, directly end the session
		return END
	else:
		newest_task = state["task_list"][-1]
		newest_message = state["messages"][-1]
		
		if newest_task["status"] == "created":
			if isinstance(newest_message, AIMessage): # Connecting to a specialist
				return "specialistset"
			else: # connected to a specialist
				return newest_task["specialist"]
			
		elif newest_task["status"] == "in_progress":
			if isinstance(newest_message, ToolMessage): # Tool finished: invoke tool evaluation
				return "evaluation_specialist"
			elif newest_message.tool_calls: # Tool called
				return "toolset"
			else: # Tool evaluated
				return newest_task["specialist"]
			
		elif newest_task["status"] == "completed": # Invoke task evaluation
			return "evaluation_specialist"

		elif newest_task["status"] == "evaluated":
			if state["end"]:
				return END
			else:
				return "research_manager"
		else:
			return "research_manager"


from agents.specialists import database_specialist, analytics_specialist, literature_specialist, evaluation_specialist

# Updated imports: Now using local DataFrame tools instead of SQL tools
# Note: Variable names remain the same for backward compatibility, but these are now DataFrame tools
from tools import df_list_table_tool, df_get_schema_tool, df_query_tool, python_shell_tool #search_name_tool, #search_policy_advanced_tool
#r_shell_tool

# For clarity, we can create aliases to make it clear these are now DataFrame tools
# while maintaining compatibility with existing code
# df_list_table_tool = sql_list_table_tool      # Now DataFrameListTableTool
# df_get_schema_tool = sql_get_schema_tool      # Now DataFrameGetSchemaTool  
# df_query_tool = sql_query_tool                # Now DataFrameQueryTool

# The toolset of ResearchManager
specialists = [database_specialist, analytics_specialist, literature_specialist]

# The toolset of each specialist
# Updated: database_specialist now uses DataFrame tools instead of SQL tools
database_specialist_tools = [
    df_list_table_tool,    # DataFrameListTableTool - lists available CSV/Parquet files
    df_get_schema_tool,    # DataFrameGetSchemaTool - shows DataFrame schemas and samples
    df_query_tool,         # DataFrameQueryTool - executes pandas operations on local files
    #search_name_tool,      # Unchanged - name search functionality
    evaluation_specialist  # Unchanged - evaluation specialist
]

# Analytics specialist tools remain unchanged
analytics_specialist_tools = [python_shell_tool, evaluation_specialist] #r_shell_tool

# Literature specialist tools remain unchanged  
literature_specialist_tools = [#search_policy_advanced_tool, 
evaluation_specialist]

from agents.evaluation import evaluate
toolsets_by_specialist = {
	"database_specialist": database_specialist_tools,    # Now uses DataFrame tools
	"analytics_specialist": analytics_specialist_tools,   # Unchanged
	"literature_specialist": literature_specialist_tools  # Unchanged
}

# All tools collection - now includes DataFrame tools instead of SQL tools

all_tools = [*database_specialist_tools, *analytics_specialist_tools, *literature_specialist_tools]
tools_by_name = {tool.name: tool for tool in all_tools}

# Debug: Print tool names to verify DataFrame tools are loaded
print("=== SciSciGPT Tool Configuration ===")
print("Database Specialist Tools:")
for tool in database_specialist_tools:
    if hasattr(tool, 'name'):
        print(f"  - {tool.name} ({type(tool).__name__})")
print(f"Analytics Specialist Tools: {len(analytics_specialist_tools)} tools")
print(f"Literature Specialist Tools: {len(literature_specialist_tools)} tools")
print(f"Total tools loaded: {len(all_tools)}")

# Verify DataFrame tools are properly loaded
df_tool_names = [tool.name for tool in database_specialist_tools[:3] if hasattr(tool, 'name')]
expected_df_names = ['df_list_table', 'df_get_schema', 'df_query']
if any(expected in str(df_tool_names) for expected in expected_df_names):
    print("✅ DataFrame tools successfully loaded")
else:
    print("⚠️  Warning: Expected DataFrame tools not detected, check tools.py configuration")
print("=====================================")


from reasoning.dynamic_prompt_pruning import dynamic_prompt_pruning
# pruning_func = dynamic_prompt_pruning
pruning_func = lambda x: x


def define_sciscigpt_graph(llm):
	"""
	Define the SciSciGPT graph with updated DataFrame tools.
	
	The database_specialist now uses local DataFrame tools instead of SQL tools:
	- df_list_table_tool: Lists available CSV/Parquet files
	- df_get_schema_tool: Shows DataFrame schemas and sample data
	- df_query_tool: Executes pandas operations on local data
	
	All other specialists remain unchanged.
	"""
	research_manager_node = partial(call_research_manager, llm, specialists, pruning_func)
	
	# Database specialist now works with local DataFrames instead of SQL databases
	database_specialist_node = partial(call_specialist, llm, database_specialist_tools, pruning_func)
	
	# Analytics and literature specialists remain unchanged
	analytics_specialist_node = partial(call_specialist, llm, analytics_specialist_tools, pruning_func)
	literature_specialist_node = partial(call_specialist, llm, literature_specialist_tools, pruning_func)
	
	# Evaluation specialist unchanged
	evaluation_specialist_node = partial(evaluate, llm, toolsets_by_specialist)

	specialistset_node = ToolNode(specialists)
	toolset_node = partial(call_tool, tools_by_name)

	sciscigpt_graph = StateGraph(AgentState)
	sciscigpt_graph.add_node("research_manager", research_manager_node)
	sciscigpt_graph.add_node("database_specialist", database_specialist_node)  # Now uses DataFrame tools
	sciscigpt_graph.add_node("analytics_specialist", analytics_specialist_node)
	sciscigpt_graph.add_node("literature_specialist", literature_specialist_node)
	sciscigpt_graph.add_node("evaluation_specialist", evaluation_specialist_node)

	sciscigpt_graph.add_node("specialistset", specialistset_node)
	sciscigpt_graph.add_node("toolset", toolset_node)

	# Graph edges remain unchanged - same workflow logic
	sciscigpt_graph.add_edge(START, "research_manager")
	sciscigpt_graph.add_conditional_edges("research_manager", select_next)
	sciscigpt_graph.add_conditional_edges("database_specialist", select_next)
	sciscigpt_graph.add_conditional_edges("analytics_specialist", select_next)
	sciscigpt_graph.add_conditional_edges("literature_specialist", select_next)
	sciscigpt_graph.add_conditional_edges("evaluation_specialist", select_next)
	sciscigpt_graph.add_conditional_edges("specialistset", select_next)
	sciscigpt_graph.add_conditional_edges("toolset", select_next)

	return sciscigpt_graph


__all__ = [
	"AgentState", "all_tools", "tools_by_name", "define_sciscigpt_graph"
]