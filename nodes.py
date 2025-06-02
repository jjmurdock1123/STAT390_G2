from langchain.hub import pull
from langchain_core.messages import AnyMessage, HumanMessage, ToolMessage, AIMessage, SystemMessage
from typing import Annotated, Dict, Any, TypedDict, Literal
from langgraph.prebuilt import ToolNode
from func.messages import reformat_messages, system_to_human, remove_inner_monologue
import os
import pickle
import sys
from uuid import uuid4
from datetime import datetime

# ───────────────────────────────────────────────────────────────────
# (A) Imports and Initialization for RAG (Pinecone + Embeddings)
# ───────────────────────────────────────────────────────────────────
import os
from pinecone import Pinecone
from langchain_openai import OpenAIEmbeddings

# 1) Set your API keys
PINECONE_API_KEY = "pcsk_6esMas_Tfb3tkmUXKj51nQX7geiCV9s4oEyxe5q8w9EVx8EMpG9tWAkk3e98fYUbTv3RV3"
PINECONE_ENV     = "us-east-1"
OPENAI_API_KEY   = "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"

os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

# 2) Initialize Pinecone v6.0.2 & connect to “energy-rag”
pc    = Pinecone(api_key=PINECONE_API_KEY, environment=PINECONE_ENV)
index = pc.Index("energy-rag")

# 3) Initialize your OpenAIEmbeddings client
embedder = OpenAIEmbeddings(model="text-embedding-ada-002")

# 4) RAG‐helper: fetch top_k similar Q&A pairs
def get_relevant_examples(user_query: str, top_k: int = 3):
    """
    Embeds the user_query, does a similarity search in Pinecone,
    and returns a list of (past_prompt, past_response) tuples.
    """
    # Embed the incoming question
    q_vec   = embedder.embed_query(user_query)
    # Query Pinecone
    results = index.query(
        vector=q_vec,
        top_k=top_k,
        include_metadata=True
    )
    # Pull out metadata fields
    examples = []
    for match in results.matches:
        md     = match.metadata
        past_q = md.get("original_prompt") or md.get("prompt", "")
        past_a = md.get("original_response") or md.get("response", "")
        examples.append((past_q, past_a))
    return examples
# ───────────────────────────────────────────────────────────────────



class Task(TypedDict):
	task: str
	loc: int
	status: Literal["created", "in_progress", "completed", "evaluated"]
	specialist: Literal["research_manager", "database_specialist", "analytics_specialist", "literature_specialist", "evaluation_specialist"]


from langgraph.graph import add_messages
class AgentState(TypedDict):
	messages: Annotated[list[AnyMessage], add_messages]
	messages_str: str
	injected_tool_args: Dict[str, Any]
	task_list: list[Task]
	end: bool

# # ────────────────────────────────────────────────────────────────────────────────
# # 2) Helper: fetch top-k prompt/response examples from Pinecone
# # ────────────────────────────────────────────────────────────────────────────────

# def get_relevant_examples(user_query: str, top_k: int = 5):
#     """
#     1) Embed `user_query` → a vector in the same space as your stored chunks.
#     2) Query Pinecone’s `energy-rag` index for the `top_k` nearest vectors.
#     3) Return a list of (prompt, response) tuples from the metadata.
#     """
#     # 1) Embed the query (returns a 1536-dim vector for ada-002)
#     q_vec = embedder.embed_query(user_query)

#     # 2) Perform vector search in Pinecone
#     results = pinecone_index.query(
#         vector=q_vec,
#         top_k=top_k,
#         include_metadata=True
#     )

#     # 3) Extract “prompt” and “response” fields from metadata
#     examples = []
#     for match in results.matches:
#         md = match.metadata
#         p = md.get("prompt", "")
#         r = md.get("response", "")
#         if p and r:
#             examples.append((p, r))
#     return examples

def ensure_proper_tool_response_sequence(messages):
	"""
	CRITICAL FIX: Ensure all AIMessages with tool_calls have corresponding ToolMessage responses
	This prevents the OpenAI API error: "assistant message with 'tool_calls' must be followed by tool messages"
	UPDATED: Handle both dictionary and object formats for tool_calls
	"""
	if not messages:
		return messages
	
	fixed_messages = []
	pending_tool_calls = {}  # track tool_call_id -> tool_call mapping
	
	for i, msg in enumerate(messages):
		fixed_messages.append(msg)
		
		# Track tool calls that need responses
		if isinstance(msg, AIMessage) and hasattr(msg, 'tool_calls') and msg.tool_calls:
			for tool_call in msg.tool_calls:
				# Handle both dictionary and object formats
				if isinstance(tool_call, dict):
					call_id = tool_call.get('id')
					if call_id:
						pending_tool_calls[call_id] = tool_call
				else:
					# Object format (has .id attribute)
					if hasattr(tool_call, 'id'):
						pending_tool_calls[tool_call.id] = tool_call
		
		# Remove handled tool calls when we see responses
		elif isinstance(msg, ToolMessage) and msg.tool_call_id in pending_tool_calls:
			del pending_tool_calls[msg.tool_call_id]
	
	# Add dummy responses for any unhandled tool calls
	for call_id, tool_call in pending_tool_calls.items():
		print(f"WARNING: Adding missing tool response for call_id: {call_id}")
		
		# Create appropriate dummy response based on tool type
		# Handle both dictionary and object formats for tool_call
		tool_name = None
		if isinstance(tool_call, dict):
			# Dictionary format
			if 'function' in tool_call and isinstance(tool_call['function'], dict):
				tool_name = tool_call['function'].get('name')
			elif 'function' in tool_call and hasattr(tool_call['function'], 'name'):
				tool_name = tool_call['function'].name
		else:
			# Object format
			if hasattr(tool_call, 'function') and tool_call.function:
				if hasattr(tool_call.function, 'name'):
					tool_name = tool_call.function.name
		
		# Generate appropriate error message
		if tool_name:
			if tool_name in ['df_list_table', 'sql_list_table']:
				dummy_content = "Error: Tool execution was interrupted - no tables listed"
			elif tool_name in ['df_get_schema', 'sql_get_schema']:
				dummy_content = "Error: Tool execution was interrupted - no schema retrieved"
			elif tool_name in ['df_query', 'sql_query']:
				dummy_content = "Error: Tool execution was interrupted - no query results"
			else:
				dummy_content = f"Error: Tool '{tool_name}' execution was interrupted"
		else:
			dummy_content = "Error: Tool execution was interrupted"
		
		dummy_response = ToolMessage(
			content=dummy_content,
			tool_call_id=call_id
		)
		fixed_messages.append(dummy_response)
	
	return fixed_messages

def call_research_manager(llm, tools, pruning_func, state: AgentState):
    """
    ResearchManager node is the first node in the agent.
    It directly interacts with the user.
    A session starts with a ResearchManager node.
    ResearchManager could assign tasks to other agents.
    ResearchManager could end a session.
    """

    import pickle
    import os

    # 1) Save current state to pickle file
    state_file = "pickle/agent_state.pkl"
    with open(state_file, "wb") as f:
        pickle.dump(state, f)

    model = llm.bind_tools(tools)

    # 2) Load your “system” messages from the LangChain hub
    system_messages = pull("liningmao/energygpt_research_manager").invoke({}).messages

    # 3) Static instruction for ResearchManager
    human_message = HumanMessage(content="""
    1. If further response is needed, assign a task to one of: database_specialist, analytics_specialist, literature_specialist
    2. If user request has been fully addressed, synthesize a final answer.
    """)

    # 4) Clean up any “inner monologue” tool messages
    cleaned_messages = remove_inner_monologue(state["messages"], ["thinking"])
    fixed_messages   = ensure_proper_tool_response_sequence(cleaned_messages)

    # ──────────────────────────────────────────────────────────
    # RAG integration: find and normalize the user’s latest HumanMessage
    # ──────────────────────────────────────────────────────────
    user_latest_question = None

    for msg in reversed(fixed_messages):
        if isinstance(msg, HumanMessage):
            raw_content = msg.content

            # If content is already a string, just strip whitespace
            if isinstance(raw_content, str):
                user_latest_question = raw_content.strip()
            # If content is a list of braces (Anthropic style), join all 'text' fields
            elif isinstance(raw_content, list):
                pieces = []
                for element in raw_content:
                    # each element might be a dict like {'type': 'text', 'text': '...'}
                    if isinstance(element, dict) and "text" in element:
                        pieces.append(element["text"])
                    # otherwise, if it’s a simple string inside the list, just append it
                    elif isinstance(element, str):
                        pieces.append(element)
                user_latest_question = "".join(pieces).strip()
            # If it’s neither a string nor a list, skip it
            else:
                continue

            # Once we’ve found and normalized the first HumanMessage, break
            break

    # ──────────────────────────────────────────────────────────
    # DEBUG #1: show exactly what we captured as the user question
    # ──────────────────────────────────────────────────────────
    print("[DEBUG] user_latest_question =", repr(user_latest_question))

    few_shot_context_message = None
    if user_latest_question:
        # 5) Pull top-5 most similar prompt/response examples from Pinecone
        examples = get_relevant_examples(user_latest_question, top_k=5)

        # ──────────────────────────────────────────────────────────
        # DEBUG #2: print out all the retrieved examples
        # ──────────────────────────────────────────────────────────
        print("[DEBUG] Retrieved RAG examples:")
        for i, (p, r) in enumerate(examples, start=1):
            print(f"  Example {i} Prompt:   {p!r}")
            print(f"  Example {i} Response: {r!r}")
            print("  ---")
        print()

        # 6) Build a single string listing those examples as few-shot context
        context_lines = []
        for i, (p, r) in enumerate(examples, start=1):
            context_lines.append(f"Example {i} Prompt:\n{p}\n")
            context_lines.append(f"Example {i} Response:\n{r}\n")
            context_lines.append("---\n")
        few_shot_text = "".join(context_lines)

        # 7) Wrap it as a SystemMessage so the LLM treats it as context
        few_shot_context_message = SystemMessage(content=few_shot_text)

    # ──────────────────────────────────────────────────────────
    # Assemble final message list, including RAG context if available
    # ──────────────────────────────────────────────────────────
    all_msgs = []
    all_msgs.extend(system_messages)

    if few_shot_context_message:
        all_msgs.append(few_shot_context_message)

    all_msgs.extend(fixed_messages)
    all_msgs.append(human_message)

    new_messages = reformat_messages(pruning_func(all_msgs))

    print("\n[DEBUG] Final messages sent to LLM:")
    for msg in new_messages:
    	print(f"  role={msg.type}, content={repr(msg.content)}\n")
    # ──────────────────────────────────────────────────────────
    # Invoke the LLM on the augmented message list
    # ──────────────────────────────────────────────────────────
    try:
        response = model.invoke(new_messages)
    except Exception as e:
        print(f"ResearchManager invoke error: {e}")
        response = AIMessage(content="I encountered an issue processing your request. Let me try a different approach.")

    # ──────────────────────────────────────────────────────────
    # If the model returned a tool call, hand off to a specialist
    # ──────────────────────────────────────────────────────────
    if response.tool_calls:
        specialist = response.tool_calls[0]["name"]
        task       = response.tool_calls[0]["args"].get("task", "")
        task_loc   = len(state["messages"]) + 2  # +2 for the AI message itself

        new_task = {
            "task": task,
            "loc": task_loc,
            "status": "created",
            "specialist": specialist
        }
        return {
            "messages": [response],
            "end": False,
            "task_list": state.get("task_list", []) + [new_task]
        }
    else:
        # No tool calls → end the session
        return {"messages": [response], "end": True}


# def call_research_manager(llm, tools, pruning_func, state: AgentState):
#     """ResearchManager node is the first node in the agent.
#     It directly interacts with the user.
#     A session starts with a ResearchManager node.
#     ResearchManager could assign tasks to other agents.
#     ResearchManager could end a session."""

#     import pickle
#     import os

#     # Save current state to pickle file
#     state_file = "pickle/agent_state.pkl"
#     with open(state_file, "wb") as f:
#         pickle.dump(state, f)

#     model = llm.bind_tools(tools)

#     system_messages = pull("liningmao/energygpt_research_manager").invoke({}).messages
#     human_message = HumanMessage(content="""
#     1. If further response is needed, assign a task to one of: database_specialist, analytics_specialist, literature_specialist
#     2. If user request has been fully addressed, synthesize a final answer.""")

#     # CRITICAL FIX: Ensure proper tool call/response pairing before processing
#     cleaned_messages = remove_inner_monologue(state['messages'], ["thinking"])
#     fixed_messages = ensure_proper_tool_response_sequence(cleaned_messages)

#     # # ──────────────────────────────────────────────────────────
#     # # RAG integration: find the user’s latest HumanMessage
#     # # ──────────────────────────────────────────────────────────
#     # user_latest_question = None
#     # for msg in reversed(fixed_messages):
#     #     if isinstance(msg, HumanMessage):
#     #         user_latest_question = msg.content.strip()
#     #         break

#     # few_shot_context_message = None
#     # if user_latest_question:
#     #     # Pull the top-5 most similar prompt/response examples
#     #     examples = get_relevant_examples(user_latest_question, top_k=5)

#     #     # ─── DEBUG: show which RAG examples were retrieved ───
#     #     print("\n[DEBUG] Retrieved RAG examples for question:")
#     #     print("  Question →", user_latest_question)
#     #     for i, (p, r) in enumerate(examples, start=1):
#     #         print(f"  Example {i} Prompt:   {p}")
#     #         print(f"  Example {i} Response: {r}")
#     #         print("  ---")
#     #     print()
#     #     # ────────────────────────────────────────────────────────

#     #     # Build a single string that lists them; each “Example i” has prompt+response
#     #     context_lines = []
#     #     for i, (p, r) in enumerate(examples, start=1):
#     #         context_lines.append(f"Example {i} Prompt:\n{p}\n")
#     #         context_lines.append(f"Example {i} Response:\n{r}\n")
#     #         context_lines.append("---\n")
#     #     few_shot_text = "".join(context_lines)

#     #     # Wrap as a SystemMessage so the LLM treats it as context
#     #     few_shot_context_message = SystemMessage(content=few_shot_text)

#     # # ──────────────────────────────────────────────────────────
#     # # Assemble final message list
#     # # ──────────────────────────────────────────────────────────
#     # all_msgs = []
#     # all_msgs.extend(system_messages)

#     # if few_shot_context_message:
#     #     all_msgs.append(few_shot_context_message)

#     # all_msgs.extend(fixed_messages)
#     # all_msgs.append(human_message)

#     new_messages = reformat_messages(
#         reformat_messages(pruning_func(all_msgs))
#     )

#     try:
#         response = model.invoke(new_messages)
#     except Exception as e:
#         print(f"ResearchManager invoke error: {e}")
#         response = AIMessage(content="I encountered an issue processing your request. Let me try a different approach.")

#     if response.tool_calls:
#         specialist = response.tool_calls[0]["name"]
#         task = response.tool_calls[0]["args"].get("task", "")
#         task_loc = len(state['messages']) + 2  # +2 for the AI message itself
#         # Assign a task to a specialist
#         new_task = {
#             "task": task,
#             "loc": task_loc,
#             "status": "created",
#             "specialist": specialist
#         }
#         return {
#             "messages": [response],
#             "end": False,
#             "task_list": state.get("task_list", []) + [new_task]
#         }
#     else:
#         # End the session
#         return {"messages": [response], "end": True}



def call_specialist(llm, tools, pruning_func, state: AgentState):
	newest_task = state["task_list"][-1]
	specialist, task, task_loc = newest_task["specialist"], newest_task["task"], newest_task["loc"]
	model = llm.bind_tools(tools)

	system_messages = pull(f"liningmao/energygpt_{specialist}").invoke({"task": task}).messages
	human_message = HumanMessage(content=f"<user>Here is the task you need to perform, follow the guidelines in the system message.<task>{task}</task></user>")

	# CRITICAL FIX: Ensure proper tool call/response pairing before processing
	specialist_messages = state['messages'][task_loc:]  # The workflow of the specialist
	fixed_specialist_messages = ensure_proper_tool_response_sequence(specialist_messages)
	
	try:
		response = model.invoke(reformat_messages(pruning_func([
			*system_messages,
			# *remove_inner_monologue(state['messages'][:task_loc], ["thinking"]), # Everything before the task
			# *system_to_human(system_messages), 
			human_message,
			*fixed_specialist_messages  # Use fixed messages instead of original
		])))
	except Exception as e:
		print(f"Specialist {specialist} invoke error: {e}")
		# Return a fallback response
		response = AIMessage(content=f"I encountered an issue while working on the task: {task}. Let me try a different approach.")

	if response.tool_calls and response.tool_calls[0]["name"] != "evaluation_specialist":
		# Continue the task (reasoning - tool call iteration)
		newest_task["status"] = "in_progress"
		return { "messages": [response], "task_list": state["task_list"] }	
	else:
		# End this task, pass to evaluation_specialist
		newest_task["status"] = "completed"
		
		# Handle both OpenAI (string) and Anthropic (list) response formats
		if isinstance(response.content, str):
			# OpenAI format - content is already a string
			response.tool_calls = []
			return { "messages": [response], "task_list": state["task_list"] } 
		elif isinstance(response.content, list) and len(response.content) > 0 and response.content[0].get("type") == "text":
			# Anthropic format - content is a list with type info
			response.tool_calls = []
			response.content = response.content[0]["text"]
			return { "messages": [response], "task_list": state["task_list"] } 
		else:
			# Fallback for other formats
			return { "task_list": state["task_list"] } 

import json
from reasoning.visual_evaluation import visual_evaluate
from reasoning.tool_evaluation import check_tool_args, tool_evaluate
from langchain_core.messages import ToolMessage

# Add this to your call_tool function in agents/nodes.py for debugging:
'''
def call_tool(tools_by_name, state: AgentState):
    for tool_call in state["messages"][-1].tool_calls:
        tool_name, tool_id, tool_args = tool_call["name"], tool_call["id"], tool_call["args"]
        tool = tools_by_name[tool_name]
        
        response = tool.invoke(tool_args)
        response = json.loads(response) if isinstance(response, str) else response

        # DEBUG: Print response structure
        print(f"DEBUG - Tool: {tool_name}")
        print(f"DEBUG - Response type: {type(response)}")
        print(f"DEBUG - Response content: {response}")
        
        # POTENTIAL FIX: Ensure response is properly serializable
        if isinstance(response, tuple) and len(response) == 2:
            # DataFrame tools return (response, response) - use first element
            response = response[0]
        
        # Ensure response is a dictionary
        if not isinstance(response, dict):
            response = {"response": str(response)}
        
        # Create serializable content
        try:
            serialized_response = json.dumps(response)
            content = [{"type": "text", "text": serialized_response}]
        except (TypeError, ValueError) as e:
            print(f"DEBUG - Serialization error: {e}")
            # Fallback to string representation
            content = [{"type": "text", "text": str(response)}]
        
        tool_message = ToolMessage(content=content, tool_call_id=tool_id)
        print(f"DEBUG - ToolMessage content: {tool_message.content}")
        
    return { "messages": [tool_message] }
'''
from langchain_core.messages import ToolMessage
import json

def call_tool(tools_by_name, state: AgentState):
    latest_msg = state["messages"][-1]

    # If no tool calls present, do nothing
    if not hasattr(latest_msg, "tool_calls") or not latest_msg.tool_calls:
        return state

    tool_messages = []

    for tool_call in latest_msg.tool_calls:
        # Support both dict and object formats
        if isinstance(tool_call, dict):
            tool_name = tool_call.get("name") or tool_call.get("function", {}).get("name")
            tool_args = tool_call.get("args") or tool_call.get("function", {}).get("arguments", "{}")
            tool_id = tool_call.get("id")
        else:
            tool_name = tool_call.function.name
            tool_args = tool_call.function.arguments
            tool_id = tool_call.id

        tool_args = json.loads(tool_args) if isinstance(tool_args, str) else tool_args

        tool = tools_by_name.get(tool_name)
        if tool is None:
            content = [{"type": "text", "text": f"Tool '{tool_name}' not found."}]
        else:
            try:
                result = tool.invoke(tool_args)
                if isinstance(result, tuple):
                    result = result[0]
                result = json.dumps(result) if not isinstance(result, str) else result
                content = [{"type": "text", "text": result}]
            except Exception as e:
                content = [{"type": "text", "text": f"Tool '{tool_name}' failed: {str(e)}"}]

        tool_messages.append(ToolMessage(content=content, tool_call_id=tool_id))

    # ✅ Return fully updated AgentState
    return {
        "messages": [*state["messages"], *tool_messages],
        "messages_str": state["messages_str"],
        "injected_tool_args": state["injected_tool_args"],
        "task_list": state.get("task_list", []),
        "end": state.get("end", False)
    }

