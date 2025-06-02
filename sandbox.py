from langchain.tools import BaseTool

import re, os, json, uuid, base64

import matplotlib.pyplot as plt


from pydantic import BaseModel, Field
from typing import Any, Dict, Optional, Sequence, Type, Union
from langchain_core.tools import InjectedToolArg
from typing_extensions import Annotated

# from func.r import execute_r_code
from func.python import execute_python_code, get_images
from func.image import upload_image

class RShellInput(BaseModel):
	query: str = Field(..., description="Python code snippet to run")
class RShellTool(BaseTool):
	name: str = "r"
	description: str = """Execute Python code in a persistent R environment. Input: Any valid R code snippet to run. Output: Standard output and error messages. Note: you need to call `print(p)` to render the figure."""
	response_format: str = "content_and_artifact"
	args_schema: Type[BaseModel] = RShellInput

	session_id: str = "test"
	envir_dict: dict = {}

	def _run(self, query: str) -> str:
		if self.session_id not in self.envir_dict:
			self.envir_dict[self.session_id] = ro.r("new.env()")
		
		response = execute_python_code(query, self.envir_dict[self.session_id])
		return response, response


from func.jupyter import JupyterSandbox
class PythonShellInput(BaseModel):
	query: str = Field(..., description="Python code snippet to run")

class PythonShellTool(BaseTool):
	name: str = "python"
	description: str = """Execute Python code in a persistent Jupyter environment. Input: Any valid Python code snippet to run. Output: Standard output, error messages, and output images. Note: Don't save output images to disk. Output images will be rendered automatically."""
	response_format: str = "content_and_artifact"
	args_schema: Type[BaseModel] = PythonShellInput

	sandbox: JupyterSandbox = None
	session_id: str = "test"

	def _run(self, query: str="""print("Error: Missing required [query] parameter")""") -> str:
		session_id = "test" #uuid.uuid4()
		cell_id = uuid.uuid4()
		results = self.sandbox.execute_code(query, session_id=session_id, cell_id=cell_id, timeout=120)
		results = [r for r in results if r["session_id"] == session_id and r["cell_id"] == cell_id]

		text_responses = [r for r in results if r['type'] == 'text']
		image_responses = [r for r in results if r['type'] == 'image_url']

		response = {}
		response["response"] = "".join([r["text"] for r in text_responses])

		if len(image_responses) > 0:
			response["images"] = []
			for i in image_responses:
				id = str(uuid.uuid4())
				name = f"data/output/{id}.png"

				with open(name, "wb") as f:
					f.write(base64.b64decode(i["image_url"]["url"].split(",")[1]))
				print(name, os.path.exists(name))
				download_link = upload_image(name)
				
				response["images"].append(
					{ "name": name, "id": id, "mime_type": "image/png", "download_link": download_link })
		return response, response


# from func.env import python_env_setup, python_env_setup_string
# python_env_setup() # Setup the python environment in system level
jupyter_sandbox = JupyterSandbox()
python_shell_tool = PythonShellTool(sandbox=jupyter_sandbox)

# import rpy2.robjects as ro
# r_shell_tool = RShellTool()