from langchain.tools import BaseTool
from langchain_experimental.utilities import PythonREPL
from typing import Optional, Dict, Any, Type
from pydantic import BaseModel, Field

class PythonREPLInput(BaseModel):
    code: str = Field(description="The Python code to execute")

class PythonREPLTool(BaseTool):
    name: str = "python_repl"
    description: str = "Executes Python code using a local REPL."
    args_schema: Type[BaseModel] = PythonREPLInput # type: ignore

    def __init__(self, **kwargs: Any):
        super().__init__(**kwargs)
        self._repl = PythonREPL()

    def _run(self, code: Optional[str] = None, **kwargs: Dict[str, Any]) -> str:
        return self._repl.run(code or str(kwargs.get("code", "")))

    async def _arun(self, *args: Any, **kwargs: Any) -> str:
        raise NotImplementedError("Async not supported.")
