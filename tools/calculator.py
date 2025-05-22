import re
from typing import Optional, Dict, Any, Type
from pydantic import BaseModel, Field
from langchain.tools import BaseTool


class CalculatorInput(BaseModel):
    input_text: str = Field(description="A math expression like 2 + 2 or (3 * 4) / 2")


class CalculatorTool(BaseTool):
    name: str = "Calculator"
    description: str = "Evaluates basic math expressions with +, -, *, /, and parentheses."
    args_schema: Type[BaseModel] = CalculatorInput # type: ignore

    def _run(self, input_text: Optional[str] = None, **kwargs: Dict[str, Any]) -> str:
        expression = input_text or str(kwargs.get("input_text", ""))
        return self.evaluate(expression)

    async def _arun(self, *args: Any, **kwargs: Any) -> str:
        raise NotImplementedError("Async not supported.")

    def evaluate(self, input_text: str) -> str:
        expression = self._extract_expression(input_text)

        try:
            result = eval(expression, {"__builtins__": {}})
            return f"🧮 Result: {result}"
        except Exception as e:
            return f"⚠️ Could not evaluate expression: {expression} — {e}"

    def _extract_expression(self, text: str) -> str:
        return re.sub(r"[^0-9\.\+\-\*\/\(\)]", "", text)
