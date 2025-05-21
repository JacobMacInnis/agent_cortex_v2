import re

class CalculatorTool:
    """
    A simple calculator tool that safely evaluates basic math expressions.

    Supports:
    - Integers and decimals
    - +, -, *, /, parentheses

    Will return an error message if the expression is invalid.
    """

    def evaluate(self, input_text: str) -> str:
        # Extract math-like expressions from the input
        expression = self._extract_expression(input_text)

        try:
            result = eval(expression, {"__builtins__": {}})
            return f"🧮 Result: {result}"
        except Exception as e:
            return f"⚠️ Could not evaluate expression: {expression} — {e}"

    def _extract_expression(self, text: str) -> str:
        # Strip all characters except digits, ops, decimals, and parentheses
        return re.sub(r"[^0-9\.\+\-\*\/\(\)]", "", text)
