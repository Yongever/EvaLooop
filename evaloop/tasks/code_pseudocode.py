import re
from typing import Any, Dict

from .code_summarization import CodeSummarizationTask
from .factory import register_task


@register_task("code_pseudocode")
class CodePseudocodeTask(CodeSummarizationTask):
    """Inverse task of code generation that describes code as pseudocode.

    Used by the code <-> pseudocode loop: the pseudocode replaces the
    one-sentence summary as the next cycle's generation input. Everything
    else is inherited from CodeSummarizationTask on purpose, in particular
    postprocess, which wraps the output as a docstring and re-attaches the
    cycle's assert line so the function name and signature stay anchored
    exactly as in the standard loop.
    """

    DEFAULT_PROMPT_TEMPLATE = (
        "Rewrite the following code as language-agnostic pseudocode that describes its "
        "algorithm step by step (use keywords such as FUNCTION, IF, FOR EACH, WHILE, RETURN). "
        "Do not write Python or any other real programming language.\n\n"
        "```\n{code}\n```\n\n"
        "Return only the pseudocode, without explanations."
    )

    _FENCE_PATTERN = re.compile(r"```[^\n]*\n([\s\S]*?)```")

    def create_prompt(self, input_text: str) -> str:
        """Create a prompt asking for pseudocode of the given code."""
        prompt_template = self.task_config.get("prompt_template", self.DEFAULT_PROMPT_TEMPLATE)
        return prompt_template.format(code=input_text)

    def extract_output(self, raw_response: str) -> str:
        """Extract the pseudocode, unwrapping the first fenced block if there is one."""
        match = self._FENCE_PATTERN.search(raw_response)
        if match:
            return match.group(1).strip()
        return raw_response.strip()

    def postprocess(self, output: str, context: Dict[str, Any]) -> str:
        """Neutralize triple quotes, then wrap like a summary (keeps the assert line).

        Pseudocode may quote strings with triple quotes, which would close the
        docstring wrapper early and change which line sits at index [-2].
        """
        return super().postprocess(output.replace('"""', "'''"), context)
