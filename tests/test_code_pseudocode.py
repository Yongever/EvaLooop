"""Unit tests for the code <-> pseudocode loop and the Vertex AI Gemini backend."""

import sys
import types as pytypes

import pytest

from evaloop.core.config import EvaluationConfig
from evaloop.evaluation.cycle import BatchEvaluationCycle
from evaloop.tasks import CodeGenerationTask, CodePseudocodeTask, TaskFactory

from conftest import ASSERT_LINE, MBPP_STYLE_PROMPT

PSEUDOCODE = (
    "FUNCTION similar_elements(a, b)\n"
    "    RETURN set of items in both a and b\n"
    "END FUNCTION"
)


class TestCodePseudocodeTask:
    def test_registered(self):
        assert isinstance(TaskFactory().create_task({"type": "code_pseudocode"}), CodePseudocodeTask)

    def test_create_prompt_embeds_code(self):
        prompt = CodePseudocodeTask({}).create_prompt("def f(x): return x")
        assert "def f(x): return x" in prompt
        assert "pseudocode" in prompt
        assert "Do not write Python" in prompt

    def test_create_prompt_uses_template(self):
        task = CodePseudocodeTask({"prompt_template": "PSEUDO:\n{code}"})
        assert task.create_prompt("x = 1") == "PSEUDO:\nx = 1"

    def test_extract_output_unwraps_fence(self):
        response = f"Here it is:\n```pseudocode\n{PSEUDOCODE}\n```\n"
        assert CodePseudocodeTask({}).extract_output(response) == PSEUDOCODE

    def test_extract_output_no_fence_returns_stripped_response(self):
        assert CodePseudocodeTask({}).extract_output(f"  {PSEUDOCODE}  \n") == PSEUDOCODE

    def test_postprocess_retains_assert_line(self):
        result = CodePseudocodeTask({}).postprocess(
            PSEUDOCODE, {"cycle_input": MBPP_STYLE_PROMPT, "task_id": "Mbpp/2", "cycle_number": 1}
        )
        assert result == f'"""\n{PSEUDOCODE}\n{ASSERT_LINE}\n"""\n'

    def test_postprocess_output_keeps_assert_on_next_cycle(self):
        # The wrapped output becomes the next cycle input; its [-2] line must still be the assert.
        task = CodePseudocodeTask({})
        first = task.postprocess(PSEUDOCODE, {"cycle_input": MBPP_STYLE_PROMPT})
        second = task.postprocess(PSEUDOCODE, {"cycle_input": first})
        assert second.splitlines()[-2] == ASSERT_LINE

    def test_postprocess_neutralizes_triple_quotes(self):
        result = CodePseudocodeTask({}).postprocess(
            'PRINT """hi"""', {"cycle_input": MBPP_STYLE_PROMPT}
        )
        assert result.count('"""') == 2
        assert result.splitlines()[-2] == ASSERT_LINE

    def test_is_not_testable(self):
        assert CodePseudocodeTask({}).produces_testable_code is False


class TestPseudocodeLoopConfig:
    def test_task_configs(self, tmp_path):
        config = EvaluationConfig(
            model="gemini-3.5-flash-lite",
            loop_type="code_generation_pseudocode",
            output_dir=str(tmp_path),
        )
        task_configs = config.get_task_configs()
        assert [c["type"] for c in task_configs] == ["code_generation", "code_pseudocode"]
        # The generation step is identical to the standard loop's
        standard = config.get_task_configs("code_generation_summarization")
        assert task_configs[0] == standard[0]

    def test_unknown_loop_type_raises(self, tmp_path):
        config = EvaluationConfig(model="gpt-4", loop_type="nope", output_dir=str(tmp_path))
        with pytest.raises(ValueError):
            config.get_task_configs()

    @pytest.mark.parametrize("model", ["gemini-3.5-flash-lite", "google/gemini-3.5-flash"])
    def test_gemini_routes_to_vertex(self, tmp_path, model):
        config = EvaluationConfig(
            model=model, gcp_project="my-project", output_dir=str(tmp_path)
        )
        model_config = config.get_model_config()
        assert model_config["type"] == "vertex_gemini"
        assert model_config["model_id"] == model.split("/")[-1]
        assert model_config["project"] == "my-project"
        assert model_config["thinking_budget"] == 0
        assert model_config["temperature"] == 0.0

    def test_loop_runs_with_mocked_model(self):
        class ScriptedLLM:
            def __init__(self):
                self.prompts = []

            def generate(self, prompt, **kwargs):
                self.prompts.append(prompt)
                if prompt.startswith("Generate"):
                    return "```python\ndef similar_elements(a, b):\n    return set(a) & set(b)\n```"
                return f"```\n{PSEUDOCODE}\n```"

        config = EvaluationConfig.__new__(EvaluationConfig)
        config.loop_type = "code_generation_pseudocode"
        tasks = [TaskFactory().create_task(c) for c in EvaluationConfig.get_task_configs(config)]
        model = ScriptedLLM()
        out = BatchEvaluationCycle(model, tasks).run_single_cycle(
            [{"task_id": "Mbpp/2", "prompt": MBPP_STYLE_PROMPT}], 1
        )
        result = out["cycle_results"][0]
        assert isinstance(tasks[0], CodeGenerationTask)
        assert result["task_1_output"].startswith("def similar_elements")
        assert result["task_2_output"] == f'"""\n{PSEUDOCODE}\n{ASSERT_LINE}\n"""\n'


class _APIError(Exception):
    """Mimics google.genai.errors.APIError, which carries the HTTP status as .code."""

    def __init__(self, code):
        super().__init__(f"{code} error")
        self.code = code


class _FakeResponse:
    def __init__(self, text):
        self.text = text
        self.candidates = []
        self.usage_metadata = pytypes.SimpleNamespace(
            prompt_token_count=10, candidates_token_count=5, thoughts_token_count=None
        )


class _FakeModels:
    def __init__(self, replies):
        self.replies = list(replies)
        self.calls = []

    def generate_content(self, model, contents, config):
        self.calls.append((model, contents, config))
        reply = self.replies.pop(0) if self.replies else f"echo:{contents}"
        if isinstance(reply, Exception):
            raise reply
        return _FakeResponse(reply)


@pytest.fixture
def fake_genai(monkeypatch):
    """Install a stand-in google.genai so the backend can be tested offline."""
    state = {"replies": []}

    class FakeClient:
        def __init__(self, vertexai, project, location):
            assert vertexai is True
            self.models = _FakeModels(state["replies"])
            state["client"] = self

    genai_mod = pytypes.ModuleType("google.genai")
    genai_mod.Client = FakeClient
    types_mod = pytypes.ModuleType("google.genai.types")
    types_mod.GenerateContentConfig = lambda **kw: kw
    types_mod.ThinkingConfig = lambda **kw: kw
    genai_mod.types = types_mod
    google_mod = pytypes.ModuleType("google")
    google_mod.genai = genai_mod
    monkeypatch.setitem(sys.modules, "google", google_mod)
    monkeypatch.setitem(sys.modules, "google.genai", genai_mod)
    monkeypatch.setitem(sys.modules, "google.genai.types", types_mod)
    return state


class TestVertexGeminiLLM:
    def _make(self, **overrides):
        from evaloop.models.vertex_gemini import VertexGeminiLLM

        config = {"name": "g", "model_id": "gemini-3.5-flash-lite", "project": "p"}
        config.update(overrides)
        llm = VertexGeminiLLM(config)
        llm.retry_delay = 0
        return llm

    def test_requires_project(self, fake_genai, monkeypatch):
        monkeypatch.delenv("GOOGLE_CLOUD_PROJECT", raising=False)
        with pytest.raises(ValueError):
            self._make(project=None)

    def test_greedy_config_and_thinking_off(self, fake_genai):
        fake_genai["replies"] = ["hello"]
        llm = self._make()
        assert llm.generate("hi") == "hello"
        model, contents, config = fake_genai["client"].models.calls[0]
        assert model == "gemini-3.5-flash-lite"
        assert config["temperature"] == 0.0
        assert config["thinking_config"] == {"thinking_budget": 0}

    def test_retries_on_error_and_empty(self, fake_genai):
        fake_genai["replies"] = [_APIError(429), _APIError(503), "", "ok"]
        assert self._make().generate("hi") == "ok"

    def test_setup_errors_raise(self, fake_genai):
        fake_genai["replies"] = [_APIError(404)]
        with pytest.raises(_APIError):
            self._make().generate("hi")

    def test_other_client_errors_not_retried(self, fake_genai):
        fake_genai["replies"] = [_APIError(400), "never reached"]
        llm = self._make()
        assert llm.generate("hi").startswith("Error:")
        assert len(fake_genai["client"].models.calls) == 1

    def test_gives_up_on_repeated_empty_responses(self, fake_genai):
        fake_genai["replies"] = ["", "", "", "never reached"]
        llm = self._make()
        assert llm.generate("hi").startswith("Error: Empty response")
        assert len(fake_genai["client"].models.calls) == llm.max_empty_retries

    def test_usage_is_accumulated(self, fake_genai):
        llm = self._make(max_workers=4)
        llm.batch_generate([f"p{i}" for i in range(6)])
        assert llm.usage == {
            "requests": 6, "input_tokens": 60, "output_tokens": 30, "thinking_tokens": 0
        }

    def test_batch_generate_preserves_order(self, fake_genai):
        prompts = [f"p{i}" for i in range(20)]
        assert self._make(max_workers=4).batch_generate(prompts) == [f"echo:{p}" for p in prompts]
