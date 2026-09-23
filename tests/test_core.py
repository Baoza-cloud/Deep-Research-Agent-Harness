# ruff: noqa: F403, F405

from tests._support import *


class PackageEntryPointTests(unittest.TestCase):
    def test_package_module_exposes_cli_help(self):
        project_root = Path(__file__).resolve().parents[1]
        env = os.environ.copy()
        env["PYTHONPATH"] = str(project_root / "src")
        completed = subprocess.run(
            [sys.executable, "-m", "research_engine", "--help"],
            cwd=project_root,
            env=env,
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertIn("Run the multi-agent deep research engine", completed.stdout)

    def test_pyproject_registers_console_script(self):
        project_root = Path(__file__).resolve().parents[1]
        pyproject = (project_root / "pyproject.toml").read_text(encoding="utf-8")
        self.assertIn('deep-research = "research_engine.cli:main"', pyproject)
        self.assertIn('package-dir = {"" = "src"}', pyproject)


class StateTests(unittest.TestCase):
    def test_nine_states_exist(self):
        self.assertEqual(len(TaskStatus), 9)

    def test_dag_progression_and_invalid_transition(self):
        plan = ResearchPlan(
            "q",
            "o",
            [
                ResearchSubtask("a", "a", "a"),
                ResearchSubtask("b", "b", "b", dependencies=["a"]),
            ],
        )
        state = ResearchRunState(plan)
        ready = state.refresh_ready()
        self.assertEqual([item.spec.subtask_id for item in ready], ["a"])
        ready[0].transition(TaskStatus.RUNNING)
        ready[0].transition(TaskStatus.SUCCEEDED)
        self.assertEqual(state.refresh_ready()[0].spec.subtask_id, "b")
        with self.assertRaises(StateTransitionError):
            ready[0].transition(TaskStatus.RUNNING)

    def test_cycle_is_rejected(self):
        plan = ResearchPlan(
            "q",
            "o",
            [
                ResearchSubtask("a", "a", "a", dependencies=["b"]),
                ResearchSubtask("b", "b", "b", dependencies=["a"]),
            ],
        )
        with self.assertRaises(ValueError):
            ResearchRunState(plan)


class ParsingTests(unittest.TestCase):
    def test_three_json_shapes(self):
        self.assertEqual(parse_json_payload('{"a": 1}')["a"], 1)
        self.assertEqual(parse_json_payload('```json\n{"a": 2}\n```')["a"], 2)
        self.assertEqual(parse_json_payload("prefix {'a': 3} suffix")["a"], 3)

    def test_vllm_backend_has_local_default(self):
        config = backend_config(LLMProvider.VLLM, model="fixture-model")
        self.assertEqual(config.base_url, "http://127.0.0.1:8000/v1")
        self.assertEqual(config.api_key, "EMPTY")
        self.assertEqual(config.request_timeout_seconds, 45.0)
        self.assertEqual(config.max_retries, 1)

    def test_env_loader_does_not_override_existing_values(self):
        import os

        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / ".env"
            path.write_text("RESEARCH_TEST_KEY=from_file\n", encoding="utf-8")
            os.environ["RESEARCH_TEST_KEY"] = "existing"
            load_env_files([path])
            self.assertEqual(os.environ["RESEARCH_TEST_KEY"], "existing")
            del os.environ["RESEARCH_TEST_KEY"]


class PlannerTests(unittest.IsolatedAsyncioTestCase):
    async def test_llm_plan_over_eight_tasks_falls_back_to_bounded_dag(self):
        async def oversized_plan(prompt):
            import json

            return json.dumps(
                {
                    "objective": "oversized",
                    "subtasks": [
                        {
                            "id": f"t{i}",
                            "question": f"q{i}",
                            "reason": "r",
                            "dependencies": [],
                        }
                        for i in range(9)
                    ],
                }
            )

        plan = await LLMPlanner(oversized_plan).plan("bounded planning")
        self.assertEqual(len(plan.subtasks), 4)
