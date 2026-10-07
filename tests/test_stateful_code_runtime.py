import json
import subprocess
import tempfile
import unittest
from datetime import datetime, timedelta, timezone

from aibridge_exec.code_runtime import (
    ArtifactMount,
    CodeRuntimeSettings,
    CodeExecutionRuntime,
    ExecutionRequest,
    ExpectedOutput,
    LocalProcessBackend,
    PackageInstallRequest,
    SessionManager,
)
from aibridge_exec.code_runtime.package_manager import _classify_pip_error


class TestStatefulCodeRuntime(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp_dir = tempfile.TemporaryDirectory(
            prefix="stateful-code-runtime-tests-"
        )
        cls.backend = LocalProcessBackend(
            base_dir=cls.temp_dir.name,
            max_output_bytes=2048,
        )
        cls.sessions = SessionManager(
            cls.backend,
            ttl_seconds=3600,
            max_workspace_bytes=100 * 1024 * 1024,
        )
        cls.runtime = CodeExecutionRuntime(
            backend=cls.backend,
            session_manager=cls.sessions,
        )

    @classmethod
    def tearDownClass(cls):
        cls.temp_dir.cleanup()

    def test_reuses_files_across_executions(self):
        first = self.runtime.execute(
            ExecutionRequest(
                session_id="reuse-files",
                language="python",
                code=(
                    "from pathlib import Path\n"
                    "Path('state/value.txt').write_text('persisted')\n"
                    "Path('outputs/first.txt').write_text('first')\n"
                ),
                expected_outputs=(ExpectedOutput("outputs/first.txt"),),
                timeout_seconds=10,
            )
        )
        second = self.runtime.execute(
            ExecutionRequest(
                session_id="reuse-files",
                language="python",
                code=(
                    "from pathlib import Path\n"
                    "value = Path('state/value.txt').read_text()\n"
                    "Path('outputs/second.txt').write_text(value)\n"
                ),
                expected_outputs=(ExpectedOutput("second.txt"),),
                timeout_seconds=10,
            )
        )

        self.assertTrue(first.succeeded)
        self.assertTrue(second.succeeded)
        self.assertEqual(second.artifacts[0].content, b"persisted")

    def test_materializes_inputs_and_collects_expected_output(self):
        result = self.runtime.execute(
            ExecutionRequest(
                session_id="artifact-flow",
                language="python",
                code=(
                    "from pathlib import Path\n"
                    "value = Path('inputs/source.txt').read_text()\n"
                    "Path('outputs/result.txt').write_text(value.upper())\n"
                ),
                input_artifacts=(
                    ArtifactMount(
                        artifact_ref="file://source",
                        target_path="inputs/source.txt",
                        content=b"hello",
                        media_type="text/plain",
                    ),
                ),
                expected_outputs=(
                    ExpectedOutput("outputs/result.txt", "text/plain"),
                ),
                timeout_seconds=10,
            )
        )

        self.assertTrue(result.succeeded)
        self.assertEqual(result.artifacts[0].content, b"HELLO")
        self.assertEqual(result.artifacts[0].media_type, "text/plain")

    def test_normalizes_recoverable_execution_errors(self):
        missing = self.runtime.execute(
            ExecutionRequest(
                session_id="errors",
                language="python",
                code="import package_that_does_not_exist_aibridge\n",
                timeout_seconds=10,
            )
        )
        syntax = self.runtime.execute(
            ExecutionRequest(
                session_id="errors",
                language="python",
                code="if True print('broken')",
                timeout_seconds=10,
            )
        )
        missing_output = self.runtime.execute(
            ExecutionRequest(
                session_id="errors",
                language="python",
                code="print('done')",
                expected_outputs=(ExpectedOutput("outputs/required.csv"),),
                timeout_seconds=10,
            )
        )

        self.assertEqual(missing.error.category, "missing_dependency")
        self.assertEqual(
            missing.error.suggested_package,
            "package_that_does_not_exist_aibridge",
        )
        self.assertEqual(syntax.error.category, "syntax_error")
        self.assertEqual(
            missing_output.error.category, "expected_output_missing"
        )

    def test_timeout_terminates_execution_as_structured_result(self):
        result = self.runtime.execute(
            ExecutionRequest(
                session_id="timeout",
                language="python",
                code="import time\ntime.sleep(5)",
                timeout_seconds=1,
            )
        )

        self.assertEqual(result.status, "failed")
        self.assertEqual(result.error.category, "timeout")
        self.assertTrue(result.error.retryable)

    def test_stdout_capture_is_bounded(self):
        result = self.runtime.execute(
            ExecutionRequest(
                session_id="bounded-output",
                language="python",
                code="print('x' * 10000)",
                timeout_seconds=10,
            )
        )

        self.assertTrue(result.succeeded)
        self.assertTrue(result.stdout_truncated)
        self.assertIn("output truncated", result.stdout)
        self.assertLess(len(result.stdout), 2200)

    def test_supplies_preloaded_stdin_and_streams_output(self):
        chunks = []
        result = self.runtime.execute(
            ExecutionRequest(
                session_id="stdin-stream",
                language="python",
                code="name = input()\nprint(f'hello {name}', flush=True)\n",
                stdin="A1\n",
                timeout_seconds=10,
            ),
            output_callback=lambda stream, data: chunks.append((stream, data)),
        )

        self.assertTrue(result.succeeded)
        self.assertEqual(result.stdout, "hello A1\n")
        self.assertEqual(
            b"".join(data for stream, data in chunks if stream == "stdout"),
            b"hello A1\n",
        )

    def test_package_validation_rejects_urls_and_accepts_available_package(self):
        blocked = self.runtime.install_packages(
            PackageInstallRequest(
                session_id="packages",
                packages=("https://example.test/package.whl",),
            )
        )
        available = self.runtime.install_packages(
            PackageInstallRequest(
                session_id="packages",
                packages=("pip>=20",),
                timeout_seconds=10,
            )
        )

        self.assertEqual(blocked.error.category, "package_blocked")
        self.assertEqual(available.status, "success")
        self.assertTrue(
            available.installed or available.already_installed
        )
        session = self.sessions.get_or_create("packages")
        manifest = json.loads(
            (session.root / "package-manifest.json").read_text(encoding="utf-8")
        )
        self.assertEqual(manifest["schema_version"], 1)
        self.assertEqual(manifest["runtime_image_digest"], "local-process")
        self.assertIn("python_abi", manifest)
        self.assertIn("pip", manifest["requested_packages"])
        self.assertIn("pip", manifest["resolved_packages"])
        self.assertIn("pip==", (session.root / "requirements.lock").read_text())

    def test_session_venv_is_isolated_from_system_packages(self):
        session = self.sessions.get_or_create("isolated-venv")
        config_path = session.root / "venv" / "pyvenv.cfg"

        self.assertIn(
            "include-system-site-packages = false",
            config_path.read_text(encoding="utf-8"),
        )
        interpreter_version = subprocess.run(
            [
                str(session.interpreter),
                "-c",
                "import platform; print(platform.python_version())",
            ],
            stdout=subprocess.PIPE,
            check=True,
            text=True,
        ).stdout.strip()
        manifest = json.loads(
            (session.root / "package-manifest.json").read_text(
                encoding="utf-8"
            )
        )
        self.assertEqual(manifest["python_version"], interpreter_version)

        config_path.write_text(
            config_path.read_text(encoding="utf-8").replace(
                "include-system-site-packages = false",
                "include-system-site-packages = true",
            ),
            encoding="utf-8",
        )
        self.backend.create_session("isolated-venv", "python-data")

        self.assertIn(
            "include-system-site-packages = false",
            config_path.read_text(encoding="utf-8"),
        )

    def test_normalizes_optional_and_broken_native_dependencies(self):
        optional = self.runtime.execute(
            ExecutionRequest(
                session_id="dependency-errors",
                language="python",
                code=(
                    "raise ImportError("
                    "\"Missing optional dependency 'tabulate'. Use pip to install tabulate.\""
                    ")\n"
                ),
                timeout_seconds=10,
            )
        )
        native = self.runtime.execute(
            ExecutionRequest(
                session_id="dependency-errors",
                language="python",
                code=(
                    "raise ImportError("
                    "\"cannot import name '_imaging' from 'PIL' (/broken/PIL/__init__.py)\""
                    ")\n"
                ),
                timeout_seconds=10,
            )
        )

        self.assertEqual(optional.error.category, "missing_dependency")
        self.assertEqual(optional.error.suggested_package, "tabulate")
        self.assertEqual(native.error.category, "dependency_error")
        self.assertEqual(native.error.suggested_package, "pillow")

    def test_cleanup_removes_expired_session(self):
        session = self.sessions.get_or_create("expired")
        metadata_path = session.root / "session.json"
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        metadata["expires_at"] = (
            datetime.now(timezone.utc) - timedelta(seconds=1)
        ).isoformat()
        metadata_path.write_text(json.dumps(metadata), encoding="utf-8")

        removed = self.sessions.cleanup_expired()

        self.assertIn(session.root.name, removed)
        self.assertFalse(session.root.exists())

    def test_runtime_settings_are_owned_and_parsed_by_code_module(self):
        settings = CodeRuntimeSettings.from_env(
            {
                "ORCHESTRATOR_CODE_RUNTIME_ENABLED": "false",
                "ORCHESTRATOR_CODE_RUNTIME_TIMEOUT_SECONDS": "45",
                "ORCHESTRATOR_CODE_RUNTIME_PACKAGE_BLOCKLIST": "one,two",
                "ORCHESTRATOR_CODE_RUNTIME_MAX_OUTPUT_FILES": "7",
            }
        )

        self.assertFalse(settings.enabled)
        self.assertEqual(settings.timeout_seconds, 45)
        self.assertEqual(settings.package_blocklist, ("one", "two"))
        self.assertEqual(settings.max_output_files, 7)

    def test_stateful_runtime_accepts_configured_profile_overrides(self):
        runtime = CodeExecutionRuntime(
            backend=self.backend,
            session_manager=self.sessions,
            environment_profile_overrides={
                "python-custom": {
                    "allowed_imports": ["json", "pathlib"],
                    "package_allowlist": {},
                }
            },
        )

        result = runtime.execute(
            ExecutionRequest(
                session_id="custom-profile",
                language="python",
                code="import json\nprint(json.dumps({'ok': True}))",
                profile="python-custom",
                timeout_seconds=10,
            )
        )

        self.assertTrue(result.succeeded)
        self.assertIn('"ok": true', result.stdout)

    def test_network_failure_takes_precedence_over_pip_not_found_footer(self):
        category = _classify_pip_error(
            "Failed to establish a new connection: Name or service not known\n"
            "ERROR: No matching distribution found for matplotlib"
        )

        self.assertEqual(category, "network_failure")


if __name__ == "__main__":
    unittest.main()
