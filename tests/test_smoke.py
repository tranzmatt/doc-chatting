"""Fast offline tests for doc-chatting. Run with python -m unittest discover -s tests -v."""
import contextlib
import importlib.util
import io
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import Mock, patch

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("doc_chatting", ROOT / "chatting_with_your_docs.py")
app = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(app)


class CLITests(unittest.TestCase):
    def test_help_runs_without_model_or_llamaindex(self):
        process = subprocess.run(
            [sys.executable, str(ROOT / "chatting_with_your_docs.py"), "--help"],
            cwd=ROOT, capture_output=True, text=True, timeout=20,
        )
        self.assertEqual(process.returncode, 0, process.stderr)
        for flag in ("--documents", "--model-path", "--question", "--reuse-index", "--gradio"):
            self.assertIn(flag, process.stdout)

    def test_default_model_url_is_gguf(self):
        self.assertTrue(app.DEFAULT_MODEL_URL.endswith(".gguf"))


class DownloadTests(unittest.TestCase):
    def test_download_streams_pdf_and_skips_existing_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            response = Mock()
            response.__enter__ = Mock(return_value=response)
            response.__exit__ = Mock(return_value=False)
            response.iter_content.return_value = [b"%PDF-1.7\n", b"contents"]
            with patch.object(app, "SAMPLE_PDFS", {"example.pdf": "https://example.invalid/test.pdf"}), \
                 patch.object(app.requests, "get", return_value=response) as get, \
                 contextlib.redirect_stdout(io.StringIO()):
                app.download_samples(Path(tmp))
                app.download_samples(Path(tmp))
            self.assertEqual((Path(tmp) / "example.pdf").read_bytes(), b"%PDF-1.7\ncontents")
            get.assert_called_once_with("https://example.invalid/test.pdf", stream=True, timeout=120)
            response.raise_for_status.assert_called_once()

    def test_download_fails_on_http_error(self):
        with tempfile.TemporaryDirectory() as tmp:
            response = Mock()
            response.__enter__ = Mock(return_value=response)
            response.__exit__ = Mock(return_value=False)
            response.raise_for_status.side_effect = app.requests.HTTPError("not found")
            with patch.object(app, "SAMPLE_PDFS", {"bad.pdf": "https://example.invalid/bad"}), \
                 patch.object(app.requests, "get", return_value=response), \
                 contextlib.redirect_stdout(io.StringIO()):
                with self.assertRaises(app.requests.HTTPError):
                    app.download_samples(Path(tmp))
            self.assertFalse((Path(tmp) / "bad.pdf").exists())


if __name__ == "__main__":
    unittest.main()
