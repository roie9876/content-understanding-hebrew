import io
import json
import os
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch

import fitz
from PIL import Image

import app as web
from public_demo import DemoError, PublicDemo


class DemoBudgetTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.environment = patch.dict(os.environ, {
            "DEMO_MAX_JOBS_PER_HOUR": "10", "DEMO_MAX_JOBS_TOTAL": "25",
            "DEMO_EXPIRES_AT": "",
        })
        self.environment.start()
        self.demo = PublicDemo(self.root)

    def tearDown(self):
        self.environment.stop()
        self.temporary.cleanup()

    def test_hourly_limit_allows_ten_then_rejects_eleven(self):
        now = self.demo.state["created_at"] + 1
        with patch("public_demo.time.time", return_value=now):
            for _ in range(10):
                self.demo.admit("starts")
            with self.assertRaisesRegex(DemoError, "hourly"):
                self.demo.admit("starts")
        with patch("public_demo.time.time", return_value=now + 3601):
            self.demo.admit("starts")

    def test_total_allowance_survives_restart_and_allows_exactly_twenty_five(self):
        base = self.demo.state["created_at"]
        for index in range(25):
            with patch("public_demo.time.time", return_value=base + index * 3601):
                self.demo.admit("starts")
        restored = PublicDemo(self.root)
        self.assertEqual(self.demo.secret_key, restored.secret_key)
        with self.assertRaisesRegex(DemoError, "allowance"):
            restored.admit("starts")

    def test_no_default_expiration(self):
        self.assertIsNone(self.demo.public_config()["expires_at"])
        with patch("public_demo.time.time", return_value=self.demo.state["created_at"] + 30 * 24 * 3600):
            self.demo.admit("starts")

    def test_expiration_rejects_at_exact_deadline(self):
        with patch.dict(os.environ, {"DEMO_EXPIRES_AT": "2030-01-01T00:00:00Z"}):
            demo = PublicDemo(self.root)
        self.assertEqual("2030-01-01T00:00:00+00:00", demo.public_config()["expires_at"])
        with patch("public_demo.time.time", return_value=demo.expires_at):
            with self.assertRaises(DemoError) as error:
                demo.admit("starts")
        self.assertEqual(410, error.exception.status)

    def test_missing_signing_key_does_not_silently_reset_ownership(self):
        (self.root / ".private/session.key").unlink()
        with self.assertRaisesRegex(RuntimeError, "signing key is missing"):
            PublicDemo(self.root)

    def test_corrupt_usage_does_not_reset_limits(self):
        (self.root / ".private/usage.json").write_text('{"starts": []}')
        with self.assertRaisesRegex(RuntimeError, "usage state"):
            PublicDemo(self.root)

    def test_invalid_limit_fails_explicitly(self):
        with patch.dict(os.environ, {"DEMO_MAX_PAGES": "0"}):
            with self.assertRaisesRegex(ValueError, "DEMO_MAX_PAGES"):
                PublicDemo(self.root)


class PublicDemoRouteTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        data = self.root / "data"
        self.old_config = dict(web.app.config)
        self.patches = [
            patch.multiple(
                web, PUBLIC_DEMO=True, DATA_DIR=data, UPLOAD_DIR=data / "uploads",
                OUTPUT_DIR=data / "outputs", RESULTS_DIR=data / "results",
                _jobs={}, _uploads={}, _processes={}, _demo=None, _demo_starting=False,
            ),
            patch.dict(os.environ, {"DEMO_EXPIRES_AT": ""}),
        ]
        for item in self.patches:
            item.start()
        web.initialize_runtime()
        self.a = web.app.test_client()
        self.b = web.app.test_client()

    def tearDown(self):
        web.app.config.clear()
        web.app.config.update(self.old_config)
        for item in reversed(self.patches):
            item.stop()
        self.temporary.cleanup()

    def upload(self, client=None):
        stream = io.BytesIO()
        Image.new("RGB", (100, 60), "white").save(stream, format="PNG")
        response = (client or self.a).post(
            "/api/uploads", base_url="https://localhost",
            data={"document": (io.BytesIO(stream.getvalue()), "synthetic.png")},
        )
        self.assertEqual(201, response.status_code, response.get_data(as_text=True))
        return response.get_json()

    def job(self):
        upload = self.upload()
        with patch.object(web.threading, "Thread"):
            response = self.a.post(
                "/api/jobs", base_url="https://localhost",
                data={"upload_id": upload["id"], "pages": "1"},
            )
        self.assertEqual(202, response.status_code, response.get_data(as_text=True))
        return upload, response.get_json()

    def complete(self, job):
        saved = web._jobs[job["id"]]
        saved.update(status="complete", stage="complete", completed_at=time.time())
        output = Path(saved["output_dir"])
        (output / "handwriting_result.json").write_text('{"regions": []}')
        (output / "merged.md").write_text("Synthetic demo result")
        web._persist_job(job["id"])

    def test_public_site_needs_no_login_and_cookie_is_secure(self):
        response = self.a.get("/", base_url="https://localhost")
        self.assertEqual(200, response.status_code)
        cookie = response.headers["Set-Cookie"]
        for expected in ("__Host-inksight-demo", "Secure", "HttpOnly", "SameSite=Lax"):
            self.assertIn(expected, cookie)
        response.close()
        config = self.a.get("/api/demo-config", base_url="https://localhost").get_json()
        self.assertTrue(config["public_demo"])
        self.assertNotIn("secret", json.dumps(config).lower())
        self.assertEqual(2, config["max_pages"])

    def test_health_does_not_allocate_a_browser_session(self):
        response = self.a.get("/healthz")
        self.assertEqual({"status": "ok"}, response.get_json())
        self.assertNotIn("Set-Cookie", response.headers)

    def test_runtime_state_cannot_be_served_as_results_or_static_assets(self):
        for data, results in (
            (self.root / "shared/private", self.root / "shared"),
            (self.root / "ui/runtime", self.root / "results"),
        ):
            with self.subTest(data=data), patch.multiple(
                web, ROOT=self.root, DATA_DIR=data, RESULTS_DIR=results,
                UPLOAD_DIR=data / "uploads",
            ):
                with self.assertRaisesRegex(RuntimeError, "served"):
                    web.initialize_runtime()

    def test_other_browser_cannot_preview_or_reuse_an_upload(self):
        upload = self.upload()
        self.assertEqual(404, self.b.get(
            upload["pages"][0]["preview_url"], base_url="https://localhost",
        ).status_code)
        self.assertEqual(404, self.b.post(
            "/api/jobs", base_url="https://localhost", data={"upload_id": upload["id"]},
        ).status_code)

    def test_all_job_surfaces_enforce_browser_ownership(self):
        _upload, job = self.job()
        self.complete(job)
        prefix = f"/api/jobs/{job['id']}"
        for path in (
            prefix, prefix + "/pages/1/diagnostics", prefix + "/pages/1/merged-document",
            prefix + "/pages/1/merged.png", prefix + "/pages/1/golden-comparison",
            prefix + "/files/run.json",
        ):
            with self.subTest(path=path):
                self.assertEqual(404, self.b.get(path, base_url="https://localhost").status_code)
        self.assertEqual(404, self.b.post(prefix + "/cancel", base_url="https://localhost").status_code)
        self.assertEqual([], self.b.get("/api/runs", base_url="https://localhost").get_json())
        self.assertEqual(1, len(self.a.get("/api/runs", base_url="https://localhost").get_json()))
        self.assertEqual(200, self.a.get(prefix, base_url="https://localhost").status_code)

    def test_ownership_uploads_history_and_key_survive_restart(self):
        upload, job = self.job()
        self.complete(job)
        secret = web._demo.secret_key
        web._uploads.clear()
        web._jobs.clear()
        web.initialize_runtime()
        self.assertEqual(secret, web._demo.secret_key)
        self.assertIn(upload["id"], web._uploads)
        self.assertEqual(200, self.a.get(
            f"/api/jobs/{job['id']}", base_url="https://localhost",
        ).status_code)
        self.assertEqual(1, len(self.a.get("/api/runs", base_url="https://localhost").get_json()))
        self.assertEqual([], self.b.get("/api/runs", base_url="https://localhost").get_json())

    def test_interrupted_job_is_not_restored_as_success(self):
        _upload, job = self.job()
        web._jobs.clear()
        web.initialize_runtime()
        restored = self.a.get(f"/api/jobs/{job['id']}", base_url="https://localhost").get_json()
        self.assertEqual("failed", restored["status"])
        self.assertIn("restart", restored["message"])

    def test_global_concurrency_and_failed_request_reservation_release(self):
        first = self.a.post("/api/jobs", base_url="https://localhost")
        self.assertEqual(400, first.status_code)
        self.assertFalse(web._demo_starting)
        upload, _job = self.job()
        blocked = self.a.post(
            "/api/jobs", base_url="https://localhost", data={"upload_id": upload["id"]},
        )
        self.assertEqual(429, blocked.status_code)

    def test_model_and_strategy_tampering_is_rejected(self):
        upload = self.upload()
        for fields in ({"model_profile": "gpt-5.4"}, {"ocr_strategy": "isolated-multiview"}):
            response = self.a.post(
                "/api/jobs", base_url="https://localhost",
                data={"upload_id": upload["id"], **fields},
            )
            self.assertEqual(400, response.status_code)

    def test_cross_origin_post_is_rejected(self):
        response = self.a.post(
            "/api/uploads", base_url="https://localhost",
            headers={"Origin": "https://untrusted.example"},
        )
        self.assertEqual(403, response.status_code)

    def test_two_pages_allowed_three_pages_rejected(self):
        for count, expected in ((2, 201), (3, 400)):
            with fitz.open() as pdf:
                for _ in range(count):
                    pdf.new_page(width=72, height=72)
                raw = pdf.tobytes()
            response = self.a.post(
                "/api/uploads", base_url="https://localhost",
                data={"document": (io.BytesIO(raw), "synthetic.pdf")},
            )
            self.assertEqual(expected, response.status_code)
            response.close()
            response.request.input_stream.close()

    def test_exact_twenty_mib_file_limit(self):
        limit = 20 * 1024 * 1024
        with patch.object(web, "_page_count", return_value=1), \
             patch.object(web.fitz, "open") as pdf:
            pdf.return_value.__enter__.return_value = []
            for size, expected in ((limit, 201), (limit + 1, 413)):
                response = self.a.post(
                    "/api/uploads", base_url="https://localhost",
                    data={"document": (io.BytesIO(b"x" * size), "synthetic.pdf")},
                )
                self.assertEqual(expected, response.status_code)
                response.close()
                response.request.environ["wsgi.input"].close()

    def test_private_state_is_not_downloadable_and_html_is_sandboxed(self):
        _upload, job = self.job()
        self.complete(job)
        prefix = f"/api/jobs/{job['id']}"
        for path in ("/.private/session.key", prefix + "/files/../../.private/session.key"):
            self.assertEqual(404, self.a.get(path, base_url="https://localhost").status_code)
        response = self.a.get(prefix + "/pages/1/merged-document", base_url="https://localhost")
        self.assertEqual(200, response.status_code)
        self.assertIn("sandbox", response.headers["Content-Security-Policy"])
        self.assertFalse(self.a.get(
            prefix + "/pages/1/golden-comparison", base_url="https://localhost",
        ).get_json()["available"])

    def test_expired_demo_keeps_existing_downloads_but_denies_new_work(self):
        _upload, job = self.job()
        self.complete(job)
        web._demo.expires_at = time.time() - 1
        self.assertEqual(410, self.a.post(
            "/api/jobs", base_url="https://localhost",
        ).status_code)
        response = self.a.get(
            f"/api/jobs/{job['id']}/files/run.json", base_url="https://localhost",
        )
        self.assertEqual(200, response.status_code)
        response.close()


if __name__ == "__main__":
    unittest.main()
