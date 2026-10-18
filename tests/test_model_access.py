import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from azure.core.exceptions import ClientAuthenticationError
from PIL import Image

import handwriting_pipeline as pipeline


class ModelAccessTests(unittest.TestCase):
    def test_credential_is_bound_to_the_project_tenant(self):
        with patch.dict(os.environ, {"AZURE_TENANT_ID": "project-tenant"}), \
             patch.object(pipeline, "AZURE_AUTH_MODE", "azure-cli"), \
             patch.object(pipeline, "AzureCliCredential") as credential:
            self.assertIs(credential.return_value, pipeline._make_azure_credential())

        credential.assert_called_once_with(tenant_id="project-tenant")

    def test_system_assigned_identity_does_not_use_cli_credentials(self):
        with patch.dict(os.environ, {"AZURE_CLIENT_ID": ""}), \
             patch.object(pipeline, "AZURE_AUTH_MODE", "managed-identity"), \
             patch.object(pipeline, "ManagedIdentityCredential") as identity, \
             patch.object(pipeline, "AzureCliCredential") as cli:
            self.assertIs(identity.return_value, pipeline._make_azure_credential())

        identity.assert_called_once_with(client_id=None)
        cli.assert_not_called()

    def test_user_assigned_identity_uses_the_configured_client_id(self):
        with patch.dict(os.environ, {"AZURE_CLIENT_ID": "identity-client-id"}), \
             patch.object(pipeline, "AZURE_AUTH_MODE", "managed-identity"), \
             patch.object(pipeline, "ManagedIdentityCredential") as identity:
            self.assertIs(identity.return_value, pipeline._make_azure_credential())

        identity.assert_called_once_with(client_id="identity-client-id")

    def test_invalid_authentication_mode_is_not_silently_ignored(self):
        with patch.object(pipeline, "AZURE_AUTH_MODE", "invalid"):
            with self.assertRaisesRegex(ValueError, "AZURE_AUTH_MODE"):
                pipeline._make_azure_credential()

    def test_managed_identity_failure_is_actionable_and_not_cached(self):
        credential = Mock()
        credential.get_token.side_effect = ClientAuthenticationError("Identity unavailable")
        with patch.object(pipeline, "AZURE_AUTH_MODE", "managed-identity"), \
             patch.object(pipeline, "_credential", credential), \
             patch.object(pipeline, "_token_cache", {}):
            with self.assertRaisesRegex(pipeline.ModelAccessError, "identity is assigned"):
                pipeline._bearer()
            self.assertEqual({}, pipeline._token_cache)

    def test_mfa_failure_is_actionable_and_not_cached_as_a_valid_token(self):
        credential = Mock()
        credential.get_token.side_effect = ClientAuthenticationError("MFA required")
        with patch.object(pipeline, "_credential", credential), \
             patch.object(pipeline, "AZURE_AUTH_MODE", "azure-cli"), \
             patch.object(pipeline, "_token_cache", {}):
            with self.assertRaisesRegex(pipeline.ModelAccessError, "AZURE_CONFIG_DIR"):
                pipeline._bearer()
            self.assertEqual({}, pipeline._token_cache)

    def test_tenant_mismatch_and_forbidden_access_are_not_retryable_ocr_errors(self):
        for status, message in (
            (400, "Tenant provided in token does not match resource token"),
            (401, "Authentication required"),
            (403, "Access denied"),
        ):
            with self.subTest(status=status), patch.object(pipeline, "_bearer", return_value="test-token"), \
                 patch.object(pipeline, "_http_session") as session:
                session.return_value.post.return_value = Mock(status_code=status, text=message)
                with self.assertRaises(pipeline.ModelAccessError):
                    pipeline._responses_text([{"type": "input_text", "text": "read"}], retries=0)
                self.assertEqual(1, session.return_value.post.call_count)

    def test_detection_does_not_turn_auth_failure_into_empty_or_color_only_results(self):
        with patch.object(
            pipeline, "_responses_call", side_effect=pipeline.ModelAccessError("wrong tenant"),
        ) as model, patch.object(pipeline, "detect_colored_pen_regions") as colors:
            with self.assertRaises(pipeline.ModelAccessError):
                pipeline.detect_handwriting_regions(Image.new("RGB", (200, 200), "white"))

        self.assertEqual(1, model.call_count)
        colors.assert_not_called()

    def test_ocr_auth_failure_stops_the_job_instead_of_writing_a_success_result(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "page.png"
            output = root / "output"
            Image.new("RGB", (100, 60), "white").save(source)
            with patch.object(pipeline, "OCR_STRATEGY", "standard"), \
                 patch.object(pipeline, "VOTE_ENABLED", False), \
                 patch.object(
                     pipeline, "ocr_image_set", side_effect=pipeline.ModelAccessError("wrong tenant"),
                 ):
                with self.assertRaises(pipeline.ModelAccessError):
                    pipeline.run_pipeline(
                        str(source), page=1, dpi=400, crops=[(0, 0, 100, 60)],
                        output_dir=str(output), chunk_width=900, overlap=150, ocr_mode="full",
                    )
            self.assertFalse((output / "handwriting_result.json").exists())

    def test_content_understanding_auth_failure_is_not_silently_ignored(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "page.png"
            output = root / "output"
            Image.new("RGB", (100, 60), "white").save(source)
            with patch.object(pipeline, "OCR_STRATEGY", "standard"), \
                 patch.object(pipeline, "VOTE_ENABLED", False), \
                 patch.object(pipeline, "ocr_image_set", return_value={
                     "classification": "handwritten_text", "full_transcription": "readable",
                 }), \
                 patch.object(pipeline, "visual_reconcile_ocr_results", return_value={
                     "classification": "handwritten_text", "final_transcription": "readable",
                     "words": [], "notes": [],
                 }), \
                 patch.object(pipeline, "should_contextual_recheck", return_value=False), \
                 patch.object(
                     pipeline, "cu_ocr_markdown", side_effect=pipeline.ModelAccessError("wrong CU tenant"),
                 ):
                with self.assertRaises(pipeline.ModelAccessError):
                    pipeline.run_pipeline(
                        str(source), page=1, dpi=400, crops=[(0, 0, 100, 60)],
                        output_dir=str(output), chunk_width=900, overlap=150, ocr_mode="full",
                        merge_md=True,
                    )
            self.assertFalse((output / "handwriting_result.json").exists())

    def test_exhausted_recognition_does_not_produce_an_empty_success(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "page.png"
            output = root / "output"
            Image.new("RGB", (100, 60), "white").save(source)
            with patch.object(pipeline, "OCR_STRATEGY", "standard"), \
                 patch.object(pipeline, "VOTE_ENABLED", False), \
                 patch.object(pipeline.time, "sleep"), \
                 patch.object(pipeline, "ocr_image_set", side_effect=RuntimeError("service unavailable")), \
                 patch.object(pipeline, "ocr_image", side_effect=RuntimeError("service unavailable")), \
                 patch.object(
                     pipeline, "visual_reconcile_ocr_results", side_effect=RuntimeError("service unavailable"),
                 ), \
                 patch.object(
                     pipeline, "reconcile_ocr_results", side_effect=RuntimeError("service unavailable"),
                 ):
                with self.assertRaisesRegex(RuntimeError, "every detected region"):
                    pipeline.run_pipeline(
                        str(source), page=1, dpi=400, crops=[(0, 0, 100, 60)],
                        output_dir=str(output), chunk_width=900, overlap=150, ocr_mode="full",
                    )
            self.assertFalse((output / "handwriting_result.json").exists())


if __name__ == "__main__":
    unittest.main()
