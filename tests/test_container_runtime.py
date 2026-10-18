import unittest
import os
import io
import json
import hashlib
import tempfile
import tarfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from PIL import Image

from container.build_context import FILES, build_context
from container import privacy_audit
import handwriting_pipeline as pipeline


class ContainerFontTests(unittest.TestCase):
    def test_linux_hebrew_font_is_used_when_mac_fonts_are_unavailable(self):
        linux_font = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
        font = Mock()
        with patch.object(pipeline.os.path, "exists", side_effect=lambda path: path == linux_font), \
             patch.object(pipeline.ImageFont, "truetype", return_value=font) as load_font:
            self.assertIs(font, pipeline._hebrew_font(24))

        load_font.assert_called_once_with(linux_font, 24)

    def test_existing_mac_font_keeps_priority(self):
        mac_font = "/System/Library/Fonts/Supplemental/Arial Unicode.ttf"
        with patch.object(pipeline.os.path, "exists", return_value=True), \
             patch.object(pipeline.ImageFont, "truetype") as load_font:
            self.assertIs(load_font.return_value, pipeline._hebrew_font(24))

        load_font.assert_called_once_with(mac_font, 24)


class BuildContextTests(unittest.TestCase):
    def test_context_contains_only_allowlisted_code(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "repository"
            for name in FILES:
                path = root / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text("synthetic source")
            for name in (".env", "sample/private.pdf", "golden/private.json", ".ui_data/token.json", "analysis_results/private.png"):
                path = root / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text("must not be copied")
            destination = root / "clean-context"
            hashes = build_context(root, destination)
            self.assertEqual(set(FILES), set(hashes))
            self.assertEqual(
                set(FILES),
                {path.relative_to(destination).as_posix() for path in destination.rglob("*") if path.is_file()},
            )
            with self.assertRaisesRegex(ValueError, "empty"):
                build_context(root, destination)

    def test_symlink_input_and_repository_destination_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "repository"
            for name in FILES:
                path = root / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text("synthetic source")
            outside = Path(directory) / "private"
            outside.write_text("must not be copied")
            (root / "app.py").unlink()
            (root / "app.py").symlink_to(outside)
            with self.assertRaisesRegex(ValueError, "symlink"):
                build_context(root, root / "context")
            with self.assertRaises(ValueError):
                build_context(root, root)


class DemoRegionLimitTests(unittest.TestCase):
    def test_forty_regions_allowed_forty_one_rejected_without_truncation(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "synthetic.png"
            Image.new("RGB", (100, 60), "white").save(source)
            with patch.dict(os.environ, {"DEMO_MAX_REGIONS": "40"}), \
                 patch.object(pipeline, "_write_json_atomic", side_effect=RuntimeError("passed region gate")) as write:
                for count, message in ((40, "passed region gate"), (41, "40 detected regions")):
                    with self.subTest(count=count):
                        write.reset_mock()
                        with self.assertRaisesRegex(RuntimeError, message):
                            pipeline.run_pipeline(
                                str(source), page=1, dpi=400, crops=[(0, 0, 10, 10)] * count,
                                output_dir=str(Path(directory) / str(count)),
                                chunk_width=900, overlap=150, ocr_mode="full",
                            )
                        if count == 41:
                            write.assert_not_called()


class ImagePrivacyAuditTests(unittest.TestCase):
    def test_context_rejects_extra_data_and_withholds_detected_credentials(self):
        with tempfile.TemporaryDirectory() as directory:
            context = Path(directory)
            for name in FILES:
                path = context / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text("synthetic source")
            self.assertTrue(privacy_audit.audit_context(context, [])["passed"])
            private = context / "private.pdf"
            private.write_bytes(b"synthetic private test document")
            with self.assertRaisesRegex(RuntimeError, "allowlist"):
                privacy_audit.audit_context(context, [])
            private.unlink()
            dummy = b"unit-test-only-credential-not-a-real-secret"
            (context / "app.py").write_bytes(dummy)
            with self.assertRaises(RuntimeError) as error:
                privacy_audit.audit_context(context, [dummy])
            self.assertNotIn(dummy.decode(), str(error.exception))

    @staticmethod
    def _archive(layers):
        outer_data = io.BytesIO()
        with tarfile.open(fileobj=outer_data, mode="w") as outer:
            names = []
            for index, files in enumerate(layers):
                inner_data = io.BytesIO()
                with tarfile.open(fileobj=inner_data, mode="w") as inner:
                    for name, content in files.items():
                        member = tarfile.TarInfo(name)
                        member.size = len(content)
                        inner.addfile(member, io.BytesIO(content))
                name = f"layer-{index}.tar"
                names.append(name)
                content = inner_data.getvalue()
                member = tarfile.TarInfo(name)
                member.size = len(content)
                outer.addfile(member, io.BytesIO(content))
            manifest = json.dumps([{"Layers": names}]).encode()
            member = tarfile.TarInfo("manifest.json")
            member.size = len(manifest)
            outer.addfile(member, io.BytesIO(manifest))
        return outer_data.getvalue()

    def test_deleted_secret_in_an_earlier_layer_still_fails(self):
        archive = self._archive([
            {"app/.env": b"unit-test-only private configuration"},
            {"app/.wh..env": b""},
        ])
        self._assert_rejected_image(archive, "Unexpected application file")

    def test_renamed_private_document_in_any_layer_is_detected_by_hash(self):
        content = b"synthetic private test document, never real data" * 8
        archive = self._archive([{"tmp/renamed.bin": content}])
        self._assert_rejected_image(archive, "local private file", content)

    def test_reviewed_public_material_cannot_exempt_a_local_credential(self):
        content, reviewed = self._public_fixture()
        archive = self._archive([{"usr/lib/fixture.so": content}])
        self._assert_rejected_image(
            archive, "Potential credentials", reviewed=reviewed, credentials=[content],
        )

    def test_reviewed_public_material_requires_exact_file_hash(self):
        content, reviewed = self._public_fixture()
        reviewed["files"][0]["sha256"] = "0" * 64
        archive = self._archive([{"usr/lib/fixture.so": content}])
        self._assert_rejected_image(archive, "differ from the reviewed", reviewed=reviewed)

    def test_reviewed_pem_material_does_not_exempt_other_credential_patterns(self):
        content, reviewed = self._public_fixture()
        content += b"\nghp_" + b"A" * 40
        reviewed["files"][0]["sha256"] = hashlib.sha256(content).hexdigest()
        archive = self._archive([{"usr/lib/fixture.so": content}])
        self._assert_rejected_image(archive, "Potential credentials", reviewed=reviewed)

    @staticmethod
    def _public_fixture():
        content = b"-----BEGIN PRIVATE KEY-----\n" + b"A" * 64 + b"\n-----END PRIVATE KEY-----"
        return content, {
            "image_digest": "sha256:synthetic",
            "files": [{
                "path": "usr/lib/fixture.so", "sha256": hashlib.sha256(content).hexdigest(),
                "pattern_indexes": [0], "public_source": "synthetic public test fixture",
            }],
        }

    def _assert_rejected_image(self, archive, message, private_content=None, reviewed=None, credentials=None):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            if private_content:
                (root / "sample").mkdir()
                (root / "sample/private.pdf").write_bytes(private_content)

            def run(arguments, **_kwargs):
                if arguments[1] == "history":
                    return SimpleNamespace(stdout=b"")
                if arguments[1:3] == ["image", "save"]:
                    Path(arguments[4]).write_bytes(archive)
                    return SimpleNamespace(returncode=0)
                raise AssertionError("Unexpected external command in image-audit test.")

            details = [{
                "Id": "sha256:synthetic", "Architecture": "amd64",
                "Config": {"User": "10001:10001", "Env": []},
            }]
            with patch.object(privacy_audit, "docker_json", return_value=details), \
                 patch.object(privacy_audit.subprocess, "run", side_effect=run):
                with self.assertRaisesRegex(RuntimeError, message):
                    image = "synthetic@" + reviewed["image_digest"] if reviewed else "synthetic:test"
                    privacy_audit.audit_image(
                        image, root, credentials or [], reviewed_public_material=reviewed,
                    )


if __name__ == "__main__":
    unittest.main()
