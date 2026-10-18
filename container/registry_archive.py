#!/usr/bin/env python3
"""Download an immutable private ACR image without exposing authentication values."""
import argparse
import hashlib
import io
import json
import os
import re
import tarfile
import tempfile
from pathlib import Path

import requests
from azure.identity import AzureCliCredential


def checked(response, operation):
    if response.status_code != 200:
        raise RuntimeError(f"Private registry {operation} failed (HTTP {response.status_code}); details withheld.")
    return response


def export_image(image: str, tenant: str, destination: Path):
    match = re.fullmatch(
        r"([a-z0-9]+\.azurecr\.io)/([a-z0-9_./-]+)@(sha256:[a-f0-9]{64})", image,
    )
    if not match:
        raise ValueError("Use an immutable private Azure registry image digest.")
    registry, repository, image_digest = match.groups()
    base = "https://" + registry
    credential = AzureCliCredential(tenant_id=tenant)
    aad_token = credential.get_token("https://management.azure.com/.default").token
    with requests.Session() as client:
        exchanged = checked(client.post(
            base + "/oauth2/exchange",
            data={"grant_type": "access_token", "service": registry, "tenant": tenant, "access_token": aad_token},
            timeout=60,
        ), "authentication").json()
        token = checked(client.post(
            base + "/oauth2/token",
            data={
                "grant_type": "refresh_token", "service": registry,
                "scope": f"repository:{repository}:pull", "refresh_token": exchanged["refresh_token"],
            }, timeout=60,
        ), "repository authorization").json()["access_token"]
        headers = {
            "Authorization": "Bearer " + token,
            "Accept": "application/vnd.docker.distribution.manifest.v2+json, application/vnd.oci.image.manifest.v1+json",
        }
        response = checked(client.get(
            f"{base}/v2/{repository}/manifests/{image_digest}", headers=headers, timeout=60,
        ), "manifest retrieval")
        if "sha256:" + hashlib.sha256(response.content).hexdigest() != image_digest:
            raise RuntimeError("Registry manifest bytes do not match the approved digest.")
        manifest = response.json()
        descriptors = [manifest["config"], *manifest["layers"]]
        if not manifest["layers"]:
            raise RuntimeError("Registry manifest has no image layers.")
        with tempfile.TemporaryDirectory(prefix="inksight-registry-audit-") as directory:
            downloaded = {}
            for index, descriptor in enumerate(descriptors):
                expected = descriptor["digest"]
                if not re.fullmatch(r"sha256:[a-f0-9]{64}", expected):
                    raise RuntimeError("Unexpected registry blob digest.")
                blob = checked(client.get(
                    f"{base}/v2/{repository}/blobs/{expected}", headers=headers,
                    timeout=120, stream=True,
                ), "blob retrieval")
                path = Path(directory) / str(index)
                fingerprint = hashlib.sha256()
                with blob, path.open("wb") as stream:
                    for chunk in blob.iter_content(1024 * 1024):
                        stream.write(chunk)
                        fingerprint.update(chunk)
                if "sha256:" + fingerprint.hexdigest() != expected:
                    raise RuntimeError("Registry blob bytes do not match the manifest digest.")
                downloaded[expected] = path
                print(f"Verified image blob {index + 1}/{len(descriptors)}", flush=True)
            config_name = "blobs/sha256/" + manifest["config"]["digest"].split(":")[1]
            layer_names = ["blobs/sha256/" + item["digest"].split(":")[1] for item in manifest["layers"]]
            with destination.open("wb") as output:
                os.chmod(destination, 0o600)
                with tarfile.open(fileobj=output, mode="w") as archive:
                    for digest, path in downloaded.items():
                        archive.add(path, arcname="blobs/sha256/" + digest.split(":")[1], recursive=False)
                    data = json.dumps([{"Config": config_name, "Layers": layer_names}]).encode()
                    member = tarfile.TarInfo("manifest.json")
                    member.size = len(data)
                    archive.addfile(member, io.BytesIO(data))
    return {"image_digest": image_digest, "layers": len(layer_names), "credentials_exported": False}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True)
    parser.add_argument("--tenant", required=True)
    parser.add_argument("--destination", required=True, type=Path)
    args = parser.parse_args()
    try:
        report = export_image(args.image, args.tenant, args.destination)
    except requests.RequestException:
        raise RuntimeError("Private registry network request failed; authentication/redirect details withheld.") from None
    print(json.dumps(report))


if __name__ == "__main__":
    main()
