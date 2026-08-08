from __future__ import annotations

import os
import ssl
from pathlib import Path

import certifi


def ssl_diagnostics() -> dict[str, object]:
    cert_dir = os.environ.get("SSL_CERT_DIR")
    cert_file = os.environ.get("SSL_CERT_FILE")
    default_paths = ssl.get_default_verify_paths()
    cert_dir_path = Path(cert_dir) if cert_dir else None

    return {
        "ssl_verification": "enabled",
        "ssl_cert_dir_set": cert_dir is not None,
        "ssl_cert_dir_exists": bool(cert_dir_path and cert_dir_path.is_dir()),
        "ssl_cert_dir_has_certificate_files": bool(
            cert_dir_path
            and cert_dir_path.is_dir()
            and any(
                path.is_file() and path.suffix.lower() in {".pem", ".crt", ".cer"}
                for path in cert_dir_path.iterdir()
            )
        ),
        "ssl_cert_file_set": cert_file is not None,
        "ssl_cert_file_exists": bool(cert_file and Path(cert_file).is_file()),
        "httpx_default_ca_source": "certifi",
        "httpx_default_ca_path_exists": Path(certifi.where()).is_file(),
        "system_default_cafile_exists": bool(
            default_paths.cafile and Path(default_paths.cafile).is_file()
        ),
        "system_default_capath_exists": bool(
            default_paths.capath and Path(default_paths.capath).is_dir()
        ),
    }
