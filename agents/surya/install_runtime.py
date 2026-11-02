"""Install the pinned official native Linux Codex binary into the private root.

Run in Ubuntu WSL: python3 agents/surya/install_runtime.py
Downloads package data only; never executes npm lifecycle scripts.
"""
import base64
import hashlib
import io
import json
import os
from pathlib import Path
import platform
import tarfile
import urllib.request

VERSION = "0.166.0"


def main():
    if platform.system() != "Linux" or platform.machine() != "x86_64":
        raise SystemExit("This installer targets Ubuntu WSL2 x86_64.")
    root = Path(os.environ.get("DSC180_PRIVATE_WIN", Path.home() / "dsc180-private-win"))
    runtime = root / "runtime"
    metadata_url = f"https://registry.npmjs.org/@openai%2Fcodex/{VERSION}-linux-x64"
    with urllib.request.urlopen(metadata_url, timeout=60) as response:
        metadata = json.load(response)
    with urllib.request.urlopen(metadata["dist"]["tarball"], timeout=120) as response:
        data = response.read()
    integrity = "sha512-" + base64.b64encode(hashlib.sha512(data).digest()).decode()
    if integrity != metadata["dist"]["integrity"]:
        raise SystemExit("Package integrity mismatch; refusing installation")
    with tarfile.open(fileobj=io.BytesIO(data), mode="r:gz") as archive:
        members = [m for m in archive.getmembers() if m.isfile() and
                   m.name.endswith(("/bin/codex", "/bin/codex-code-mode-host", "/codex-resources/bwrap",
                                    "/codex-path/rg", "/codex-package.json"))]
        native = [m for m in members if m.name.endswith("/bin/codex")]
        if len(native) != 1:
            raise SystemExit("Unexpected package layout; refusing installation")
        runtime.mkdir(parents=True, exist_ok=True)
        for member in members:
            relative = Path(member.name).relative_to("package")
            if ".." in relative.parts:
                raise SystemExit("Unexpected archive path")
            target = runtime / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            temporary = target.with_name(target.name + ".new")
            temporary.write_bytes(archive.extractfile(member).read())
            temporary.chmod(0o755)
            temporary.replace(target)
        native_path = runtime / Path(native[0].name).relative_to("package")
        temporary_link = runtime / "codex.new-link"
        temporary_link.symlink_to(native_path)
        temporary_link.replace(runtime / "codex")
    print(f"Installed official Linux Codex {VERSION}: {runtime / 'codex'}")


if __name__ == "__main__":
    main()
