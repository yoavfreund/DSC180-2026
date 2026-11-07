"""ChatGPT plan OAuth for the local coordinator (Linux/WSL or Docker)."""
import base64
import contextlib
import fcntl
import hashlib
import http.server
import json
import os
from pathlib import Path
import secrets
import sqlite3
import tempfile
import time
import urllib.parse
import urllib.request
import uuid

import jwt

AUTH = "https://auth.openai.com"
RESOURCE = "https://api.openai.com/v1"
SCOPES = "openid profile email offline_access resource.invoke chatgpt.tokens.use.direct"
DIR = Path(os.environ.get("BOOSTING_AUTH_DIR", str(Path.home() / ".config" / "face-boosting")))


def private_dir():
    DIR.mkdir(mode=0o700, parents=True, exist_ok=True)
    os.chmod(DIR, 0o700)


def save(name, data):
    private_dir()
    fd, temp = tempfile.mkstemp(dir=DIR, prefix=".tmp-")
    try:
        with os.fdopen(fd, "w") as out:
            json.dump(data, out)
            out.flush()
            os.fsync(out.fileno())
        os.replace(temp, DIR / name)
    finally:
        if os.path.exists(temp):
            os.unlink(temp)


def load(name):
    return json.loads((DIR / name).read_text())


@contextlib.contextmanager
def lock():
    private_dir()
    fd = os.open(DIR / "auth.lock", os.O_CREAT | os.O_RDWR, 0o600)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX)
        yield
    finally:
        fcntl.flock(fd, fcntl.LOCK_UN)
        os.close(fd)


def post_token(fields):
    req = urllib.request.Request(AUTH + "/api/accounts/oauth/token",
        data=urllib.parse.urlencode(fields).encode(),
        headers={"Content-Type": "application/x-www-form-urlencoded"})
    with urllib.request.urlopen(req, timeout=15) as response:
        return json.load(response)


def sign_in():
    private_dir()
    with lock():
        try:
            host = load("host.json")["id"]
        except FileNotFoundError:
            host = "urn:uuid:" + str(uuid.uuid4())
            save("host.json", {"id": host})
        try:
            previous = load("credentials.json")
        except FileNotFoundError:
            previous = None
        state, nonce, verifier = (secrets.token_urlsafe(32) for _ in range(3))
        challenge = base64.urlsafe_b64encode(hashlib.sha256(verifier.encode()).digest()).rstrip(b"=").decode()
        redirect = "http://127.0.0.1:1455/auth/callback"
        result = {}

        class Handler(http.server.BaseHTTPRequestHandler):
            def do_GET(self):
                url = urllib.parse.urlsplit(self.path)
                if url.path != "/auth/callback":
                    self.send_error(404)
                    return
                result.update(urllib.parse.parse_qs(url.query))
                self.send_response(200)
                self.end_headers()
                self.wfile.write(b"Return to your terminal to see the sign-in result.")

            def log_message(self, *args):
                pass  # callback URL contains an authorization code

        bind = os.environ.get("BOOSTING_AUTH_BIND", "127.0.0.1")
        server = http.server.HTTPServer((bind, 1455), Handler)
        server.timeout = 300
        params = {"client_id": previous["client_id"] if previous else "dynamic_agent_client",
                  "ext_agent_host_id": host, "response_type": "code", "redirect_uri": redirect,
                  "scope": SCOPES, "resource": RESOURCE, "state": state, "nonce": nonce,
                  "code_challenge_method": "S256", "code_challenge": challenge}
        if not previous:
            params["agent_name_hint"] = "face-boosting"
        print("Open this URL in your browser to authorize ChatGPT plan use:\n" + AUTH +
              "/api/accounts/authorize?" + urllib.parse.urlencode(params), flush=True)
        try:
            while not result:
                server.handle_request()
                if not result:
                    raise TimeoutError("No authorization callback received")
        finally:
            server.server_close()
        if result.get("state", [None])[0] != state or "error" in result:
            raise ValueError("Authorization declined or state did not match")
        client_id = result.get("client_id", [None])[0] or (previous and previous["client_id"])
        if not client_id or client_id == "dynamic_agent_client" or (previous and client_id != previous["client_id"]):
            raise ValueError("Invalid client registration")
        tokens = post_token({"grant_type": "authorization_code", "client_id": client_id,
                             "code": result["code"][0], "code_verifier": verifier,
                             "redirect_uri": redirect, "resource": RESOURCE})
        if "chatgpt.tokens.use.direct" not in tokens.get("scope", "").split():
            raise ValueError("ChatGPT plan permission was not granted")
        with urllib.request.urlopen(AUTH + "/.well-known/openid-configuration", timeout=15) as response:
            metadata = json.load(response)
        issuer = metadata["issuer"]
        jwks_uri = metadata["jwks_uri"]
        if urllib.parse.urlparse(jwks_uri).scheme != "https":
            raise ValueError("Invalid JWKS endpoint")
        signing_key = jwt.PyJWKClient(jwks_uri).get_signing_key_from_jwt(tokens["id_token"])
        claims = jwt.decode(tokens["id_token"], signing_key.key, algorithms=["RS256"],
                            audience=client_id, issuer=issuer,
                            options={"require": ["exp", "iss", "aud", "sub", "nonce"]})
        if claims["nonce"] != nonce or (previous and previous["subject"] != claims["sub"]):
            raise ValueError("Account identity or nonce did not match")
        record = {**tokens, "client_id": client_id, "subject": claims["sub"],
                  "expires_at": time.time() + int(tokens["expires_in"]), "host_id": host}
        save("credentials.json", record)
        print("Sign-in complete. Credentials are saved in the private auth volume.")


def access_token():
    with lock():
        record = load("credentials.json")
        if time.time() >= record["expires_at"] - 60:
            updated = post_token({"grant_type": "refresh_token", "client_id": record["client_id"],
                                  "refresh_token": record["refresh_token"], "resource": RESOURCE})
            record.update(updated)
            record["expires_at"] = time.time() + int(updated["expires_in"])
            save("credentials.json", record)
        if "chatgpt.tokens.use.direct" not in record.get("scope", "").split():
            raise ValueError("ChatGPT plan permission missing")
        return record["access_token"]


def reserve_request():
    """Limit this installation to 50 attempted inference calls per rolling 5h."""
    private_dir()
    with sqlite3.connect(DIR / "usage.sqlite3", timeout=10) as db:
        db.execute("CREATE TABLE IF NOT EXISTS calls (at REAL NOT NULL)")
        db.execute("BEGIN IMMEDIATE")
        now = time.time()
        db.execute("DELETE FROM calls WHERE at < ?", (now - 5 * 3600,))
        if db.execute("SELECT count(*) FROM calls").fetchone()[0] >= 50:
            raise RuntimeError("Local ChatGPT plan request limit reached")
        db.execute("INSERT INTO calls VALUES (?)", (now,))
    os.chmod(DIR / "usage.sqlite3", 0o600)


if __name__ == "__main__":
    sign_in()
