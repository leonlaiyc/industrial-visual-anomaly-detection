"""Canonical navigation settings; release checks use anonymous HTTPS requests."""
from __future__ import annotations

import json
import ipaddress
from pathlib import Path
from urllib.parse import urlparse
from urllib.request import Request, urlopen

DEFAULT = Path(__file__).with_name("release_links.json")


def public_url(value: str) -> bool:
    u = urlparse(value)
    if u.scheme != "https" or not u.hostname or u.username or u.password or u.query or u.fragment:
        return False
    host = u.hostname.lower()
    if host == "localhost" or host.endswith((".local", ".invalid", ".test", ".example")) or host in ("example.com", "example.org", "example.net"):
        return False
    try:
        return ipaddress.ip_address(host).is_global
    except ValueError:
        return "." in host


def check_access(url: str) -> dict:
    """Fail on login redirects, HTTP errors and invalid/private GitHub repositories."""
    with urlopen(Request(url, headers={"User-Agent": "IVAD-release-link-check"}), timeout=20) as r:
        final = r.geturl()
        if r.status != 200 or not public_url(final) or any(p in urlparse(final).path.lower() for p in ("/login", "/signin", "/sign-in")):
            raise ValueError(f"not anonymously accessible: {url}")
        # Do not download large evidence assets just to validate accessibility.
        r.read(1024)
    u = urlparse(url)
    if u.hostname == "github.com":
        parts = u.path.strip("/").split("/")
        if len(parts) < 2:
            raise ValueError("repository URL must identify a repository")
        api = "https://api.github.com/repos/" + "/".join(parts[:2])
        with urlopen(Request(api, headers={"User-Agent": "IVAD-release-link-check"}), timeout=20) as r:
            repo = json.load(r)
        if repo.get("private") is not False:
            raise ValueError("repository is not public")
    return {"url": url, "status": 200, "final_url": final, "anonymous": True}


def load(path: Path = DEFAULT, release: bool = False) -> dict:
    c = json.loads(Path(path).read_text(encoding="utf-8"))
    if c.get("source_mode") not in ("public-repository", "public-evidence", "private-repository"):
        raise ValueError("source_mode must explicitly describe public-repository, public-evidence or private-repository")
    for k in ("website_url", "technical_repository_url", "technical_evidence_url"):
        if not isinstance(c.get(k), str) or (c[k] and not public_url(c[k])):
            raise ValueError(f"{k} must be an actual public HTTPS URL or empty for preview")
    required = ["website_url"]
    if c["source_mode"] != "private-repository":
        required.append("technical_repository_url" if c["source_mode"] == "public-repository" else "technical_evidence_url")
    elif not c["technical_repository_url"]:
        raise ValueError("private source must identify its repository, with an explicit access-required disclosure")
    if release:
        missing = [k for k in required if not c[k]]
        if missing:
            raise ValueError("release URLs missing: " + ", ".join(missing))
        # Every rendered external navigation target is checked, not just the required ones.
        c["access_checks"] = [check_access(c[k]) for k in ("website_url", "technical_repository_url", "technical_evidence_url")
                              if c[k] and not (k == "technical_repository_url" and c["source_mode"] == "private-repository")]
    c["source_anonymous"] = c["source_mode"] == "public-repository" and release
    c["preview"] = not release
    c["d5_decision_date"] = "2026-10-11"
    return c
