"""Release navigation must not silently publish inaccessible or preview targets."""
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "site"))
import release_links


def settings(tmp_path, **overrides):
    c = {"website_url": "", "technical_repository_url": "", "technical_evidence_url": "", "source_mode": "public-repository"}
    c.update(overrides)
    p = tmp_path / "links.json"
    p.write_text(json.dumps(c))
    return p


def test_release_missing_urls_fails_before_network(tmp_path):
    assert release_links.load(settings(tmp_path))["preview"]
    with pytest.raises(ValueError, match="release URLs missing"):
        release_links.load(settings(tmp_path), release=True)


@pytest.mark.parametrize("url", ["http://example.org", "https://127.0.0.1/", "https://localhost/", "https://example.com/", "https://user:pass@public.org/", "https://public.org/?token=x"])
def test_preview_does_not_accept_fake_or_credential_urls(tmp_path, url):
    with pytest.raises(ValueError, match="actual public HTTPS"):
        release_links.load(settings(tmp_path, website_url=url))


def test_release_evidence_mode_needs_no_repository(tmp_path, monkeypatch):
    checked = []
    monkeypatch.setattr(release_links, "check_access", lambda u: checked.append(u) or {"url": u, "status": 200})
    p = settings(tmp_path, website_url="https://case.org/", technical_evidence_url="https://case.org/evidence/", source_mode="public-evidence")
    c = release_links.load(p, release=True)
    assert not c["preview"] and checked == [c["website_url"], c["technical_evidence_url"]]


def test_release_propagates_inaccessible_target(tmp_path, monkeypatch):
    def unavailable(u):
        raise OSError("HTTP 404")
    monkeypatch.setattr(release_links, "check_access", unavailable)
    p = settings(tmp_path, website_url="https://case.org/", technical_repository_url="https://github.com/owner/repo")
    with pytest.raises(OSError, match="404"):
        release_links.load(p, release=True)


@pytest.mark.parametrize("filename,pages", [("executive-summary-one-page.pdf", 1)])
def test_canonical_recipient_preview_has_pending_notice_and_only_public_uris(filename, pages):
    import render
    root = Path(__file__).resolve().parents[1]
    render.verify_pdf(root / "reports" / filename, pages, release_links.load())


@pytest.mark.parametrize("uri", ["http://127.0.0.1:8765/", "http://localhost/", "file:///private/preview.html", "https://192.168.1.1/"])
def test_pdf_renderer_rejects_local_links_even_in_preview(tmp_path, monkeypatch, uri):
    import render
    import pypdf
    class Annotation:
        def get_object(self):
            return {'/A': {'/URI': uri}}
    class Page(dict):
        mediabox = type('Box', (), {'width': 595, 'height': 842})()
        def extract_text(self):
            return 'Website not yet deployed.'
    reader = type('Reader', (), {'pages': [Page({'/Annots': [Annotation()]})]})()
    monkeypatch.setattr(pypdf, 'PdfReader', lambda path: reader)
    with pytest.raises(ValueError, match='non-public URI'):
        render.verify_pdf(tmp_path / 'preview.pdf', 1, {**release_links.load(), 'website_url': '',
                          'technical_repository_url': '', 'technical_evidence_url': ''})


def test_private_source_mode_never_claims_anonymous_access(tmp_path, monkeypatch):
    checked = []
    monkeypatch.setattr(release_links, 'check_access', lambda u: checked.append(u) or {'status': 200})
    p = settings(tmp_path, website_url='https://case.org/', technical_repository_url='https://github.com/owner/repo', source_mode='private-repository')
    c = release_links.load(p, release=True)
    assert not c['preview'] and c['source_anonymous'] is False
    assert checked == ['https://case.org/']
