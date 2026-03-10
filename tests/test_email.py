import pytest
import pickle
from zotero_arxiv_daily.protocol import Paper
from zotero_arxiv_daily.construct_email import render_email
from zotero_arxiv_daily.utils import send_email
@pytest.fixture
def papers() -> list[Paper]:
    paper = Paper(
        source="arxiv",
        title="Test Paper",
        authors=["Test Author","Test Author 2"],
        abstract="Test Abstract",
        url="https://arxiv.org/abs/2512.04296",
        pdf_url="https://arxiv.org/pdf/2512.04296",
        full_text="Test Full Text",
        tldr="Test TLDR",
        affiliations=["Test Affiliation","Test Affiliation 2"],
        score=0.5
    )
    return [paper]*10

def test_render_email(papers:list[Paper]):
    email_content = render_email(papers)
    assert email_content is not None
    assert "Test TLDR" in email_content


def test_render_email_uses_abstract_when_tldr_is_disabled(papers:list[Paper]):
    email_content = render_email(papers, show_tldr=False)
    assert email_content is not None
    assert "Abstract:" in email_content
    assert "Test Abstract" in email_content
    assert "Test TLDR" not in email_content


def test_render_email_escapes_abstract_with_braces(papers:list[Paper]):
    papers[0].abstract = "Uses {medical perception bottleneck} & <xml> markers."
    email_content = render_email(papers, show_tldr=False)
    assert email_content is not None
    assert "medical perception bottleneck" in email_content
    assert "{medical perception bottleneck}" in email_content
    assert "&lt;xml&gt;" in email_content

def test_send_email(config,papers:list[Paper]):
    send_email(config, render_email(papers))
