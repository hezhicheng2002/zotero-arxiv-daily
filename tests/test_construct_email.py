"""Tests for zotero_arxiv_daily.construct_email: render_email, score formatting, get_stars, get_block_html."""

from zotero_arxiv_daily.construct_email import (
    _format_relative_relevance_scores,
    get_block_html,
    get_empty_html,
    get_stars,
    render_email,
)
from tests.canned_responses import make_sample_paper


def test_render_email_with_papers():
    papers = [make_sample_paper(score=7.5, tldr="A great paper.", affiliations=["MIT"])]
    html = render_email(papers)
    assert "Sample Paper Title" in html
    assert "A great paper." in html
    assert "MIT" in html


def test_render_email_empty_list():
    html = render_email([])
    assert "No Papers Today" in html


def test_render_email_author_truncation():
    authors = [f"Author {i}" for i in range(10)]
    paper = make_sample_paper(authors=authors, score=7.0, tldr="ok")
    html = render_email([paper])
    assert "Author 0" in html
    assert "Author 1" in html
    assert "Author 2" in html
    assert "..." in html
    assert "Author 8" in html
    assert "Author 9" in html
    # Middle authors should be truncated
    assert "Author 5" not in html


def test_render_email_affiliation_truncation():
    affiliations = [f"Uni {i}" for i in range(8)]
    paper = make_sample_paper(affiliations=affiliations, score=7.0, tldr="ok")
    html = render_email([paper])
    assert "Uni 0" in html
    assert "Uni 4" in html
    assert "..." in html
    assert "Uni 7" not in html


def test_render_email_no_affiliations():
    paper = make_sample_paper(affiliations=None, score=7.0, tldr="ok")
    html = render_email([paper])
    assert "Unknown Affiliation" in html


def test_get_stars_low_score():
    assert get_stars(5.0) == ""
    assert get_stars(6.0) == ""


def test_get_stars_high_score():
    stars = get_stars(8.0)
    assert stars.count("full-star") == 5


def test_get_stars_mid_score():
    stars = get_stars(7.0)
    assert "star" in stars
    assert stars.count("full-star") + stars.count("half-star") > 0


def test_get_block_html_contains_all_fields():
    html = get_block_html("Title", "Auth", "3.5", "Summary", "Abstract", "http://pdf.url", "MIT")
    assert "Title" in html
    assert "Auth" in html
    assert "3.5" in html
    assert "Summary" in html
    assert "http://pdf.url" in html
    assert "MIT" in html


def test_render_email_uses_abstract_when_tldr_is_disabled():
    paper = make_sample_paper(abstract="Test Abstract", tldr="Test TLDR")
    html = render_email([paper], show_tldr=False)
    assert "Abstract:" in html
    assert "Test Abstract" in html
    assert "Test TLDR" not in html


def test_relative_relevance_scores_are_rank_normalized():
    papers = [
        make_sample_paper(score=0.03),
        make_sample_paper(score=0.11),
        make_sample_paper(score=0.07),
    ]
    scores = _format_relative_relevance_scores(papers)
    assert scores == ["6.0/10", "9.5/10", "7.8/10"]


def test_render_email_uses_relative_relevance_label():
    papers = [
        make_sample_paper(title="Top", score=0.20),
        make_sample_paper(title="Lower", score=0.05),
    ]
    html = render_email(papers)
    assert "Relative Relevance:" in html
    assert "9.5/10" in html
    assert "6.0/10" in html


def test_render_email_escapes_abstract_with_braces():
    paper = make_sample_paper(abstract="Uses {medical perception bottleneck} & <xml> markers.", tldr=None)
    html = render_email([paper], show_tldr=False)
    assert "medical perception bottleneck" in html
    assert "{medical perception bottleneck}" in html
    assert "&lt;xml&gt;" in html


def test_get_empty_html():
    html = get_empty_html()
    assert "No Papers Today" in html
