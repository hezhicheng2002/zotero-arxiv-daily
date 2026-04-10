from .protocol import Paper
import html
import math


framework = """
<!DOCTYPE HTML>
<html>
<head>
  <style>
    .star-wrapper {
      font-size: 1.3em; /* 调整星星大小 */
      line-height: 1; /* 确保垂直对齐 */
      display: inline-flex;
      align-items: center; /* 保持对齐 */
    }
    .half-star {
      display: inline-block;
      width: 0.5em; /* 半颗星的宽度 */
      overflow: hidden;
      white-space: nowrap;
      vertical-align: middle;
    }
    .full-star {
      vertical-align: middle;
    }
  </style>
</head>
<body>

<div>
    __CONTENT__
</div>

<br><br>
<div>
To unsubscribe, remove your email in your Github Action setting.
</div>

</body>
</html>
"""

def get_empty_html():
  block_template = """
  <table border="0" cellpadding="0" cellspacing="0" width="100%" style="font-family: Arial, sans-serif; border: 1px solid #ddd; border-radius: 8px; padding: 16px; background-color: #f9f9f9;">
  <tr>
    <td style="font-size: 20px; font-weight: bold; color: #333;">
        No Papers Today. Take a Rest!
    </td>
  </tr>
  </table>
  """
  return block_template


def _escape_text(value: str | None) -> str:
    if value is None:
        return ""
    return html.escape(str(value), quote=True)


def _format_relative_relevance_scores(papers: list[Paper]) -> list[str]:
    valid_scores = [
        (idx, float(p.score))
        for idx, p in enumerate(papers)
        if p.score is not None
    ]
    if not valid_scores:
        return ["Unknown"] * len(papers)

    if len(valid_scores) == 1:
        only_idx, _ = valid_scores[0]
        result = ["Unknown"] * len(papers)
        result[only_idx] = "8.0/10"
        return result

    ranked = sorted(valid_scores, key=lambda item: item[1], reverse=True)
    max_rank = len(ranked) - 1
    display_scores: list[str] = ["Unknown"] * len(papers)
    top_display = 9.5
    bottom_display = 6.0
    for rank, (idx, _) in enumerate(ranked):
        relative_score = top_display - (top_display - bottom_display) * (rank / max_rank)
        display_scores[idx] = f"{relative_score:.1f}/10"
    return display_scores

def get_block_html(
    title: str,
    authors: str,
    rate: str,
    tldr: str,
    abstract: str,
    pdf_url: str,
    affiliations: str = None,
    show_tldr: bool = True,
    show_affiliations: bool = True,
):
    author_row = f"""
    <tr>
        <td style="font-size: 14px; color: #666; padding: 8px 0;">
            {_escape_text(authors)}
"""
    if show_affiliations:
        author_row += f"""
            <br>
            <i>{_escape_text(affiliations)}</i>
"""
    author_row += """
        </td>
    </tr>
"""

    summary_text = tldr if show_tldr and tldr else abstract
    summary_label = "TLDR" if show_tldr and tldr else "Abstract"

    tldr_row = ""
    if summary_text:
        tldr_row = f"""
    <tr>
        <td style="font-size: 14px; color: #333; padding: 8px 0;">
            <strong>{summary_label}:</strong> {_escape_text(summary_text)}
        </td>
    </tr>
"""

    return f"""
    <table border="0" cellpadding="0" cellspacing="0" width="100%" style="font-family: Arial, sans-serif; border: 1px solid #ddd; border-radius: 8px; padding: 16px; background-color: #f9f9f9;">
    <tr>
        <td style="font-size: 20px; font-weight: bold; color: #333;">
            {_escape_text(title)}
        </td>
    </tr>
    {author_row}
    <tr>
        <td style="font-size: 14px; color: #333; padding: 8px 0;">
            <strong>Relative Relevance:</strong> {rate}
        </td>
    </tr>
    {tldr_row}
    <tr>
        <td style="padding: 8px 0;">
            <a href="{_escape_text(pdf_url)}" style="display: inline-block; text-decoration: none; font-size: 14px; font-weight: bold; color: #fff; background-color: #d9534f; padding: 8px 16px; border-radius: 4px;">PDF</a>
        </td>
    </tr>
</table>
"""

def get_stars(score:float):
    full_star = '<span class="full-star">⭐</span>'
    half_star = '<span class="half-star">⭐</span>'
    low = 6
    high = 8
    if score <= low:
        return ''
    elif score >= high:
        return full_star * 5
    else:
        interval = (high-low) / 10
        star_num = math.ceil((score-low) / interval)
        full_star_num = int(star_num/2)
        half_star_num = star_num - full_star_num * 2
        return '<div class="star-wrapper">'+full_star * full_star_num + half_star * half_star_num + '</div>'


def render_email(papers:list[Paper], show_tldr: bool = True, show_affiliations: bool = True) -> str:
    parts = []
    if len(papers) == 0 :
        return framework.replace('__CONTENT__', get_empty_html())

    display_scores = _format_relative_relevance_scores(papers)

    for p, rate in zip(papers, display_scores):
        #rate = get_stars(p.score)
        author_list = [a for a in p.authors]
        num_authors = len(author_list)
        if num_authors <= 5:
            authors = ', '.join(author_list)
        else:
            authors = ', '.join(author_list[:3] + ['...'] + author_list[-2:])
        if p.affiliations is not None:
            affiliations = p.affiliations[:5]
            affiliations = ', '.join(affiliations)
            if len(p.affiliations) > 5:
                affiliations += ', ...'
        else:
            affiliations = 'Unknown Affiliation'
        parts.append(
            get_block_html(
                p.title,
                authors,
                rate,
                p.tldr,
                p.abstract,
                p.pdf_url,
                affiliations,
                show_tldr=show_tldr,
                show_affiliations=show_affiliations,
            )
        )

    content = '<br>' + '</br><br>'.join(parts) + '</br>'
    return framework.replace('__CONTENT__', content)
