"""Accuracy scoring against the synthetic ground truth.

What this measures, and why each metric is here
-----------------------------------------------

**Character Error Rate** is the standard transcription metric, and on its own it
is dangerously easy to misread. A system that omits half a document *and* emits
equal garbage can score the same CER as a clean transcription. So CER is always
reported next to:

**Word recall** — fraction of reference words present in the output. This is what
catches *omission*, which is the dominant failure mode of document converters.
**Word precision** — fraction of output words that came from the source. This is
what catches *invention*, which is the failure mode of a converter that
hallucinates structure or OCRs noise into text.

Together they bracket the error, and a report that shows only CER is hiding one
of the two ways to be wrong.

**Reading-order accuracy** — the fraction of reference blocks that appear in the
output in the correct relative order. Catches multi-column scrambling, which CER
cannot see at all.

**Table structure** — header agreement and cell coverage. A converter that
flattens a table to prose keeps every character and still destroys the meaning,
so table fidelity has to be measured separately from text fidelity.

**Coverage** — pages, slides or sheets accounted for, so "extracted nothing"
cannot quietly score as high recall on a document with no ground-truth text.

Fairness between systems
------------------------
Both systems are reduced to Markdown, both are stripped of HTML comments and
diagnostic trailers, and both are split into blocks by the same function. Neither
system gets credit for structure the other cannot express. Where a system has
no equivalent (MarkItDown exposes no intermediate representation), that is
stated rather than scored as a win.
"""

from __future__ import annotations

import re
import unicodedata
from collections import Counter
from collections.abc import Iterable, Sequence
from dataclasses import asdict, dataclass, field

from tests.ground_truth import GroundTruth, reference_table_cells

# --------------------------------------------------------------------------
# normalisation
# --------------------------------------------------------------------------

_COMMENT_RE = re.compile(r"<!--.*?-->", re.DOTALL)
_DIAGNOSTIC_BLOCK_RE = re.compile(r"\n---\n\n<!--\s*docweave:diagnostics\s*-->.*\Z", re.DOTALL)
_WS_RE = re.compile(r"\s+")
_PUNCT_RE = re.compile(r"[^\w\s]", re.UNICODE)
_MD_NOISE_RE = re.compile(r"^[#>\-*+\d.|\s`]+$")
_HTML_TAG_RE = re.compile(r"<[^>]+>")
_MD_IMAGE_RE = re.compile(r"!\[([^\]]*)\]\([^)]*\)")
_MD_LINK_RE = re.compile(r"\[([^\]]*)\]\([^)]*\)")
#: Text DocWeave adds to explain an asset it did not extract. It is scaffolding,
#: not document content, and scoring it as content would penalise a system for
#: reporting its own behaviour. Applied to every system; a no-op for those that
#: do not emit it.
_SCAFFOLD_RE = re.compile(
    r"_\(\s*asset not extracted[^)]*\)_"
    r"|^\s*>\s*\d+\s+formula cell\(s\)\..*$"
    r"|\s*\(no cached result\)"
    r"|^\s*>\s*Truncated:.*$"
    r"|^\s*>\s*\*\*Speaker notes\*\*\s*$"
    r"|\bWorksheet:\s*\S+"
    r"|\((?:hidden)\)"
    # A code fence's language token is a renderer choice, not document content.
    r"|^\s*```[A-Za-z0-9_+#.-]*\s*$"
    # The ``- A1: value`` prefix of a list-rendered spreadsheet. Matched only
    # in that form, so a document that genuinely contains "A1" is untouched.
    r"|^\s*-\s+[A-Z]{1,3}\d+\s*:",
    re.IGNORECASE | re.MULTILINE,
)
_MEDIA_LIKE_RE = re.compile(r"^[\w.-]+/[\w.+-]+$|^\S+\.\w{2,4}$")
_ENTITY_RE = re.compile(r"&(?:amp|lt|gt|quot|#\d+|#x[0-9a-fA-F]+);")


def _image_alt(match: re.Match[str]) -> str:
    """Keep a real description, drop a synthesised media type or filename.

    ``![image/png](image://a3)`` is a placeholder the engine invented, so its
    label is not content. ``![Company logo](logo.png)`` came from the source
    and is.
    """
    alt = match.group(1).strip()
    if not alt or _MEDIA_LIKE_RE.match(alt):
        return " "
    return alt


def strip_markup(text: str) -> str:
    """Remove markup and engine scaffolding, keeping document content.

    Markdown and HTML syntax are not document content: scoring ``<td>`` or an
    asset-placeholder sentence as if the source contained those words would
    penalise a system for saying how it is reporting, not for what it read.
    Link labels and genuine image descriptions are kept, because those do come
    from the source.
    """
    # The diagnostics trailer must go before comments are stripped: its marker is
    # an HTML comment, and removing that first leaves the report text behind.
    text = _DIAGNOSTIC_BLOCK_RE.sub(" ", text)
    text = _COMMENT_RE.sub(" ", text)
    text = _SCAFFOLD_RE.sub(" ", text)
    text = _HTML_TAG_RE.sub(" ", text)
    text = _MD_IMAGE_RE.sub(_image_alt, text)
    text = _MD_LINK_RE.sub(r"\1", text)
    text = _ENTITY_RE.sub(" ", text)
    return text


def normalize(text: str) -> str:
    """Casefold, strip punctuation, collapse whitespace.

    Deliberately conservative: it removes only things that carry no information
    about content, so a missing word cannot be normalised into matching one.
    """
    text = unicodedata.normalize("NFC", strip_markup(text))
    text = text.casefold()
    text = _PUNCT_RE.sub(" ", text)
    return _WS_RE.sub(" ", text).strip()


def words(text: str) -> list[str]:
    normalized = normalize(text)
    return normalized.split() if normalized else []


# --------------------------------------------------------------------------
# distance
# --------------------------------------------------------------------------


def levenshtein(a: str, b: str) -> int:
    """Edit distance. Uses RapidFuzz when available; the fallback is O(n*m)."""
    try:
        from rapidfuzz.distance import Levenshtein

        return int(Levenshtein.distance(a, b))
    except ImportError:
        pass
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)
    previous = list(range(len(b) + 1))
    for i, ca in enumerate(a, start=1):
        current = [i]
        for j, cb in enumerate(b, start=1):
            current.append(
                min(
                    previous[j] + 1,
                    current[j - 1] + 1,
                    previous[j - 1] + (ca != cb),
                )
            )
        previous = current
    return previous[-1]


def character_error_rate(reference: str, hypothesis: str) -> float:
    """CER as a fraction. An empty reference scores 0.0 when both are empty."""
    ref = normalize(reference)
    hyp = normalize(hypothesis)
    if not ref:
        return 0.0 if not hyp else 1.0
    return levenshtein(ref, hyp) / len(ref)


def multiset_scores(reference: str, hypothesis: str) -> tuple[float, float, float]:
    """Word-level (recall, precision, F1) using multiset overlap.

    Multiset rather than set: repeating a word twice in the source and once in
    the output is a real omission and must cost something.
    """
    ref = Counter(words(reference))
    hyp = Counter(words(hypothesis))
    if not ref and not hyp:
        return 1.0, 1.0, 1.0
    overlap = sum((ref & hyp).values())
    recall = overlap / sum(ref.values()) if ref else 1.0
    precision = overlap / sum(hyp.values()) if hyp else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    return recall, precision, f1


# --------------------------------------------------------------------------
# blocks and reading order
# --------------------------------------------------------------------------


def split_blocks(markdown: str) -> list[str]:
    """Split Markdown into content blocks, ignoring syntax-only lines."""
    cleaned = strip_markup(markdown)
    blocks: list[str] = []
    for chunk in re.split(r"\n\s*\n", cleaned):
        lines = [line.strip() for line in chunk.splitlines() if line.strip()]
        if not lines:
            continue
        if all(_MD_NOISE_RE.match(line) for line in lines):
            # A rule or a fence boundary carries no content.
            continue
        block = " ".join(lines)
        if block.strip():
            blocks.append(block)
    return blocks


def _block_key(block: str) -> str:
    normalized = normalize(block)
    return normalized[:80]


def reading_order_accuracy(reference_blocks: Sequence[str], output_blocks: Sequence[str]) -> float:
    """Fraction of reference blocks appearing in the correct relative order.

    Compares by prefix match, because a converter may legitimately wrap or
    reflow a block without changing its identity.
    """
    if not reference_blocks:
        return 1.0
    output_keys = [_block_key(b) for b in output_blocks]
    positions: list[int] = []
    for block in reference_blocks:
        key = _block_key(block)
        for index, candidate in enumerate(output_keys):
            if candidate.startswith(key) or key.startswith(candidate):
                positions.append(index)
                break
        else:
            positions.append(-1)

    present = [p for p in positions if p >= 0]
    if not present:
        return 0.0
    if len(present) == 1:
        return 1.0

    # Fraction of ordered pairs that keep their relative order.
    inversions = 0
    total = 0
    for i in range(len(present)):
        for j in range(i + 1, len(present)):
            total += 1
            if present[i] > present[j]:
                inversions += 1
    found = len(present) / len(reference_blocks)
    order = 1.0 - inversions / total if total else 1.0
    return round(found * order, 4)


# --------------------------------------------------------------------------
# tables
# --------------------------------------------------------------------------


@dataclass
class TableScore:
    expected: int = 0
    detected: int = 0
    header_match: float = 0.0
    cell_recall: float = 0.0
    cell_precision: float = 0.0

    @property
    def f1(self) -> float:
        if self.cell_recall + self.cell_precision == 0:
            return 0.0
        return (2 * self.cell_recall * self.cell_precision) / (
            self.cell_recall + self.cell_precision
        )


def score_tables(truth: GroundTruth, output_tables: Sequence[dict]) -> TableScore:
    """Compare detected tables against the expected ones.

    ``output_tables`` is a list of ``{"header": [...], "cells": [...]}`` dicts,
    which both systems are reduced to by the caller.
    """
    score = TableScore(expected=len(truth.tables), detected=len(output_tables))
    if not truth.tables:
        return score

    expected_cells = reference_table_cells(truth)
    detected_cells: set[str] = set()
    headers_ok = 0
    for index, expected in enumerate(truth.tables):
        if index < len(output_tables):
            actual = output_tables[index]
            if normalize(" ".join(actual.get("header", []))) == normalize(
                " ".join(expected.header)
            ):
                headers_ok += 1
            detected_cells.update(normalize(c) for c in actual.get("cells", []) if c)

    normalized_expected = {normalize(c) for c in expected_cells}
    score.header_match = headers_ok / len(truth.tables)
    if normalized_expected:
        score.cell_recall = len(normalized_expected & detected_cells) / len(normalized_expected)
    if detected_cells:
        score.cell_precision = len(normalized_expected & detected_cells) / len(detected_cells)
    return score


# --------------------------------------------------------------------------
# per-document result
# --------------------------------------------------------------------------


@dataclass
class DocumentScore:
    name: str
    fmt: str
    ok: bool = True
    error: str | None = None
    cer: float = 0.0
    word_recall: float = 0.0
    word_precision: float = 0.0
    word_f1: float = 0.0
    reading_order: float = 0.0
    tables: TableScore = field(default_factory=TableScore)
    container_coverage: float | None = None
    reference_blocks: int = 0
    matched_blocks: int = 0

    def to_dict(self) -> dict:
        payload = asdict(self)
        payload["tables"] = asdict(self.tables)
        return payload


def score_document(
    truth: GroundTruth,
    markdown: str,
    output_tables: Sequence[dict],
    *,
    containers_expected: int | None = None,
    containers_found: int | None = None,
) -> DocumentScore:
    """Score one document's Markdown output against its ground truth."""
    score = DocumentScore(name=truth.name, fmt=truth.fmt)

    # Running furniture that correct output omits must not count as an omission.
    for excluded in truth.excluded:
        markdown = markdown.replace(excluded, " ")
    truth_blocks = [b for b in truth.blocks if not any(e in b for e in truth.excluded)]

    reference_text = " ".join(truth_blocks)
    # A spreadsheet's content lives in cells, not paragraphs. Scoring only the
    # paragraph text would score a perfectly recovered sheet as total invention.
    table_words: list[str] = []
    for table in truth.tables:
        table_words.extend(c for c in table.header if c)
        for row in table.rows:
            table_words.extend(c for c in row if c)
    if table_words:
        reference_text = f"{reference_text} {' '.join(table_words)}".strip()

    score.cer = round(character_error_rate(reference_text, markdown), 4)
    recall, precision, f1 = multiset_scores(reference_text, markdown)
    score.word_recall = round(recall, 4)
    score.word_precision = round(precision, 4)
    score.word_f1 = round(f1, 4)

    output_blocks = split_blocks(markdown)
    order_blocks = truth_blocks or table_words
    score.reading_order = reading_order_accuracy(order_blocks, output_blocks)
    score.reference_blocks = len(order_blocks)
    output_keys = [_block_key(b) for b in output_blocks]
    score.matched_blocks = sum(
        1
        for block in order_blocks
        if any(
            k.startswith(_block_key(block)) or _block_key(block).startswith(k) for k in output_keys
        )
    )

    score.tables = score_tables(truth, output_tables)

    expected_containers = containers_expected
    if expected_containers is None:
        expected_containers = truth.pages or truth.slides or truth.sheets
    if expected_containers:
        score.container_coverage = round(
            min(containers_found or 0, expected_containers) / expected_containers, 4
        )
    return score


def aggregate(scores: Sequence[DocumentScore]) -> dict:
    """Mean of each metric over the documents that were scored."""
    usable = [s for s in scores if s.ok]
    if not usable:
        return {"documents": 0}

    def mean(values: Iterable[float]) -> float:
        collected = [v for v in values if v is not None]
        return round(sum(collected) / len(collected), 4) if collected else 0.0

    by_format: dict[str, list[DocumentScore]] = {}
    for score in usable:
        by_format.setdefault(score.fmt, []).append(score)

    # A document with no expected table has nothing to recall. Averaging it in as
    # 0.0 invents failures and, worse, hides real gains by burying them under
    # documents that were never asked the question.
    with_tables = [s for s in usable if s.tables.expected]

    return {
        "documents": len(usable),
        "table_documents": len(with_tables),
        "cer": mean(s.cer for s in usable),
        "word_recall": mean(s.word_recall for s in usable),
        "word_precision": mean(s.word_precision for s in usable),
        "word_f1": mean(s.word_f1 for s in usable),
        "reading_order": mean(s.reading_order for s in usable),
        "table_cell_recall": mean(s.tables.cell_recall for s in with_tables),
        "table_cell_precision": mean(s.tables.cell_precision for s in with_tables),
        "container_coverage": mean(
            s.container_coverage for s in usable if s.container_coverage is not None
        ),
        "by_format": {
            fmt: {
                "documents": len(group),
                "word_recall": mean(s.word_recall for s in group),
                "word_f1": mean(s.word_f1 for s in group),
                "cer": mean(s.cer for s in group),
            }
            for fmt, group in sorted(by_format.items())
        },
    }


# --------------------------------------------------------------------------
# markdown table extraction (shared by both systems)
# --------------------------------------------------------------------------

_PIPE_ROW = re.compile(r"^\s*\|(.+)\|\s*$")
_HTML_ROW = re.compile(r"<tr>(.*?)</tr>", re.DOTALL | re.IGNORECASE)
_HTML_CELL = re.compile(r"<t[hd][^>]*>(.*?)</t[hd]>", re.DOTALL | re.IGNORECASE)
_SHEET_LINE = re.compile(r"^\s*-\s+(.*)$")
_SHEET_REF = re.compile(r"\s*[A-Z]{1,3}\d+\s*:")
_CODE_FENCE = re.compile(r"^\s*```")


def _parse_sheet_cells(text: str) -> list[str]:
    """Split a spreadsheet row line into its cell values.

    A scanner rather than a regex, because the renderer quotes a value that
    contains what would otherwise look like another cell. A regex cannot see
    that ``; B1:`` inside a quoted string is not a separator, so it split one
    cell into two — which is exactly the bug the quoting exists to prevent, in
    the consumer this time.
    """
    values: list[str] = []
    index, end = 0, len(text)
    while index < end:
        reference = _SHEET_REF.match(text, index)
        if not reference:
            break
        index = reference.end()
        while index < end and text[index] in " \t":
            index += 1
        if index < end and text[index] == '"' and '"' in text[index + 1 :]:
            # Only a quote with a partner opens a quoted value. A cell whose
            # entire content is one double quote has no partner, and treating
            # the lone quote as an opener silently dropped the cell.
            index += 1
            chunk: list[str] = []
            while index < end:
                if text[index] == '"':
                    if text[index : index + 2] == '""':
                        chunk.append('"')
                        index += 2
                        continue
                    index += 1
                    break
                chunk.append(text[index])
                index += 1
            values.append("".join(chunk))
        else:
            start = index
            while index < end:
                if text[index] == ";" and _SHEET_REF.match(text, index + 1):
                    break
                index += 1
            values.append(text[start:index].strip())
        if index < end and text[index] == ";":
            index += 1
    return [v for v in values if v]


def tables_from_markdown(markdown: str) -> list[dict]:
    """Pull tables out of Markdown, handling pipe grids, HTML and cell lists.

    One extractor for both systems, so neither is advantaged by the format its
    renderer happens to prefer.

    Every branch must advance the line index, or a lookahead match on a later
    line loops forever. That invariant is enforced here rather than left to each
    branch: two separate spin bugs in this function were only found because one
    document happened to have the shape that triggered them.
    """
    tables: list[dict] = []
    lines = markdown.splitlines()
    index = 0
    total = len(lines)

    while index < total:
        entry = index
        try:
            line = lines[index]

            if _CODE_FENCE.match(line):
                index += 1
                while index < total and not _CODE_FENCE.match(lines[index]):
                    index += 1
                index += 1
                continue

            if "<table" in line or "<tr" in line:
                block: list[str] = []
                while index < total:
                    block.append(lines[index])
                    closed = "</table>" in lines[index]
                    index += 1
                    if closed:
                        break
                table = _from_html("\n".join(block))
                if table["cells"]:
                    tables.append(table)
                continue

            if _PIPE_ROW.match(line):
                rows: list[list[str]] = []
                header: list[str] = []
                while index < total:
                    match = _PIPE_ROW.match(lines[index])
                    if not match:
                        break
                    cells = [c.strip() for c in match.group(1).split("|")]
                    if rows and all(c and set(c) <= set("-: ") for c in cells):
                        index += 1
                        continue  # separator row carries no content
                    if not rows:
                        header = cells
                    rows.append(cells)
                    index += 1
                if rows:
                    tables.append(
                        {
                            "header": header if any(c.strip(" -:") for c in header) else [],
                            "cells": [c for row in rows for c in row if c.strip()],
                        }
                    )
                continue

            if _SHEET_LINE.match(line) and _SHEET_REF.match(line[2:]):
                rows: list[list[str]] = []
                while index < total:
                    match = _SHEET_LINE.match(lines[index])
                    if not match or not _SHEET_REF.match(match.group(1)):
                        break
                    # The references (A1, B2) position the value but are not content.
                    rows.append(_parse_sheet_cells(match.group(1)))
                    index += 1
                if rows:
                    tables.append(
                        {"header": rows[0], "cells": [c for row in rows for c in row]}
                    )
                continue

            index += 1
        finally:
            if index == entry:
                # A branch consumed a line without making progress. Skip it
                # rather than spin on it forever.
                index += 1
    return tables


def _from_html(block: str) -> dict:
    rows = [_HTML_CELL.findall(row) for row in _HTML_ROW.findall(block)]
    rows = [[c.strip() for c in row] for row in rows if row]
    header = rows[0] if rows and all(c.strip() for c in rows[0]) else []
    return {"header": header, "cells": [c for row in rows for c in row if c.strip()]}


def strip_comment_noise(markdown: str) -> str:
    """Remove prose that cannot be table content, keeping all table markup.

    This runs *before* :func:`tables_from_markdown`, so it must not strip the
    HTML tags and cell-reference lists the extractor parses. Stripping markup
    first and then looking for tables finds nothing, which silently scored every
    HTML-fallback table as zero cells.
    """
    text = _DIAGNOSTIC_BLOCK_RE.sub(" ", markdown)
    return _COMMENT_RE.sub(" ", text)
