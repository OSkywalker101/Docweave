"""Visual understanding pipeline.

An image is not a file: it is a claim about something. This package decides what
an image *is* -- text page, diagram, table, chart, photo, screenshot -- so the
rest of the engine picks the tool that actually applies.

Without that decision every image takes the same path, or worse, an expensive
model is run on a spreadsheet screenshot to be told it is a photo.

The rule the rest of this project lives by applies here too: **an unknown is a
reportable outcome.** ``classify_image`` returns ``UNKNOWN`` with the signals it
used rather than guessing, because a wrong routing decision costs far more than
an honest "I cannot tell".
"""

from docweave.visual.classify import (
    VisualClassification,
    VisualKind,
    classify_image,
)

__all__ = ["VisualClassification", "VisualKind", "classify_image"]
