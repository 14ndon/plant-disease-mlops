"""PIL checks, manifest tables, and Great Expectations on tabular summaries."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from PIL import Image

try:
    import great_expectations as gx
    from great_expectations.core import ExpectationSuite
except ImportError:  # pragma: no cover
    gx = None  # type: ignore[assignment]
    ExpectationSuite = None  # type: ignore[misc]


def validate_pil_image(image: Any) -> tuple[bool, str]:
    """Validate a decoded PIL image (HuggingFace ``datasets`` image column)."""
    try:
        if not isinstance(image, Image.Image):
            return False, f"Expected PIL.Image, got {type(image).__name__}"

        width, height = image.size
        if width < 50 or height < 50:
            return False, "Image too small"

        if image.mode not in ("RGB", "RGBA", "L"):
            return False, f"Unsupported image mode: {image.mode}"

        np.array(image.convert("RGB"))
        return True, "Valid"
    except Exception as e:  # noqa: BLE001
        return False, str(e)


def validate_hf_example(example: dict, num_classes: int) -> tuple[bool, str]:
    ok, msg = validate_pil_image(example["image"])
    if not ok:
        return ok, msg
    label = int(example["label"])
    if label < 0 or label >= num_classes:
        return False, f"Invalid label {label} (num_classes={num_classes})"
    return True, "Valid"


def build_manifest_dataframe(
    dataset,
    *,
    limit: int | None = None,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    n = len(dataset) if limit is None else min(limit, len(dataset))
    for i in range(n):
        ex = dataset[i]
        img = ex["image"]
        if not isinstance(img, Image.Image):
            img = Image.open(img)
        img = img.convert("RGB")
        w, h = img.size
        rows.append(
            {
                "idx": i,
                "label": int(ex["label"]),
                "width": w,
                "height": h,
                "mode": img.mode,
            }
        )
    return pd.DataFrame(rows)


def validate_manifest_with_great_expectations(
    df: pd.DataFrame,
    num_classes: int,
    *,
    suite_name: str = "plant_disease_validation",
):
    """
    Run Great Expectations on a manifest built by ``build_manifest_dataframe``.

    Returns the GX ``ValidationResult`` (``.success`` is overall pass/fail).
    """
    if gx is None or ExpectationSuite is None:
        raise ImportError("great_expectations is required for this check")

    context = gx.get_context(mode="ephemeral")
    data_source = context.data_sources.add_pandas("plant_manifest_source")
    data_asset = data_source.add_dataframe_asset(name="manifest")
    batch_def = data_asset.add_batch_definition_whole_dataframe("whole")
    batch_request = batch_def.build_batch_request(batch_parameters={"dataframe": df})

    suite = context.suites.add(ExpectationSuite(name=suite_name))
    suite.add_expectation(
        gx.expectations.ExpectColumnValuesToNotBeNull(column="label")
    )
    suite.add_expectation(
        gx.expectations.ExpectColumnValuesToBeBetween(
            column="label",
            min_value=0,
            max_value=num_classes - 1,
        )
    )
    suite.add_expectation(
        gx.expectations.ExpectColumnValuesToBeBetween(
            column="width",
            min_value=50,
            max_value=8192,
        )
    )
    suite.add_expectation(
        gx.expectations.ExpectColumnValuesToBeBetween(
            column="height",
            min_value=50,
            max_value=8192,
        )
    )
    suite.add_expectation(
        gx.expectations.ExpectColumnValuesToBeInSet(
            column="mode",
            value_set=["RGB", "RGBA", "L"],
        )
    )
    context.suites.add_or_update(suite)

    validator = context.get_validator(
        batch_request=batch_request,
        expectation_suite_name=suite_name,
    )
    return validator.validate()


def validate_dataset_sample(
    dataset,
    num_classes: int,
    *,
    max_samples: int | None = 500,
) -> tuple[int, list[tuple[int, str]]]:
    """Scan up to ``max_samples`` rows; return (invalid_count, first issues)."""
    invalid: list[tuple[int, str]] = []
    n = len(dataset) if max_samples is None else min(len(dataset), max_samples)
    for idx in range(n):
        ok, msg = validate_hf_example(dataset[idx], num_classes)
        if not ok:
            invalid.append((idx, msg))
    return len(invalid), invalid
