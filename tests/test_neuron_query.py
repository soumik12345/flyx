from dataclasses import FrozenInstanceError
from operator import and_, or_

import pandas as pd
import pytest

from flyx.core import NeuronField, NeuronQuery


@pytest.fixture
def neurons():
    return pd.DataFrame(
        {
            "type": ["L1", "L2", "Tm3", "ORN", None],
            "class": ["visual", "visual", "visual", "olfactory", None],
            "side": ["left", "right", "left", "left", "right"],
        },
        index=pd.Index([50, 10, 40, 20, 30], name="bodyId"),
    )


@pytest.mark.parametrize(
    "query,expected",
    [
        (NeuronQuery.field("type").eq("L1"), [True, False, False, False, False]),
        (
            NeuronQuery.field("type").isin(["L1", "Tm3"]),
            [True, False, True, False, False],
        ),
        (
            NeuronQuery.field("class").is_null(),
            [False, False, False, False, True],
        ),
    ],
    ids=["equality", "membership", "missingness"],
)
def test_predicates_return_aligned_boolean_masks(neurons, query, expected):
    mask = query.evaluate(neurons)

    pd.testing.assert_series_equal(
        mask,
        pd.Series(expected, index=neurons.index, dtype=bool),
        check_names=False,
    )
    assert not mask.isna().any()


@pytest.mark.parametrize(
    "dtype,values,operand",
    [
        ("object", ["L1", "L2", None], "L1"),
        ("string", ["L1", "L2", pd.NA], "L1"),
        ("string[pyarrow]", ["L1", "L2", pd.NA], "L1"),
        ("category", ["L1", "L2", None], "L1"),
        ("Int64", [1, 2, pd.NA], 1),
        ("Float64", [0.5, 1.5, pd.NA], 0.5),
        ("boolean", [True, False, pd.NA], True),
    ],
)
@pytest.mark.parametrize("predicate", ["eq", "isin"])
def test_comparisons_preserve_missingness_under_negation(
    dtype, values, operand, predicate
):
    frame = pd.DataFrame({"annotation": pd.Series(values, dtype=dtype)})
    field = NeuronQuery.field("annotation")
    query = field.eq(operand) if predicate == "eq" else field.isin([operand])

    assert query.evaluate(frame).tolist() == [True, False, False]
    assert (~query).evaluate(frame).tolist() == [False, True, False]
    assert (~~query).evaluate(frame).tolist() == [True, False, False]


@pytest.mark.parametrize("dtype", ["object", "string"])
@pytest.mark.parametrize(
    "combine,expected",
    [
        (and_, [True, False, False, False, False, False, False, False, False]),
        (or_, [True, True, True, True, False, False, True, False, False]),
        (
            lambda left, right: ~(left & right),
            [False, True, False, True, True, True, False, True, False],
        ),
        (
            lambda left, right: ~(left | right),
            [False, False, False, False, True, False, False, False, False],
        ),
    ],
    ids=["and", "or", "negated-and", "negated-or"],
)
def test_nullable_truth_tables(dtype, combine, expected):
    # All pairs of true, false, and unknown, in that order. Literal expected
    # masks ensure a regression that fills nulls inside subexpressions fails.
    frame = pd.DataFrame(
        {
            "left": pd.Series(["match"] * 3 + ["other"] * 3 + [None] * 3, dtype=dtype),
            "right": pd.Series(["match", "other", None] * 3, dtype=dtype),
        }
    )
    left = NeuronQuery.field("left").eq("match")
    right = NeuronQuery.field("right").eq("match")

    assert combine(left, right).evaluate(frame).tolist() == expected


def test_nested_query_selects_rows_without_modifying_table(neurons):
    before = neurons.copy(deep=True)
    query = (
        NeuronQuery.field("type").isin(["L1", "Tm3"])
        & NeuronQuery.field("side").eq("left")
    ) | NeuronQuery.field("class").is_null()

    selected = neurons.loc[query.evaluate(neurons)]

    assert selected.index.tolist() == [50, 40, 30]
    pd.testing.assert_frame_equal(neurons, before)


def test_unknown_annotations_can_be_included_explicitly(neurons):
    field = NeuronQuery.field("class")
    query = ~field.eq("olfactory") | field.is_null()

    assert query.evaluate(neurons).tolist() == [True, True, True, False, True]


def test_is_null_distinguishes_missing_values_from_annotation_labels():
    frame = pd.DataFrame(
        {"class": [None, pd.NA, float("nan"), pd.NaT, "", "unclear", "visual"]}
    )
    query = NeuronQuery.field("class").is_null()

    assert query.evaluate(frame).tolist() == [
        True,
        True,
        True,
        True,
        False,
        False,
        False,
    ]
    assert (~query).evaluate(frame).tolist() == [
        False,
        False,
        False,
        False,
        True,
        True,
        True,
    ]


def test_empty_membership_does_not_turn_unknown_into_a_match_when_negated():
    frame = pd.DataFrame({"type": ["L1", None, "L2"]})
    query = NeuronQuery.field("type").isin([])

    assert query.evaluate(frame).tolist() == [False, False, False]
    assert (~query).evaluate(frame).tolist() == [True, False, True]


def test_membership_captures_values_before_collection_is_mutated(neurons):
    values = ["L1", "L2"]
    query = NeuronQuery.field("type").isin(values)
    values[:] = ["ORN"]

    assert query.evaluate(neurons).tolist() == [True, True, False, False, False]


def test_membership_consumes_generator_once_and_query_is_reusable(neurons):
    values = (value for value in ["L1", "Tm3", "L1"])
    query = NeuronQuery.field("type").isin(values)

    assert list(values) == []
    assert query.evaluate(neurons).tolist() == [True, False, True, False, False]
    assert query.evaluate(neurons.iloc[::-1]).tolist() == [
        False,
        False,
        True,
        False,
        True,
    ]


def test_duplicate_row_labels_are_preserved():
    frame = pd.DataFrame({"type": ["L1", "L2", "L1"]}, index=[7, 7, 2])
    mask = NeuronQuery.field("type").eq("L1").evaluate(frame)

    pd.testing.assert_index_equal(mask.index, frame.index)
    assert mask.tolist() == [True, False, True]
    assert frame.loc[mask].index.tolist() == [7, 2]


@pytest.mark.parametrize(
    "query",
    [
        NeuronQuery.field("type").eq("L1"),
        NeuronQuery.field("type").isin(["L1"]),
        NeuronQuery.field("type").is_null(),
        ~(NeuronQuery.field("type").eq("L1") | NeuronQuery.field("type").is_null()),
    ],
    ids=["equality", "membership", "missingness", "composition"],
)
def test_empty_table_returns_empty_boolean_mask(query):
    frame = pd.DataFrame({"type": pd.Series(dtype="string")})

    pd.testing.assert_series_equal(
        query.evaluate(frame),
        pd.Series(index=frame.index, dtype=bool),
        check_names=False,
    )


@pytest.mark.parametrize("name", ["", None, 1, False, []])
@pytest.mark.parametrize("builder", [NeuronField, NeuronQuery.field])
def test_invalid_field_names_are_rejected(builder, name):
    with pytest.raises(ValueError, match="nonempty string"):
        builder(name)


def test_field_names_are_used_exactly_as_supplied():
    frame = pd.DataFrame({" type ": ["L1"], "Type": ["L2"]})

    assert NeuronQuery.field(" type ").eq("L1").evaluate(frame).tolist() == [True]
    assert NeuronQuery.field("Type").eq("L2").evaluate(frame).tolist() == [True]
    with pytest.raises(KeyError, match="type"):
        NeuronQuery.field("type").eq("L1").evaluate(frame)


@pytest.mark.parametrize("value", [None, pd.NA, [], {}, ("L1",), object()])
@pytest.mark.parametrize("predicate", ["eq", "isin"])
def test_unsupported_comparison_values_are_rejected(value, predicate):
    field = NeuronQuery.field("type")

    with pytest.raises(TypeError, match="Python string, number, or boolean"):
        if predicate == "eq":
            field.eq(value)
        else:
            field.isin(["L1", value])


@pytest.mark.parametrize("predicate", ["eq", "isin"])
def test_nan_comparison_values_require_is_null(predicate):
    field = NeuronQuery.field("type")

    with pytest.raises(ValueError, match="is_null"):
        if predicate == "eq":
            field.eq(float("nan"))
        else:
            field.isin(["L1", float("nan")])


@pytest.mark.parametrize("values", ["L1", b"L1", 1, None])
def test_membership_rejects_strings_and_noniterables(values):
    with pytest.raises(TypeError):
        NeuronQuery.field("type").isin(values)


@pytest.mark.parametrize(
    "truth_test",
    [bool, lambda q: q and q, lambda q: q or q, lambda q: not q],
    ids=["bool", "and", "or", "not"],
)
def test_python_truth_testing_is_rejected(truth_test):
    query = NeuronQuery.field("type").eq("L1")

    with pytest.raises(TypeError, match="Combine queries"):
        truth_test(query)


@pytest.mark.parametrize("combine", [and_, or_], ids=["and", "or"])
@pytest.mark.parametrize("operand", [True, None, "L1", 1])
def test_composition_with_nonquery_operand_is_rejected(combine, operand):
    query = NeuronQuery.field("type").eq("L1")

    with pytest.raises(TypeError):
        combine(query, operand)


@pytest.mark.parametrize(
    "query",
    [
        NeuronQuery.field("missing").eq("L1"),
        NeuronQuery.field("missing").isin(["L1"]),
        NeuronQuery.field("missing").is_null(),
        NeuronQuery.field("type").eq("absent") & NeuronQuery.field("missing").eq("L1"),
    ],
    ids=["equality", "membership", "missingness", "composition"],
)
def test_missing_columns_are_rejected_at_evaluation(neurons, query):
    with pytest.raises(KeyError, match="missing"):
        query.evaluate(neurons)


@pytest.mark.parametrize("columns", [["type", "type"], ["type", "side", "side"]])
def test_duplicate_columns_are_rejected_even_if_not_referenced(columns):
    frame = pd.DataFrame([["L1"] * len(columns)], columns=columns)

    with pytest.raises(ValueError, match="column names must be unique"):
        NeuronQuery.field("type").eq("L1").evaluate(frame)


def test_unsupported_operation_is_rejected(neurons):
    query = NeuronQuery("unsupported", ())

    with pytest.raises(ValueError, match="Unsupported query operation"):
        query.evaluate(neurons)


@pytest.mark.parametrize(
    "instance,attribute,value",
    [
        (NeuronField("type"), "name", "class"),
        (NeuronQuery.field("type").eq("L1"), "op", "is_null"),
        (NeuronQuery.field("type").eq("L1"), "args", ("class", "visual")),
    ],
)
def test_fields_and_queries_are_frozen(instance, attribute, value):
    with pytest.raises(FrozenInstanceError):
        setattr(instance, attribute, value)
