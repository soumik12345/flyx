"""Build composable filters for neuron annotation tables.

Create predicates through `NeuronQuery.field`, combine them with `&`, `|`,
and `~`, and call `NeuronQuery.evaluate` to obtain a pandas selection mask.
Building a query does not read data or modify an annotation table.

Comparisons preserve missing annotations as unknown values while expressions
are combined. Only the final selection mask replaces unresolved unknowns with
`False`. Use `NeuronField.is_null` to select missing annotations explicitly.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


def _check_value(value):
    """Validate a scalar operand without coercing its type.

    Args:
        value (str | int | float | bool): Candidate equality or membership
            operand, checked against the supported Python scalar types.

    Raises:
        TypeError: The value is not a supported scalar type, including
            missing-value sentinels such as `None` and `pandas.NA`.
        ValueError: The value has a supported type but is missing, such as
            floating-point NaN. Use `NeuronField.is_null` instead.
    """
    if not isinstance(value, (str, int, float, bool)):
        raise TypeError(
            "Use a Python string, number, or boolean; use is_null() for missing values."
        )

    if pd.isna(value):
        raise ValueError("Use is_null() for missing values.")


@dataclass(frozen=True)
class NeuronField:
    """An immutable builder for predicates on one annotation column.

    Obtain a builder through `NeuronQuery.field` or construct it directly.
    Column existence is checked when the resulting query is evaluated, so
    builders can be reused across annotation tables with the same schema.

    Attributes:
        name: Exact column name. Names are case-sensitive and are not trimmed
            or otherwise normalized.

    Raises:
        ValueError: The name is not a nonempty string.

    Examples:
        Explicit field access supports Python keywords such as `class`:

        >>> from flyx import NeuronQuery as Q
        >>> query = Q.field("class").eq("olfactory")
        >>> query.op
        'eq'
    """

    name: str

    def __post_init__(self):
        """Validate the field name after dataclass initialization.

        Raises:
            ValueError: The name is not a string or is an empty string.
        """
        if not isinstance(self.name, str) or not self.name:
            raise ValueError("Field name must be a nonempty string.")

    def eq(self, value) -> NeuronQuery:
        """Build an equality predicate for a scalar annotation value.

        Missing annotations remain unknown during evaluation, including
        when the predicate is negated. Missing comparison operands must be
        expressed with `is_null` instead.

        Args:
            value (str | int | float | bool): Scalar value to compare with
                the annotation column using pandas equality semantics.

        Returns:
            A query describing the equality comparison.

        Raises:
            TypeError: The operand is not a supported scalar type.
            ValueError: The operand is a missing scalar such as NaN.

        Examples:
            >>> import pandas as pd
            >>> from flyx import NeuronQuery as Q
            >>> neurons = pd.DataFrame({"class": ["visual", "olfactory", None]})
            >>> Q.field("class").eq("visual").evaluate(neurons).tolist()
            [True, False, False]
        """
        _check_value(value)
        return NeuronQuery("eq", (self.name, value))

    def isin(self, values) -> NeuronQuery:
        """Build a membership predicate from an iterable of scalar values.

        Values are consumed and copied into a tuple when the query is built.
        Later changes to the original collection do not change the query.
        An empty iterable matches no nonmissing annotations; missing
        annotations still evaluate to unknown until final mask conversion.

        Args:
            values (Iterable[str | int | float | bool]): Candidate annotation
                values. A standalone string or bytes object is not accepted.

        Returns:
            A query that checks membership in the captured collection.

        Raises:
            TypeError: The argument is a string, bytes object, or noniterable,
                or an element is not a supported scalar type.
            ValueError: An element is a missing scalar such as NaN.

        Examples:
            >>> import pandas as pd
            >>> from flyx import NeuronQuery as Q
            >>> neurons = pd.DataFrame({"type": ["L1", "L2", "Tm3", None]})
            >>> Q.field("type").isin(["L1", "L2"]).evaluate(neurons).tolist()
            [True, True, False, False]
        """
        if isinstance(values, (str, bytes)):
            raise TypeError("Pass a collection of values, not a single string.")

        values = tuple(values)
        for value in values:
            _check_value(value)

        return NeuronQuery("isin", (self.name, values))

    def is_null(self) -> NeuronQuery:
        """Build a predicate that explicitly selects missing annotations.

        Missingness follows pandas `isna` semantics. Empty strings and
        annotation labels such as `"unclear"` are ordinary nonmissing values.

        Returns:
            A query that evaluates to `True` for missing values and `False`
            for nonmissing values.

        Examples:
            >>> import pandas as pd
            >>> from flyx import NeuronQuery as Q
            >>> neurons = pd.DataFrame({"class": [None, "", "unclear"]})
            >>> Q.field("class").is_null().evaluate(neurons).tolist()
            [True, False, False]
        """
        return NeuronQuery("is_null", (self.name,))


@dataclass(frozen=True)
class NeuronQuery:
    """An immutable expression for selecting neuron annotation rows.

    Build predicates with `field` and combine them with `&` (conjunction),
    `|` (disjunction), and `~` (negation). These operations construct new
    expression trees without evaluating or modifying their operands.
    Python's `and`, `or`, and `not` operators are not supported.

    Evaluation uses nullable boolean logic. A comparison against a missing
    annotation produces an unknown value, and negating unknown remains
    unknown. Conjunction and disjunction retain unknowns only where the
    result cannot otherwise be determined: `False & unknown` is `False`,
    while `True | unknown` is `True`. `evaluate` converts any remaining
    unknown results to `False` at the outermost boundary.

    Attributes:
        op: Operation name: `"eq"`, `"isin"`, `"is_null"`, `"and"`,
            `"or"`, or `"not"`.
        args: Operation operands. Equality stores `(field_name, value)`;
            membership stores `(field_name, values_tuple)`; missingness
            stores `(field_name,)`; conjunction and disjunction store
            `(left_query, right_query)`; negation stores `(query,)`.

    Note:
        Prefer the builders and composition operators. The dataclass
        constructor does not validate the operation name or operand
        structure, and freezing it does not recursively freeze arbitrary
        objects supplied directly as operands.

    Examples:
        Exclude known olfactory annotations while leaving unknowns unselected:

        >>> import pandas as pd
        >>> from flyx import NeuronQuery as Q
        >>> neurons = pd.DataFrame({"class": ["visual", "olfactory", None]})
        >>> olfactory = Q.field("class").eq("olfactory")
        >>> (~olfactory).evaluate(neurons).tolist()
        [True, False, False]

        Include missing annotations explicitly when the experiment requires it:

        >>> query = ~olfactory | Q.field("class").is_null()
        >>> query.evaluate(neurons).tolist()
        [True, False, True]
    """

    op: str
    args: tuple

    @staticmethod
    def field(name: str) -> NeuronField:
        """Create a predicate builder for a named annotation column.

        Args:
            name: Exact, nonempty column name. The column need not exist
                until a resulting query is evaluated.

        Returns:
            A field builder exposing `eq`, `isin`, and `is_null` predicates.

        Raises:
            ValueError: The name is not a nonempty string.
        """
        return NeuronField(name)

    def __and__(self, other):
        """Build the conjunction `self & other` without evaluating it.

        Args:
            other (object): The right-hand operand, expected to be a query.

        Returns:
            result (object): A new `NeuronQuery` for conjunction, or `NotImplemented`
                when the operand is not a query, allowing Python's operator
                dispatch to handle unsupported operand types.
        """
        if not isinstance(other, NeuronQuery):
            return NotImplemented
        return NeuronQuery("and", (self, other))

    def __or__(self, other):
        """Build the disjunction `self | other` without evaluating it.

        Args:
            other (object): The right-hand operand, expected to be a query.

        Returns:
            result (object): A new `NeuronQuery` for disjunction, or `NotImplemented`
                when the operand is not a query, allowing Python's operator
                dispatch to handle unsupported operand types.
        """
        if not isinstance(other, NeuronQuery):
            return NotImplemented
        return NeuronQuery("or", (self, other))

    def __invert__(self):
        """Build the negation `~self` without evaluating it.

        Returns:
            query (NeuronQuery): A new negated expression. Unknown annotation
                comparisons remain unknown when the expression is evaluated.
        """
        return NeuronQuery("not", (self,))

    def __bool__(self):
        """Reject implicit truth testing of an unevaluated query.

        Raises:
            TypeError: Always raised for `bool(query)`, conditional truth
                testing, and Python's `and`, `or`, and `not` operators when
                they attempt to test the query. Use `&`, `|`, and `~` instead.
        """
        raise TypeError("Combine queries with &, |, and ~; not and, or, or not.")

    def evaluate(self, neurons: pd.DataFrame) -> pd.Series:
        """Evaluate the complete expression into a boolean selection mask.

        Unknown results become `False` only after all subexpressions have
        been combined. The result preserves the input table's row order and
        index and can be used with `neurons.loc[mask]`. The table is not
        modified, and no rows or connections are loaded from disk.

        Args:
            neurons: Annotation table with unique column names and every
                column referenced by the expression. An empty table is
                supported when the required columns are present.

        Returns:
            A pandas Series with ordinary boolean dtype, the same index as
            `neurons`, and no missing values. `True` marks a selected row.

        Raises:
            KeyError: An annotation column referenced by the query is absent.
            ValueError: Column names are duplicated or the expression
                contains an unsupported operation.

        Examples:
            >>> import pandas as pd
            >>> from flyx import NeuronQuery as Q
            >>> neurons = pd.DataFrame({"type": ["L1", "L2"]}, index=[10, 20])
            >>> mask = Q.field("type").eq("L1").evaluate(neurons)
            >>> neurons.loc[mask].index.tolist()
            [10]
        """
        if not neurons.columns.is_unique:
            raise ValueError("Neuron column names must be unique.")

        result = self._evaluate_nullable(neurons)
        return result.fillna(False).astype(bool)

    def _evaluate_nullable(self, neurons: pd.DataFrame) -> pd.Series:
        """Evaluate recursively without resolving unknown comparisons.

        Recursive branches call this method rather than `evaluate` so that
        missing values remain unknown through composition and negation.
        Leaf comparisons restore missing annotations to `pandas.NA` even
        when pandas equality or membership would otherwise return `False`.

        Args:
            neurons: Annotation table whose column names have already been
                checked for uniqueness by `evaluate`.

        Returns:
            A Series with pandas nullable `boolean` dtype and the input
            table's index. Unresolved comparisons contain `pandas.NA`.

        Raises:
            KeyError: A referenced annotation column is absent.
            ValueError: The expression contains an unsupported operation.
        """
        if self.op == "and":
            left = self.args[0]._evaluate_nullable(neurons)
            right = self.args[1]._evaluate_nullable(neurons)
            return left & right

        if self.op == "or":
            left = self.args[0]._evaluate_nullable(neurons)
            right = self.args[1]._evaluate_nullable(neurons)
            return left | right

        if self.op == "not":
            return ~self.args[0]._evaluate_nullable(neurons)

        if self.op not in {"eq", "isin", "is_null"}:
            raise ValueError(f"Unsupported query operation: {self.op}")

        column = neurons[self.args[0]]

        if self.op == "is_null":
            return column.isna().astype("boolean")

        if self.op == "eq":
            result = column.eq(self.args[1])
        else:
            result = column.isin(self.args[1])

        return result.astype("boolean").mask(column.isna(), pd.NA)
