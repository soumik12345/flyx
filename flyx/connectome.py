"""Discover and validate local [MaleCNS v1.0](https://male-cns.janelia.org/) connectome datasets.

Use `Connectome.from_directory` to create an immutable dataset handle after
checking the required Feather files and their schemas. Table contents remain
on disk until explicitly read by the caller.
"""

import os
from dataclasses import dataclass

import pyarrow as pa


def _validate_schema(
    path: str,
    required: dict[str, pa.DataType],
) -> None:
    """Validate required columns and their exact Arrow data types.

    The file is memory-mapped to inspect its Arrow IPC schema without loading
    the full table. Additional columns are permitted, and column order does
    not affect validation.

    Args:
        path: Path to an Arrow IPC file, such as a Feather V2 file.
        required: Mapping of required column names to their expected Arrow
            data types. Types must match exactly; no coercion is performed.

    Raises:
        ValueError: A required column is missing or has a different data type.
        FileNotFoundError: The file does not exist.
        OSError: The file cannot be opened or memory-mapped.
        pyarrow.ArrowInvalid: Arrow cannot read a valid IPC file schema.
    """
    with pa.memory_map(path, "r") as source:
        schema = pa.ipc.open_file(source).schema

    missing = sorted(set(required) - set(schema.names))
    if missing:
        raise ValueError(
            f"{os.path.basename(path)}: missing required columns: {missing}"
        )

    for name, expected_type in required.items():
        actual_type = schema.field(name).type
        if not actual_type.equals(expected_type):
            raise ValueError(
                f"{os.path.basename(path)}: column {name!r} has type "
                f"{actual_type}; expected {expected_type}"
            )


@dataclass(frozen=True)
class Connectome:
    """An immutable handle to the files in a [MaleCNS v1.0](https://male-cns.janelia.org/) dataset.

    Construct a handle with `from_directory` to validate file availability
    and required schemas. Calling the dataclass constructor directly assigns
    the supplied fields without validation.

    The handle stores paths and dataset identity, rather than loaded neuron
    tables, connection arrays, or neural-network parameters. Freezing the
    handle prevents field reassignment but does not protect the files on disk
    from modification.

    Attributes:
        directory: Absolute dataset directory with user-home expansion and
            symbolic links resolved when created through `from_directory`.
        dataset_id: Dataset identifier, set to `"male-cns:v1.0"` by the loader.
        annotations_path: Path to the neuron annotations Feather file.
        neurotransmitters_path: Path to the per-body neurotransmitter
            annotations Feather file.
        connections_path: Path to the directed connection-count Feather file.

    Examples:
        Load a locally downloaded dataset:

        >>> from flyx import Connectome
        >>> connectome = Connectome.from_directory("data/male-cns-v1.0")
        >>> connectome.dataset_id
        'male-cns:v1.0'
    """

    directory: str
    dataset_id: str
    annotations_path: str
    neurotransmitters_path: str
    connections_path: str

    @classmethod
    def from_directory(cls, directory: str | os.PathLike[str]) -> "Connectome":
        """Create a dataset handle after validating the local files.

        The directory must contain these MaleCNS v1.0 files:

        - `body-annotations-male-cns-v1.0-minconf-0.5.feather`
        - `body-neurotransmitters-male-cns-v1.0.feather`
        - `connectome-weights-male-cns-v1.0-minconf-0.5.feather`

        Required columns and Arrow types are:

        | File | Required columns |
        | --- | --- |
        | Annotations | `bodyId: int64`, `superclass: string`, `type: string` |
        | Neurotransmitters | `body: int64`, `consensus_nt: string` |
        | Connections | `body_pre: int64`, `body_post: int64`, `weight: int64` |

        Additional columns are accepted. Validation checks file availability
        and schemas without materializing the tables. It does not check row
        values, identifier uniqueness, connection endpoints, or data provenance.
        No files are downloaded or modified.

        Args:
            directory: Local dataset directory as a string or path-like
                object. Relative paths are resolved against the working
                directory, `~` is expanded, and symbolic links are resolved.

        Returns:
            A handle containing the resolved file paths and the dataset
            identifier `"male-cns:v1.0"`.

        Raises:
            NotADirectoryError: The resolved path is not an existing directory.
            FileNotFoundError: One or more required files are missing or cease
                to exist before their schemas can be read.
            ValueError: A required column is missing or its Arrow type does
                not match the expected type.
            OSError: A required file cannot be opened or memory-mapped.
            pyarrow.ArrowInvalid: A required file has an invalid IPC schema.

        Examples:
            Load a dataset using a `pathlib.Path`:

            >>> from pathlib import Path
            >>> from flyx import Connectome
            >>> connectome = Connectome.from_directory(Path("data/male-cns-v1.0"))
            >>> Path(connectome.connections_path).name
            'connectome-weights-male-cns-v1.0-minconf-0.5.feather'
        """
        root = os.path.realpath(os.path.expanduser(directory))

        if not os.path.isdir(root):
            raise NotADirectoryError(f"Connectome directory does not exist: {root}")

        annotations = os.path.join(
            root, "body-annotations-male-cns-v1.0-minconf-0.5.feather"
        )
        neurotransmitters = os.path.join(
            root, "body-neurotransmitters-male-cns-v1.0.feather"
        )
        connections = os.path.join(
            root, "connectome-weights-male-cns-v1.0-minconf-0.5.feather"
        )

        paths = (annotations, neurotransmitters, connections)
        missing = [os.path.basename(path) for path in paths if not os.path.isfile(path)]

        if missing:
            raise FileNotFoundError(
                f"Incomplete MaleCNS v1.0 dataset at {root}. "
                f"Missing files: {', '.join(missing)}"
            )

        _validate_schema(
            annotations,
            {
                "bodyId": pa.int64(),
                "superclass": pa.string(),
                "type": pa.string(),
            },
        )
        _validate_schema(
            neurotransmitters,
            {
                "body": pa.int64(),
                "consensus_nt": pa.string(),
            },
        )
        _validate_schema(
            connections,
            {
                "body_pre": pa.int64(),
                "body_post": pa.int64(),
                "weight": pa.int64(),
            },
        )

        return cls(
            directory=root,
            dataset_id="male-cns:v1.0",
            annotations_path=annotations,
            neurotransmitters_path=neurotransmitters,
            connections_path=connections,
        )
