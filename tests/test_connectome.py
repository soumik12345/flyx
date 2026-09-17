from pathlib import Path

import pyarrow as pa
import pyarrow.feather as feather
import pytest

from flyx.core import Connectome

FILES = {
    "annotations": "body-annotations-male-cns-v1.0-minconf-0.5.feather",
    "neurotransmitters": "body-neurotransmitters-male-cns-v1.0.feather",
    "connections": "connectome-weights-male-cns-v1.0-minconf-0.5.feather",
}

REQUIRED_COLUMNS = [
    ("annotations", "bodyId"),
    ("neurotransmitters", "consensus_nt"),
    ("connections", "weight"),
]


@pytest.fixture
def dataset_dir(tmp_path: Path) -> Path:
    tables = {
        "annotations": pa.table(
            {
                "bodyId": [1, 2],
                "superclass": ["ol_sensory", "ol_intrinsic"],
                "type": ["R7y", "L1"],
            }
        ),
        "neurotransmitters": pa.table(
            {
                "body": [1, 2],
                "consensus_nt": ["histamine", "glutamate"],
            }
        ),
        "connections": pa.table(
            {
                "body_pre": [1],
                "body_post": [2],
                "weight": [5],
            }
        ),
    }
    for name, table in tables.items():
        feather.write_feather(table, tmp_path / FILES[name])
    return tmp_path


@pytest.mark.parametrize("path_type", [str, Path])
def test_loads_valid_dataset(dataset_dir, path_type):
    connectome = Connectome.from_directory(path_type(dataset_dir))

    assert connectome.dataset_id == "male-cns:v1.0"
    assert connectome.directory == str(dataset_dir.resolve())
    for name, filename in FILES.items():
        assert getattr(connectome, f"{name}_path") == str(dataset_dir / filename)


@pytest.mark.parametrize("exists_as_file", [False, True])
def test_rejects_non_directory(tmp_path, exists_as_file):
    path = tmp_path / "not-a-directory"
    if exists_as_file:
        path.write_text("ordinary file")

    with pytest.raises(NotADirectoryError):
        Connectome.from_directory(path)


@pytest.mark.parametrize("name", FILES)
def test_reports_missing_file(dataset_dir, name):
    (dataset_dir / FILES[name]).unlink()

    with pytest.raises(FileNotFoundError) as error:
        Connectome.from_directory(dataset_dir)

    assert FILES[name] in str(error.value)


@pytest.mark.parametrize("name,column", REQUIRED_COLUMNS)
def test_rejects_missing_column(dataset_dir, name, column):
    path = dataset_dir / FILES[name]
    table = feather.read_table(path).drop([column])
    feather.write_feather(table, path)

    with pytest.raises(ValueError) as error:
        Connectome.from_directory(dataset_dir)

    assert FILES[name] in str(error.value)
    assert column in str(error.value)


@pytest.mark.parametrize("name,column", REQUIRED_COLUMNS)
def test_rejects_wrong_column_type(dataset_dir, name, column):
    path = dataset_dir / FILES[name]
    table = feather.read_table(path)
    wrong_values = pa.array([1] * table.num_rows, type=pa.int32())
    table = table.set_column(table.schema.get_field_index(column), column, wrong_values)
    feather.write_feather(table, path)

    with pytest.raises(ValueError) as error:
        Connectome.from_directory(dataset_dir)

    assert FILES[name] in str(error.value)
    assert column in str(error.value)


@pytest.mark.parametrize("name", FILES)
def test_rejects_corrupt_feather_file(dataset_dir, name):
    (dataset_dir / FILES[name]).write_bytes(b"not a Feather file")

    with pytest.raises(pa.ArrowInvalid):
        Connectome.from_directory(dataset_dir)


def test_allows_extra_annotation_columns(dataset_dir):
    path = dataset_dir / FILES["annotations"]
    table = feather.read_table(path).append_column(
        "notes", pa.array(["first", "second"])
    )
    feather.write_feather(table, path)

    connectome = Connectome.from_directory(dataset_dir)
    assert connectome.dataset_id == "male-cns:v1.0"
