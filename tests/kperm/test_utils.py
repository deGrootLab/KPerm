from kperm.utils import _create_logger, _write_list_of_tuples


def test_create_logger(tmp_path):
    d = tmp_path / "logger"
    d.mkdir()
    p = d / "kperm.log"

    _create_logger(p)


def test_write_list_of_tuples(tmp_path):
    d = tmp_path / "write_tuples"
    d.mkdir()
    p = d / "test.dat"
    _write_list_of_tuples(p, [("23", "213")])
