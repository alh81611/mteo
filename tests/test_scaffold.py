import mteo


def test_version():
    assert mteo.__version__ == "0.1.0"


def test_exceptions_importable():
    assert issubclass(mteo.IdentificationError, ValueError)
    assert issubclass(mteo.WeakInstrumentWarning, UserWarning)
