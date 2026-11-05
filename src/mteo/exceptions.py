class IdentificationError(ValueError):
    """Raised when a requested quantity is not identified from the data/assumptions."""


class WeakInstrumentWarning(UserWarning):
    """Emitted when the take-up gap p_I - p_C is too thin to support extrapolation."""
