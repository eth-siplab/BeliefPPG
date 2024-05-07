class InputConfig:
    """
    Configuration for input to network.

    This class defines the parameters for windowing the input data sequences. It specifies the
    window size and the stride between consecutive windows.
    """
    WINSIZE = 8  # Length of the input window in seconds.
    STRIDE = 2   # Stride between consecutive windows in seconds.