import os


def verify_path(path: str, is_dir: bool = False) -> str:
    """Verify that the given path exists and is of the expected type (file or directory).

    :param str path: The path to verify
    :param bool is_dir: (optional) If True, checks that the path is a directory. If False, checks
        that the path is a file

    :return: The verified path if it exists and matches the expected type
    :rtype: str

    :raises FileNotFoundError: If the path does not exist or does not match the expected type
    """

    if is_dir:
        if not os.path.isdir(path):  # Verify if path truly leads to directory
            err_msg = f"""
            Path must be a directory!\n
            `{path}` leads to a file.
            """
            raise FileNotFoundError(err_msg)
        else:
            return path
    else:
        if not os.path.isfile(path):  # Verify if path truly leads to file
            err_msg = f"""
            Path must lead to file!\n 
            `{path}` leads to a directory.
            """

            raise FileNotFoundError(err_msg)
        else:
            return path
