"""Utilities for computing file hashes."""

import hashlib


def compute_hash(file_path: str, hash_algorithm: str = "sha256") -> str:
    """Compute the hash of a file.

    Parameters
    ----------
    file_path : str
        The path to the file to hash
    hash_algorithm : str, optional
        The hash algorithm to use, by default "sha256"

    Returns
    -------
    str
        The hexadecimal hash of the file
    """

    hash_func = hashlib.new(hash_algorithm)
    with open(file_path, "rb") as f:
        while chunk := f.read(4096):
            hash_func.update(chunk)

    return hash_func.hexdigest()
