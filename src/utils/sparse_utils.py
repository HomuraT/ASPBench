import numpy as np
from scipy.sparse import coo_matrix

class SparseUtilsError(Exception):
    """Custom exception for sparse utility errors."""
    pass

def dense_to_sparse_serializable(matrix: np.ndarray) -> dict:
    """
    Converts a dense NumPy array to a serializable dictionary representation
    of a sparse COO matrix.

    Args:
        matrix: The input dense NumPy array.

    Returns:
        A dictionary with 'data', 'row', 'col', and 'shape' keys,
        suitable for JSON serialization.

    Raises:
        TypeError: If the input is not a NumPy array.
    """
    if not isinstance(matrix, np.ndarray):
        raise TypeError(f"Input must be a NumPy array, got {type(matrix)}")

    sparse_coo = coo_matrix(matrix)
    return {
        'data': sparse_coo.data.tolist(),
        'row': sparse_coo.row.tolist(),
        'col': sparse_coo.col.tolist(),
        'shape': sparse_coo.shape
    }

def sparse_serializable_to_dense(sparse_dict: dict) -> np.ndarray:
    """
    Converts a serializable dictionary representation of a sparse COO matrix
    back to a dense NumPy array.

    Args:
        sparse_dict: A dictionary with 'data', 'row', 'col', and 'shape' keys.

    Returns:
        The reconstructed dense NumPy array.

    Raises:
        KeyError: If the dictionary is missing required keys.
        ValueError: If the dictionary values are inconsistent.
        SparseUtilsError: For other conversion issues.
    """
    required_keys = {'data', 'row', 'col', 'shape'}
    if not required_keys.issubset(sparse_dict.keys()):
        missing = required_keys - sparse_dict.keys()
        raise KeyError(f"Sparse dictionary is missing required keys: {missing}")

    try:
        data = sparse_dict['data']
        row = sparse_dict['row']
        col = sparse_dict['col']
        shape = tuple(sparse_dict['shape']) # Ensure shape is a tuple

        if not (len(data) == len(row) == len(col)):
             raise ValueError("Length mismatch between 'data', 'row', and 'col' arrays.")
        if not (isinstance(shape, tuple) and len(shape) == 2 and
                all(isinstance(dim, int) and dim >= 0 for dim in shape)):
            raise ValueError(f"Invalid shape '{shape}'. Must be a tuple of two non-negative integers.")

        # Basic validation for indices within shape bounds
        if row and (max(row) >= shape[0] or min(row) < 0):
            raise ValueError("Row indices are out of bounds for the given shape.")
        if col and (max(col) >= shape[1] or min(col) < 0):
            raise ValueError("Column indices are out of bounds for the given shape.")


        sparse_coo = coo_matrix((data, (row, col)), shape=shape)
        return sparse_coo.toarray()

    except ValueError as e: # Catch specific ValueErrors from coo_matrix or our checks
        raise ValueError(f"Error creating sparse matrix from dictionary: {e}")
    except Exception as e:
        raise SparseUtilsError(f"An unexpected error occurred during sparse to dense conversion: {e}")

# Example Usage (Optional)
if __name__ == '__main__':
    print("--- Testing Sparse Utils ---")

    # 1. Dense to Sparse
    dense_matrix = np.array([[1, 0, 0], [0, 0, 2], [3, 0, 0]])
    print("Original Dense Matrix:\n", dense_matrix)
    try:
        serializable_sparse = dense_to_sparse_serializable(dense_matrix)
        print("\nSerializable Sparse Dictionary:\n", serializable_sparse)

        # Check types for JSON compatibility
        assert isinstance(serializable_sparse['data'], list)
        assert isinstance(serializable_sparse['row'], list)
        assert isinstance(serializable_sparse['col'], list)
        assert isinstance(serializable_sparse['shape'], tuple)
        print("Type check for serialization: PASSED")

        # 2. Sparse to Dense
        print("\nConverting back to Dense...")
        reconstructed_dense = sparse_serializable_to_dense(serializable_sparse)
        print("Reconstructed Dense Matrix:\n", reconstructed_dense)

        assert np.array_equal(dense_matrix, reconstructed_dense)
        print("Reconstruction Test: PASSED")

    except (SparseUtilsError, TypeError, KeyError, ValueError) as e:
        print(f"Test FAILED: {e}")

    # Test Error Handling
    print("\n--- Testing Error Handling ---")
    invalid_dict_missing_key = {'data': [1], 'row': [0], 'shape': (1,1)}
    invalid_dict_bad_shape = {'data': [1], 'row': [0], 'col': [0], 'shape': (1,)}
    invalid_dict_mismatch = {'data': [1, 2], 'row': [0], 'col': [0], 'shape': (1,1)}
    invalid_dict_out_of_bounds = {'data': [1], 'row': [1], 'col': [0], 'shape': (1,1)}

    test_cases = [
        (invalid_dict_missing_key, KeyError),
        (invalid_dict_bad_shape, ValueError),
        (invalid_dict_mismatch, ValueError),
        (invalid_dict_out_of_bounds, ValueError),
    ]

    for sparse_dict, expected_error in test_cases:
        try:
            sparse_serializable_to_dense(sparse_dict)
            print(f"Error test FAILED for dict: {sparse_dict} - Expected {expected_error}")
        except expected_error as e:
            print(f"Caught expected {expected_error} for dict {list(sparse_dict.keys())}: PASSED ({e})")
        except Exception as e:
             print(f"Error test FAILED for dict: {sparse_dict} - Caught unexpected {type(e)}: {e}")

    try:
        dense_to_sparse_serializable([1, 2, 3]) # Pass a list instead of ndarray
    except TypeError as e:
        print(f"Caught expected TypeError for non-ndarray input: PASSED ({e})")
    except Exception as e:
        print(f"Error test FAILED for non-ndarray input - Caught unexpected {type(e)}: {e}")

    print("\nSparse Utils tests completed.")
