import numpy as np


def allocate_resources(z, R):
    """
    Convert continuous outputs to integer allocations summing to R.

    Parameters:
    z (np.array): Raw outputs from the neural network (positive real numbers).
    R (int): Total resources to allocate.

    Returns:
    np.array: Integer allocations summing to R.
    """
    # Normalize to get probabilities
    p = z / np.sum(z)
    # Scale by R
    r = R * p
    # Floor the allocations
    x = np.floor(r).astype(int)
    # Compute the remainder
    remainder = r - x
    # Distribute the remaining resources
    deficit = R - np.sum(x)
    # Get indices sorted by descending remainder
    indices = np.argsort(-remainder)
    # print("Indices sorted by remainder:", indices)
    for i in range(deficit):
        x[indices[i]] += 1
    return x

def bounded_hare_niemeyer(q, R, M_upper):
    x = np.minimum(np.floor(q), M_upper).astype(int)
    r = R - np.sum(x)
    remainders = q - np.floor(q)
    remainders[x >= M_upper] = -np.inf  # exclude saturated
    while r > 0:
        i = np.argmax(remainders)
        if x[i] < M_upper[i]:
            x[i] += 1
            r -= 1
            if x[i] == M_upper[i]:
                remainders[i] = -np.inf
        else:
            break  # No more eligible regions
    return x

if __name__ == "__main__":
    # Neural network outputs
    z = np.array([0.1, 0.3, 0.4])
    R = 10
    allocations = allocate_resources(z, R)
    print(allocations)  # Output: array of integers summing to 10
    # Example usage of bounded_hare_niemeyer
    q = np.array([0.1, 0.3, 0.4])
    R = 10
    M_upper = np.array([5, 3, 5])
    allocations = bounded_hare_niemeyer(q, R, M_upper)
    print(allocations)  # Output: array of integers summing to 10
