#!/usr/bin/env python3
"""Verify quaternion-to-rotation implementation."""

import numpy as np
import sys

# Reference QUAT_TO_ROT tensor
QUAT_TO_ROT = np.zeros((4, 4, 3, 3), dtype=np.float32)
QUAT_TO_ROT[0, 0] = [[ 1, 0, 0], [ 0, 1, 0], [ 0, 0, 1]]
QUAT_TO_ROT[1, 1] = [[ 1, 0, 0], [ 0,-1, 0], [ 0, 0,-1]]
QUAT_TO_ROT[2, 2] = [[-1, 0, 0], [ 0, 1, 0], [ 0, 0,-1]]
QUAT_TO_ROT[3, 3] = [[-1, 0, 0], [ 0,-1, 0], [ 0, 0, 1]]
QUAT_TO_ROT[1, 2] = [[ 0, 2, 0], [ 2, 0, 0], [ 0, 0, 0]]
QUAT_TO_ROT[1, 3] = [[ 0, 0, 2], [ 0, 0, 0], [ 2, 0, 0]]
QUAT_TO_ROT[2, 3] = [[ 0, 0, 0], [ 0, 0, 2], [ 0, 2, 0]]
QUAT_TO_ROT[0, 1] = [[ 0, 0, 0], [ 0, 0,-2], [ 0, 2, 0]]
QUAT_TO_ROT[0, 2] = [[ 0, 0, 2], [ 0, 0, 0], [-2, 0, 0]]
QUAT_TO_ROT[0, 3] = [[ 0,-2, 0], [ 2, 0, 0], [ 0, 0, 0]]

def quat_to_rot_reference(q):
    """Reference implementation using tensor multiplication."""
    # q = [w, x, y, z]
    rot_tensor = np.sum(
        np.reshape(QUAT_TO_ROT, (4, 4, 9))
        * q[:, None, None]
        * q[None, :, None],
        axis=(0, 1),
    )
    return rot_tensor.reshape(3, 3)

def quat_to_rot_rust_formula(q):
    """Rust formula (from structure_module.rs)."""
    w, x, y, z = q
    return np.array([
        [
            1. - 2. * (y * y + z * z),
            2. * (x * y - w * z),
            2. * (x * z + w * y),
        ],
        [
            2. * (x * y + w * z),
            1. - 2. * (x * x + z * z),
            2. * (y * z - w * x),
        ],
        [
            2. * (x * z - w * y),
            2. * (y * z + w * x),
            1. - 2. * (x * x + y * y),
        ],
    ])

def test_quaternion_formula():
    """Test various quaternions."""
    test_cases = [
        # Identity: [1, 0, 0, 0]
        np.array([1.0, 0.0, 0.0, 0.0]),
        # 180° rotation around Z
        np.array([0.0, 0.0, 0.0, 1.0]),
        # Random
        np.array([0.5, 0.5, 0.5, 0.5]),
        # Small rotation (like delta frames)
        np.array([0.99, 0.05, 0.02, 0.01]) / np.linalg.norm([0.99, 0.05, 0.02, 0.01]),
    ]
    
    print("Comparing quaternion-to-rotation implementations:")
    print("=" * 70)
    
    max_error = 0
    for q in test_cases:
        ref = quat_to_rot_reference(q)
        rust = quat_to_rot_rust_formula(q)
        
        error = np.max(np.abs(ref - rust))
        max_error = max(max_error, error)
        
        print(f"\nQuaternion: {q}")
        print(f"Reference:\n{ref}")
        print(f"Rust formula:\n{rust}")
        print(f"Max element-wise difference: {error:.2e}")
        
        # Check if rotation matrix is valid (orthogonal, det=1)
        det_ref = np.linalg.det(ref)
        det_rust = np.linalg.det(rust)
        print(f"  Reference det={det_ref:.6f}, Rust det={det_rust:.6f}")
    
    print("\n" + "=" * 70)
    if max_error < 1e-5:
        print(f"✓ PASS: Implementations match (max error: {max_error:.2e})")
        return True
    else:
        print(f"✗ FAIL: Implementations differ (max error: {max_error:.2e})")
        return False

if __name__ == '__main__':
    success = test_quaternion_formula()
    sys.exit(0 if success else 1)
