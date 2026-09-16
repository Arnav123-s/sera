import numpy as np

from experiments.sparse_mechanisms.arrays import ArrayReader, ArrayWriter
from experiments.sparse_mechanisms.imaging import haar_matrix, recover, sampling_mask
from experiments.sparse_mechanisms.variants import forward, repair_dense, sketch_operator


def test_haar_fourier_adjoint_and_energy():
    rng = np.random.default_rng(117)
    h = haar_matrix(32)
    x, y = rng.normal(size=(2, 32, 32))
    np.testing.assert_allclose(h.T@h, np.eye(32), atol=1e-14)
    np.testing.assert_allclose(h.T@(h@x@h.T)@h, x, atol=1e-14)
    mask = sampling_mask(32, 128, "points_variable", 19)
    ax = mask*np.fft.fft2(h.T@x@h, norm="ortho")
    z = y+1j*rng.normal(size=(32, 32))
    adjoint = h@np.fft.ifft2(mask*z, norm="ortho").real@h.T
    np.testing.assert_allclose(np.vdot(ax, z).real, np.vdot(x, adjoint).real, atol=1e-12)


def test_full_measurements_recover_image_and_masks_pay_exact_counts():
    rng = np.random.default_rng(111)
    image = rng.normal(size=(8, 8))
    reconstructed, work = recover(np.fft.fft2(image, norm="ortho"), np.ones((8, 8), bool), "haar", 0, 10)
    assert np.linalg.norm(reconstructed-image)/np.linalg.norm(image) < .001
    assert work["proximal_gradient_relative_norm"] < 1e-12
    for geometry in ("points_uniform", "points_variable", "cartesian_regular", "cartesian_variable"):
        mask = sampling_mask(32, 128, geometry, 111)
        assert mask.sum() == 128 and mask[0, 0]
        if geometry.startswith("cartesian"):
            assert np.all((mask.sum(axis=1) == 0) | (mask.sum(axis=1) == 32))


def test_unknown_sparse_corruption_repairs_dense_payload():
    rng = np.random.default_rng(219)
    a = sketch_operator(121, 64, 32)
    payload = rng.normal(size=32)
    received = a@payload
    received[[1, 17, 42]] += [4., -3., 2.]
    decoded, error, _ = repair_dense(a, received)
    np.testing.assert_allclose(decoded, payload, atol=1e-7)
    np.testing.assert_allclose(a@decoded+error, received, atol=1e-7)
    assert np.linalg.norm(np.linalg.lstsq(a, received, rcond=None)[0]-payload) > .1


def test_array_restore_and_executable_weight_update(tmp_path):
    writer = ArrayWriter(tmp_path/"arrays.zip")
    rng = np.random.default_rng(71)
    hidden, x = rng.normal(size=(3, 8)), rng.normal(size=(13, 3))
    parent, delta = rng.normal(size=(2, 8))
    reference = writer.put(parent+delta)
    assert writer.put(parent+delta) == reference and len(writer.known) == 1
    writer.close()
    reader = ArrayReader(tmp_path/"arrays.zip")
    restored = reader.get(reference)
    reader.close()
    np.testing.assert_allclose(forward(x, hidden, restored, "tanh"), forward(x, hidden, parent, "tanh")+np.tanh(x@hidden)@delta, atol=1e-13)
