"""Synthetic Fourier undersampling: basis, sampling geometry and faint features."""

import argparse
import gzip
import json
import time
import zipfile

import numpy as np

from .arrays import ArrayWriter
from .core import relative
from .study import RELEASE, ROOT, source_files, write


def haar_matrix(n):
    if n < 2 or n & (n-1):
        raise ValueError("Haar dimension must be a power of two")
    result = np.eye(n)
    length = n
    while length > 1:
        before = result[:length].copy()
        result[:length//2] = (before[::2]+before[1::2])/np.sqrt(2)
        result[length//2:length] = (before[::2]-before[1::2])/np.sqrt(2)
        length //= 2
    return result


def analysis(image, basis, h):
    return h@image@h.T if basis == "haar" else image.copy()


def synthesis(coefficients, basis, h):
    return h.T@coefficients@h if basis == "haar" else coefficients.copy()


def soft(x, threshold):
    return np.sign(x)*np.maximum(abs(x)-threshold, 0)


def recover(measured, mask, basis, noise_sigma, iterations=400):
    """Real-image FISTA for .5||M F x-y||² + lambda ||W x||1.

    Unitary Fourier and orthonormal product-Haar make Lipschitz bound 1 valid.
    Noise scale is supplied sensor metadata; no true image enters this solver.
    """
    started = time.perf_counter()
    n = mask.shape[0]
    h = haar_matrix(n)
    back = np.fft.ifft2(measured, norm="ortho").real
    scale = max(float(np.linalg.norm(measured))/n, 1e-6)
    penalty = max(1e-4*scale, noise_sigma*np.sqrt(2*np.log(n*n)))
    coefficients = analysis(back, basis, h)
    extrapolated, momentum, trace = coefficients.copy(), 1., []
    initial_penalty = max(penalty, .1*float(abs(coefficients).max()))
    for step in range(iterations):
        stage = min(4, 5*step//iterations)
        active_penalty = penalty if stage == 4 else max(penalty, initial_penalty*10**(-stage))
        if step and stage != min(4, 5*(step-1)//iterations):
            # Restart acceleration when the continuation objective changes.
            extrapolated, momentum = coefficients.copy(), 1.
        current = synthesis(extrapolated, basis, h)
        residual = mask*np.fft.fft2(current, norm="ortho")-measured
        gradient = analysis(np.fft.ifft2(residual, norm="ortho").real, basis, h)
        updated = soft(extrapolated-gradient, active_penalty)
        next_momentum = (1+np.sqrt(1+4*momentum**2))/2
        extrapolated = updated+(momentum-1)/next_momentum*(updated-coefficients)
        coefficients, momentum = updated, next_momentum
        if step % 25 == 0 or step+1 == iterations:
            image = synthesis(coefficients, basis, h)
            objective = .5*np.linalg.norm(mask*np.fft.fft2(image, norm="ortho")-measured)**2+active_penalty*abs(coefficients).sum()
            trace.append([step+1, float(objective), active_penalty])
    reconstructed = synthesis(coefficients, basis, h)
    residual = mask*np.fft.fft2(reconstructed, norm="ortho")-measured
    gradient = analysis(np.fft.ifft2(residual, norm="ortho").real, basis, h)
    stationarity = float(np.linalg.norm(coefficients-soft(coefficients-gradient, penalty))/max(np.linalg.norm(coefficients), 1e-12))
    return reconstructed, {"iterations": iterations, "penalty": penalty, "objective_trace": trace,
                           "proximal_gradient_relative_norm": stationarity,
                           "measured_relative_residual": float(np.linalg.norm(residual)/max(np.linalg.norm(measured), 1e-12)),
                           "fft_calls": 2*iterations+3+len(trace), "worker_seconds": time.perf_counter()-started}


def sampling_mask(n, count, geometry, seed):
    rng = np.random.default_rng(seed)
    frequency = np.fft.fftfreq(n)*n
    mask = np.zeros((n, n), bool)
    if geometry in ("cartesian_variable", "cartesian_regular"):
        lines = count//n
        if count % n or n % lines:
            raise ValueError("Declared Cartesian setting needs integral line count")
        if geometry == "cartesian_regular":
            chosen = np.arange(0, n, n//lines)
        else:
            candidates = np.arange(1, n)
            probability = 1/(.5+abs(frequency[candidates]))**1.5
            chosen = np.r_[0, rng.choice(candidates, lines-1, replace=False, p=probability/probability.sum())]
        mask[chosen, :] = True
    else:
        candidates = np.arange(1, n*n)
        probability = None
        if geometry == "points_variable":
            radius = np.sqrt(frequency[:, None]**2+frequency[None, :]**2).ravel()[1:]
            probability = 1/(.5+radius)**1.5
            probability /= probability.sum()
        elif geometry != "points_uniform":
            raise ValueError("Unknown sampling geometry")
        mask.ravel()[np.r_[0, rng.choice(candidates, count-1, replace=False, p=probability)]] = True
    assert mask.sum() == count
    return mask


def phantom(seed, family, n=32):
    rng = np.random.default_rng(seed)
    h, roi = haar_matrix(n), np.zeros((n, n), bool)
    if family == "sparse_pixels":
        image = np.zeros((n, n))
        image.ravel()[rng.choice(n*n, 12, replace=False)] = rng.uniform(.4, 1., 12)
    elif family == "sparse_haar":
        coefficients = np.zeros((n, n))
        coefficients.ravel()[rng.choice(n*n, 12, replace=False)] = rng.normal(size=12)
        image = h.T@coefficients@h
    elif family == "phantom_faint":
        yy, xx = np.mgrid[-1:1:complex(n), -1:1:complex(n)]
        dx, dy = rng.uniform(-.08, .08, 2)
        image = .7*((xx-dx)**2/.7**2+(yy-dy)**2/.85**2 < 1)
        image -= .25*((xx-dx+.18)**2/.2**2+(yy-dy)**2/.45**2 < 1)
        image += .4*((xx-dx-.12)**2/.12**2+(yy-dy+.35)**2/.15**2 < 1)
        center = np.array([19, 21])+rng.integers(-1, 2, 2)
        roi[center[0]:center[0]+2, center[1]:center[1]+2] = True
        image[roi] += .04
    elif family == "dense_texture":
        image = rng.normal(scale=.3, size=(n, n))
    else:
        raise ValueError("Unknown synthetic family")
    return image, roi


def run(args):
    folder = RELEASE/args.name
    folder.mkdir(exist_ok=False)
    protocol = {"id": args.name, "config": vars(args), "sources": source_files(),
                "families": ["sparse_pixels", "sparse_haar", "phantom_faint", "dense_texture"],
                "geometry": ["points_uniform", "points_variable", "cartesian_variable", "cartesian_regular"],
                "complex_measurements": [128, 256], "image_shape": [32, 32], "noise_sigma_per_real_component": [0., .001],
                "methods": ["zero_filled", "pixel", "haar"], "iterations": 400,
                "regularizer": "Final max(1e-4*||measured||2/32, sigma*sqrt(2*log(1024))). Five equal-length continuation stages from .1*max|W F* y|, divided by 10 each stage, floored at final lambda; final stage uses final lambda. Restart acceleration at each transition. Changed after underconverged first pilot, before final evaluation.",
                "primary": "Global NRMSE plus error within the known 2x2 faint-feature ROI, separately reported. Every method gets identical k-space values and mask.",
                "basis": "Pixel or separable orthonormal one-dimensional Haar product basis; not a reproduction of wavelet+TV SparseMRI implementation.",
                "cost": "Each complex observation costs two real scalars; a synthetic Fourier oracle constructs full k-space once, counted separately. No assumption that arbitrary SERA trajectories can be projected for free.",
                "boundary": "Synthetic, single-coil, real image, ideal Fourier model only. No patient data, clinical evaluation, coil sensitivity, phase/motion/trajectory timing or gradient/slew constraints. Random points are idealized; Cartesian lines are a limited acquisition proxy.",
                "decision": "Report matched and mismatched bases, geometry, residuals and stationarity. No live-owner or medical deployment from this screening study."}
    write(folder/"protocol.json", protocol)
    with zipfile.ZipFile(folder/"sources.zip", "x", compression=zipfile.ZIP_DEFLATED) as archive:
        for name in protocol["sources"]:
            archive.write(ROOT/name, name)
    arrays = ArrayWriter(folder/"arrays.zip")
    started, rows = time.perf_counter(), []
    with gzip.open(folder/"records.jsonl.gz", "wt", encoding="utf-8") as stream:
        for index in range(args.seeds):
            for family_index, family in enumerate(protocol["families"]):
                seed = args.start_seed+index*101+family_index*10007
                image, roi = phantom(seed, family)
                actual_kspace = np.fft.fft2(image, norm="ortho")
                original = arrays.put(image)
                region = arrays.put(roi)
                for count in protocol["complex_measurements"]:
                    for geometry in protocol["geometry"]:
                        mask = sampling_mask(32, count, geometry, seed+500003)
                        for noise in protocol["noise_sigma_per_real_component"]:
                            rng = np.random.default_rng(seed+700001)
                            measured = mask*(actual_kspace+noise*(rng.normal(size=(32, 32))+1j*rng.normal(size=(32, 32))))
                            measurement, mask_reference = arrays.put(measured), arrays.put(mask)
                            for method in protocol["methods"]:
                                if method == "zero_filled":
                                    start = time.perf_counter()
                                    reconstructed = np.fft.ifft2(measured, norm="ortho").real
                                    work = {"fft_calls": 1, "worker_seconds": time.perf_counter()-start}
                                else:
                                    reconstructed, work = recover(measured, mask, method, noise)
                                row = {"seed": seed, "family": family, "count": count, "geometry": geometry,
                                       "noise_sigma": noise, "method": method, "image": original, "roi": region,
                                       "measured": measurement, "mask": mask_reference, "reconstructed": arrays.put(reconstructed),
                                       "relative_error": relative(reconstructed, image),
                                       "roi_absolute_error": float(np.mean(abs(reconstructed[roi]-image[roi]))) if roi.any() else None,
                                       "faint_amplitude": .04 if roi.any() else None,
                                       "observed_scalar_cost": 2*count, "work": work}
                                stream.write(json.dumps(row, separators=(",", ":"))+"\n")
                                rows.append({k: row[k] for k in ("seed", "family", "count", "geometry", "noise_sigma", "method", "relative_error", "roi_absolute_error", "work")})
                stream.flush()
                write(folder/"progress.json", {"records": len(rows), "last_seed": seed})
    arrays.close()
    write(folder/"summary.json", {"status": "COMPLETE", "records": len(rows), "full_oracle_ffts": args.seeds*4,
                                  "worker_seconds": time.perf_counter()-started, "rows": rows})
    print(json.dumps({"status": "COMPLETE", "records": len(rows), "worker_seconds": time.perf_counter()-started}))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", required=True)
    parser.add_argument("--seeds", type=int, required=True)
    parser.add_argument("--start-seed", type=int, required=True)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
