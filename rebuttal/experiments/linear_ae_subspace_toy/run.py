#!/usr/bin/env python3
"""R2-W1 A/D: tied linear AE counterexample and angular projector analysis.

For a = w.T @ x and b = w.T @ w,
    L(w; x) = x.T @ x + (b - 2) * a**2,
    grad_w L = 2 * a * (b - 2) * x + 2 * a**2 * w.
The stipulated labels do not follow uniquely from the coordinates. Probes
never enter the update. The normalized projector is a scoring-only control
using the same two parameter states, not another optimization procedure.
Runs on CPU, writes only to stdout, and requires NumPy and PyTorch.
An optional --geometry-output CSV saves the Experiment D numerical checks.
--controlled-shift-output runs the paired distribution-level rotation test.
"""

import argparse
import csv
import json
from pathlib import Path
import numpy as np
import torch


def analytical_gradient(w, x):
    a, b = w @ x, w @ w
    return 2 * a * (b - 2) * x + 2 * a**2 * w


def controlled_shift(output):
    """Prespecified rotation experiment; scalar SGD on an orthogonal projector.

    Source normal: random sign * r*[cos(e), sin(e)], r~U[.8,1.2],
    e~U[-10,10] degrees. Target normals are the same law rotated +30 degrees.
    Anomalies: r~U[1.2,1.6], e~U[-2,2], angle +(30+e) (harmful candidate)
    or -(30+e) (favorable candidate). Reflection gives identical Frozen
    scores. Labels are stipulated; harmful anomalies overlap normal support
    in direction and are more concentrated around the target principal line.
    Source initialization is the exact population PCA direction, not a fit
    to target or anomalous observations. No distribution/hyperparameter search.
    """
    from sklearn.metrics import auc, precision_recall_curve

    out = Path(output)
    out.mkdir(parents=True, exist_ok=False)
    design = dict(seeds=list(range(42, 47)), shift_degrees=30, normal_radius=[.8, 1.2],
                  normal_halfwidth_degrees=10, anomaly_radius=[1.2, 1.6],
                  anomaly_halfwidth_degrees=2, audit_points=512, update_points=128,
                  evaluation_points_per_class=512, learning_rate=.02,
                  initialization="exact source population principal direction phi=0",
                  conditions={"favorable": "angle=-(30+epsilon)", "harmful": "angle=+(30+epsilon)"},
                  optimizer="CPU float64 SGD on phi; momentum=0; weight_decay=0",
                  metric="sklearn PR-curve trapezoidal area; strict pair ordering",
                  protocol="one paired normal-only trajectory/seed; probes never enter updates")
    (out / "design.json").write_text(json.dumps(design, indent=2) + "\n")

    def score(points, phi):
        v = np.array([np.cos(phi), np.sin(phi)])
        return np.sum((points - (points @ v)[:, None] * v)**2, axis=1)

    def torque(points, phi):
        # r^2*sin(2*(alpha-phi)), evaluated without polar-angle rounding.
        v, perpendicular = np.array([np.cos(phi), np.sin(phi)]), np.array([-np.sin(phi), np.cos(phi)])
        return 2 * (points @ v) * (points @ perpendicular)

    def metrics(normal, anomaly, phi):
        n, a = score(normal, phi), score(anomaly, phi)
        precision, recall, _ = precision_recall_curve(np.r_[np.zeros(len(n)), np.ones(len(a))], np.r_[n, a])
        return dict(normal_score=n.mean(), anomaly_score=a.mean(), margin=a.mean()-n.mean(),
                    pair_fraction=float((a[:, None] > n[None, :]).mean()), auprc=float(auc(recall, precision)))

    def draw(rng, n, radii, halfwidth, center):
        radius = rng.uniform(*radii, n)
        angles = np.deg2rad(center + rng.uniform(-halfwidth, halfwidth, n))
        signs = rng.choice([-1., 1.], n)
        return (radius * signs)[:, None] * np.column_stack([np.cos(angles), np.sin(angles)])

    shift = np.deg2rad(30)
    rotation = np.array([[np.cos(shift), -np.sin(shift)], [np.sin(shift), np.cos(shift)]])
    radial_moment = (.8**2 + .8*1.2 + 1.2**2) / 3
    angular_moment = np.sin(2*np.deg2rad(10)) / (2*np.deg2rad(10))
    source_cov = np.diag([radial_moment*(1+angular_moment)/2, radial_moment*(1-angular_moment)/2])
    target_cov = rotation @ source_cov @ rotation.T
    assert source_cov[0, 0] > source_cov[1, 1]
    design.update(source_population_covariance=source_cov.tolist(), target_population_covariance=target_cov.tolist())
    (out / "design.json").write_text(json.dumps(design, indent=2) + "\n")
    rows, local, audits = [], [], []
    for seed in design["seeds"]:
        rng = np.random.default_rng(seed)
        source = draw(rng, 512, (.8, 1.2), 10, 0)
        shifted = source @ rotation.T
        source_empirical, target_empirical = source.T @ source / len(source), shifted.T @ shifted / len(shifted)
        np.testing.assert_allclose(target_empirical, rotation @ source_empirical @ rotation.T, rtol=0, atol=1e-12)
        principal = np.linalg.eigh(target_empirical)[1][:, -1]
        source_principal = np.linalg.eigh(source_empirical)[1][:, -1]
        measured_shift = np.degrees(np.arccos(np.clip(abs(principal @ source_principal), 0, 1)))
        np.testing.assert_allclose(measured_shift, 30, rtol=0, atol=1e-10)
        audits.append(dict(seed=seed, principal_shift_degrees=measured_shift,
                           source_frozen_loss=score(source, 0).mean(), target_frozen_loss=score(shifted, 0).mean()))
        updates = draw(rng, 128, (.8, 1.2), 10, 30)
        normal = draw(rng, 512, (.8, 1.2), 10, 30)
        harmful = draw(rng, 512, (1.2, 1.6), 2, 30)
        favorable = harmful * np.array([1., -1.])
        anomalies = dict(favorable=favorable, harmful=harmful)
        np.testing.assert_array_equal(score(favorable, 0), score(harmful, 0))
        frozen = {k: metrics(normal, a, 0) for k, a in anomalies.items()}
        phi = torch.tensor(0., dtype=torch.float64, requires_grad=True)
        optimizer = torch.optim.SGD([phi], lr=.02, momentum=0., weight_decay=0.)
        update_loss_before = score(updates, 0).mean()
        for step, point in enumerate(updates):
            old = phi.item()
            u = torque(point[None, :], old)[0]
            x = torch.tensor(point, dtype=torch.float64)
            direction = torch.stack([phi.cos(), phi.sin()])
            loss = (x - direction * (direction @ x)).square().sum()
            optimizer.zero_grad()
            loss.backward()
            np.testing.assert_allclose(phi.grad.item(), -u, rtol=0, atol=1e-12)
            optimizer.step()
            new = phi.item()
            np.testing.assert_allclose(new, old + .02*u, rtol=0, atol=1e-12)
            normal_update_delta = score(point[None, :], new)[0] - loss.item()
            for name, anomaly in anomalies.items():
                g = torque(anomaly, old).mean() - torque(normal, old).mean()
                before_margin = score(anomaly, old).mean() - score(normal, old).mean()
                after_margin = score(anomaly, new).mean() - score(normal, new).mean()
                predicted, actual = -.02*u*g, after_margin-before_margin
                bound = ((anomaly**2).sum(axis=1).mean() + (normal**2).sum(axis=1).mean()) * (new-old)**2
                assert abs(actual-predicted) <= bound + 1e-12
                local.append(dict(seed=seed, condition=name, step=step, phi_before=old, phi_after=new,
                                  U=u, G=g, UG=u*g, normal_update_delta=normal_update_delta,
                                  margin_predicted=predicted, margin_actual=actual, remainder_bound=bound))
        for name, anomaly in anomalies.items():
            final = metrics(normal, anomaly, phi.item())
            rows.append(dict(seed=seed, condition=name, final_direction_degrees=np.degrees(phi.item()),
                             normal_objective_before=update_loss_before, normal_objective_after=score(updates, phi.item()).mean(),
                             frozen_headroom_to_one=1-frozen[name]["auprc"],
                             **{f"frozen_{k}": v for k, v in frozen[name].items()},
                             **{f"oracle_{k}": v for k, v in final.items()},
                             **{f"delta_{k}": final[k]-frozen[name][k] for k in final}))
    for filename, data in (("per_seed.csv", rows), ("local_updates.csv", local), ("shift_audit.csv", audits)):
        with (out / filename).open("x", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(data[0]))
            writer.writeheader()
            writer.writerows(data)
    for name in anomalies:
        subset = [r for r in rows if r["condition"] == name]
        print(name, json.dumps({k: float(np.mean([r[k] for r in subset])) for k in subset[0] if k not in ("condition", "seed")}))
    print("PASS: five paired trajectories; 30-degree shift, Frozen reflection parity, angular gradients and Taylor bounds verified.")


def geometric_analysis():
    """SGD in the angle of a unit-norm projector, not unconstrained w-SGD.

    U = r_u^2 sin(2(alpha_u-phi)); G = r_A^2 sin(2(alpha_A-phi))
                                           - r_N^2 sin(2(alpha_N-phi)).
    grad_phi L_u = -U; phi_new = phi + eta*U (exact scalar SGD).
    Delta L_u = -eta*U^2 + O(eta^2); Delta d = -eta*U*G + O(eta^2).
    U*G > 0 damages separation for sufficiently small positive eta, while
    U != 0 ensures a first-order normal-loss decrease. U*G = 0 is inconclusive.
    Taylor remainder bounds: r_u^2*Delta_phi^2 and (r_A^2+r_N^2)*Delta_phi^2.
    These distinguish a first-order condition from a finite-step theorem.
    """
    scenarios = (
        ("damaging", [1, 1], [.2, 1], [1, 1.1]),
        ("improving", [1, 1], [1, 1.1], [.2, 1]),
        ("first_order_boundary", [1, 1], [1, .5], [.5, 1]),
        ("stationary", [1, 0], [.2, 1], [1, 1.1]),
    )
    records = []
    print("\nD: orthogonal projector with scalar angular SGD (radians)")
    for name, update, normal, anomaly in scenarios:
        points = np.asarray([update, normal, anomaly], dtype=np.float64)
        radii2 = (points**2).sum(axis=1)
        angles = np.arctan2(points[:, 1], points[:, 0])
        phi = 0.0
        torque = radii2 * np.sin(2 * (angles - phi))
        u, g = torque[0], torque[2] - torque[1]
        angular = torch.tensor(phi, dtype=torch.float64, requires_grad=True)
        direction = torch.stack([angular.cos(), angular.sin()])
        x = torch.tensor(points, dtype=torch.float64)
        reconstructed = (x @ direction)[:, None] * direction[None, :]
        scores = (x - reconstructed).square().sum(dim=1)
        auto_u = torch.autograd.grad(scores[0], angular, retain_graph=True)[0].item()
        auto_d = torch.autograd.grad(scores[2] - scores[1], angular)[0].item()
        np.testing.assert_allclose([auto_u, auto_d], [-u, -g], rtol=0, atol=1e-12)
        before = scores.detach().numpy()
        np.testing.assert_allclose(before, radii2 * np.sin(angles - phi)**2, rtol=0, atol=1e-12)
        for eta in (1e-2, 1e-3, 1e-4):
            delta_phi = eta * u
            new_phi = phi - eta * auto_u
            np.testing.assert_allclose(new_phi, phi + delta_phi, rtol=0, atol=1e-12)
            direction_new = np.array([np.cos(new_phi), np.sin(new_phi)])
            after = ((points - (points @ direction_new)[:, None] * direction_new)**2).sum(axis=1)
            change = after - before
            normal_predicted, margin_predicted = -eta * u**2, -eta * u * g
            margin_actual = change[2] - change[1]
            normal_bound = radii2[0] * delta_phi**2
            margin_bound = (radii2[1] + radii2[2]) * delta_phi**2
            assert abs(change[0] - normal_predicted) <= normal_bound + 1e-12
            assert abs(margin_actual - margin_predicted) <= margin_bound + 1e-12
            if abs(margin_predicted) > margin_bound + 1e-12:
                assert np.sign(margin_actual) == np.sign(margin_predicted)
            if u != 0:
                assert change[0] < 0  # These tested eta satisfy eta < 1/r_u^2.
            row = dict(scenario=name, eta=eta, phi=phi, U=u, G=g, delta_phi=delta_phi,
                       normal_gradient=auto_u, margin_gradient=auto_d,
                       normal_delta_predicted=normal_predicted, normal_delta_actual=change[0],
                       anomaly_delta_actual=change[2], margin_delta_predicted=margin_predicted,
                       margin_delta_actual=margin_actual, margin_remainder=margin_actual-margin_predicted,
                       normal_remainder_bound=normal_bound, margin_remainder_bound=margin_bound)
            records.append(row)
            print(f"{name:22} eta={eta:g} U*G={u*g:+.6g} "
                  f"DeltaL={change[0]:+.9g} DeltaD(pred)={margin_predicted:+.9g} "
                  f"DeltaD(exact)={margin_actual:+.9g}")
    print("PASS: angular autograd, exact updates, and Taylor bounds for 12 checks.")
    return records


def main():
    w0 = np.array([1.0, 0.0], dtype=np.float64)
    x_u = np.array([1.0, 1.0], dtype=np.float64)
    eta = 0.1
    w = torch.tensor(w0, dtype=torch.float64, device="cpu", requires_grad=True)
    update = torch.tensor(x_u, dtype=torch.float64, device="cpu")
    optimizer = torch.optim.SGD([w], lr=eta, momentum=0.0, weight_decay=0.0)

    # Only the stipulated normal update sample participates in differentiation.
    loss = (update - w * torch.dot(w, update)).square().sum()
    loss.backward()
    analytical = analytical_gradient(w0, x_u)
    automatic = w.grad.detach().numpy().copy()
    np.testing.assert_allclose(analytical, [0.0, -2.0], rtol=0, atol=1e-12)
    np.testing.assert_allclose(automatic, analytical, rtol=0, atol=1e-12)
    optimizer.step()  # Exactly one SGD step; no renormalization.
    w1 = w.detach().numpy().copy()
    np.testing.assert_allclose(w1, w0 - eta * analytical, rtol=0, atol=1e-12)
    np.testing.assert_allclose(w1, [1.0, 1.0 / 5.0], rtol=0, atol=1e-12)

    # Introduce the unseen probes only after completing the update.
    x_N = np.array([0.2, 1.0], dtype=np.float64)
    x_A = np.array([1.0, 1.1], dtype=np.float64)
    samples = np.stack([x_u, x_N, x_A])
    states = np.stack([w0, w1])
    angles = np.degrees(np.arctan2(states[:, 1], states[:, 0]))
    norms_squared = np.sum(states**2, axis=1)

    print("CPU float64; stipulated labels; eta = 0.1; SGD steps = 1")
    print(f"x_u = {x_u}; x_N = {x_N}; x_A = {x_A}")
    print(f"Analytical gradient: {analytical}; autograd: {automatic}")
    print(f"Maximum gradient discrepancy: {np.max(np.abs(automatic - analytical)):.3e}")
    print(f"w0 = {w0}; w1 = {w1}")
    print(f"Squared parameter norms: {norms_squared}")

    # Exact hand-derived targets, ordered as update sample, normal, anomaly.
    comparisons = (
        ("Tied AE: w w^T", False,
         [386 / 625, 554 / 625, 48821 / 62500], -6579 / 62500),
        ("Normalized projector: w w^T / (w^T w)", True,
         [8 / 13, 288 / 325, 81 / 104], -279 / 2600),
    )
    for name, normalized, expected_after, expected_margin in comparisons:
        values = []
        for state in states:
            operator = np.outer(state, state)
            if normalized:
                operator = operator / (state @ state)
            residual = samples - samples @ operator.T
            values.append(np.sum(residual**2, axis=1))
        values = np.array(values)
        margins = values[:, 2] - values[:, 1]
        np.testing.assert_allclose(
            values, [[1, 1, 121 / 100], expected_after], rtol=0, atol=1e-12
        )
        np.testing.assert_allclose(
            margins, [21 / 100, expected_margin], rtol=0, atol=1e-12
        )
        assert values[1, 0] < values[0, 0], "Normal update loss did not decrease"
        assert values[1, 2] < values[0, 2], "Unseen anomaly score did not decrease"
        assert margins[1] < 0 < margins[0], "Probe ranking did not reverse"

        print(f"\n{name}")
        print(f"{'Quantity':<30} {'Before':>16} {'After':>16} {'Change':>17}")
        rows = [
            ("Normal update loss", values[:, 0]),
            ("Unseen normal score", values[:, 1]),
            ("Unseen anomaly score", values[:, 2]),
            ("Anomaly-normal margin", margins),
            ("Direction angle (degrees)", angles),
        ]
        for label, (before, after) in rows:
            print(f"{label:<30} {before:16.12f} {after:16.12f} {after-before:+17.12f}")

    print("\nPASS: analytical/autograd gradient, exact SGD state, both score tables,")
    print("and both ranking reversals agree within absolute tolerance 1e-12.")
    print("This is an existence counterexample, not a full-stream PR-AUC result.")
    return geometric_analysis()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--geometry-output", help="Optional CSV path for Experiment D")
    mode.add_argument("--controlled-shift-output", help="Run paired synthetic distributions and save to this new directory")
    args = parser.parse_args()
    if args.controlled_shift_output:
        controlled_shift(args.controlled_shift_output)
    else:
        rows = main()
    if args.geometry_output:
        with open(args.geometry_output, "x", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
