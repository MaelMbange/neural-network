//! Shared visualisation geometry helpers used by the training-history visualiser.
//!
//! These functions are neuron-agnostic: the decision boundary depends only on
//! `weights` and `bias`, regardless of whether the underlying neuron is a
//! Perceptron, Gradient, or ADALINE.

use egui_plot::PlotPoints;

/// Weights below this magnitude are treated as effectively zero when deciding
/// whether a meaningful boundary or regression line can be drawn.
///
/// Used in [`boundary_line`] and mirrored in the visualiser's `degenerate` check
/// so both agree on when to suppress geometry.
pub const WEIGHT_ZERO_THRESHOLD: f64 = 1e-10;

// ── Decision boundary ─────────────────────────────────────────────────────────

/// Decision boundary line clipped to the viewport rectangle using Liang-Barsky.
///
/// Both endpoints are guaranteed to lie within
/// `[vx_min, vx_max] × [vy_min, vy_max]`, so `egui_plot` never expands its
/// auto-bounds to include off-screen points.
///
/// `threshold` is the net-input value that defines the boundary
/// (`w1·x + w2·y + bias = threshold`).  Pass `0.0` for bipolar `{-1,+1}` labels
/// and `0.5` for unipolar `{0,1}` labels.
///
/// Returns `None` when both weights are effectively zero (no meaningful boundary).
pub fn boundary_line(
    w1: f64,
    w2: f64,
    bias: f64,
    threshold: f64,
    vx_min: f64,
    vx_max: f64,
    vy_min: f64,
    vy_max: f64,
) -> Option<PlotPoints<'static>> {
    if w1.abs() < WEIGHT_ZERO_THRESHOLD && w2.abs() < WEIGHT_ZERO_THRESHOLD {
        return None;
    }

    // Parametric direction along the line (perpendicular to the normal (w1, w2))
    let dx = -w2;
    let dy = w1;

    // Any point on the line w1·x + w2·y + bias = threshold
    let (ox, oy) = if w2.abs() > WEIGHT_ZERO_THRESHOLD {
        (0.0_f64, (threshold - bias) / w2)
    } else {
        ((threshold - bias) / w1, 0.0_f64)
    };

    // Clip P(t) = (ox + t·dx, oy + t·dy) to the viewport with Liang-Barsky.
    let mut t_lo = f64::NEG_INFINITY;
    let mut t_hi = f64::INFINITY;

    // Clip one axis: coord(t) = origin + t·delta must stay in [lo, hi].
    let mut clip_axis = |delta: f64, origin: f64, lo: f64, hi: f64| -> bool {
        if delta.abs() < 1e-15 {
            return origin >= lo && origin <= hi;
        }
        let (ta, tb) = ((lo - origin) / delta, (hi - origin) / delta);
        let (t_enter, t_exit) = if delta > 0.0 { (ta, tb) } else { (tb, ta) };
        t_lo = t_lo.max(t_enter);
        t_hi = t_hi.min(t_exit);
        t_lo < t_hi
    };

    if !clip_axis(dx, ox, vx_min, vx_max) {
        return None;
    }
    if !clip_axis(dy, oy, vy_min, vy_max) {
        return None;
    }

    Some(
        vec![
            [ox + t_lo * dx, oy + t_lo * dy],
            [ox + t_hi * dx, oy + t_hi * dy],
        ]
        .into(),
    )
}

// ── Sutherland-Hodgman clipping ───────────────────────────────────────────────

/// Clip a polygon to the half-plane `w1·x + w2·y + bias ≥ threshold`
/// using the Sutherland-Hodgman algorithm.
pub fn clip_to_halfplane(
    polygon: &[[f64; 2]],
    w1: f64,
    w2: f64,
    bias: f64,
    threshold: f64,
) -> Vec<[f64; 2]> {
    if polygon.is_empty() {
        return vec![];
    }
    let inside = |p: [f64; 2]| w1 * p[0] + w2 * p[1] + bias >= threshold;
    let intersect = |a: [f64; 2], b: [f64; 2]| -> [f64; 2] {
        let dx = b[0] - a[0];
        let dy = b[1] - a[1];
        let denom = w1 * dx + w2 * dy;
        // t such that w1·(a+t·d).x + w2·(a+t·d).y + bias = threshold
        let t = if denom.abs() > 1e-15 {
            (threshold - w1 * a[0] - w2 * a[1] - bias) / denom
        } else {
            0.0
        };
        [a[0] + t * dx, a[1] + t * dy]
    };

    let n = polygon.len();
    let mut result = Vec::with_capacity(n + 1);
    for i in 0..n {
        let curr = polygon[i];
        let next = polygon[(i + 1) % n];
        let curr_in = inside(curr);
        let next_in = inside(next);
        if curr_in {
            result.push(curr);
            if !next_in {
                result.push(intersect(curr, next));
            }
        } else if next_in {
            result.push(intersect(curr, next));
        }
    }
    result
}

/// Viewport-clipped polygon for the class-1 half-plane
/// (`w1·x + w2·y + bias ≥ threshold`).
///
/// Takes the current plot bounds so the polygon never extends outside the
/// visible area.
pub fn class1_polygon(
    w1: f64,
    w2: f64,
    bias: f64,
    threshold: f64,
    vx_min: f64,
    vx_max: f64,
    vy_min: f64,
    vy_max: f64,
) -> Vec<[f64; 2]> {
    let rect: &[[f64; 2]] = &[
        [vx_min, vy_min],
        [vx_max, vy_min],
        [vx_max, vy_max],
        [vx_min, vy_max],
    ];
    clip_to_halfplane(rect, w1, w2, bias, threshold)
}

// ── Regression line ───────────────────────────────────────────────────────────

/// Regression line `y = w1·x + bias` evaluated at both viewport x-endpoints.
///
/// Returns a two-point [`PlotPoints`] spanning `[vx_min, vx_max]`.  Unlike the
/// 2-D decision boundary this is always a function of x, so no Liang-Barsky
/// clipping is needed — the caller controls the x-range directly.
///
/// The caller is responsible for suppressing this call when `w1` is near zero
/// (use [`WEIGHT_ZERO_THRESHOLD`]); otherwise a flat horizontal line would be
/// drawn that is indistinguishable from a bias-only prediction.
pub fn regression_line(w1: f64, bias: f64, vx_min: f64, vx_max: f64) -> PlotPoints<'static> {
    vec![
        [vx_min, w1 * vx_min + bias],
        [vx_max, w1 * vx_max + bias],
    ]
    .into()
}
