/// Module for linear regression calculations.
use ndarray::prelude::*;

use crate::{LinearFit, UncertaintyBand};

/// Data structure.
///
/// In order to perform a linear fit, you must package your data as a `Data` struct. The following
/// list provides an overview of the fields, which all need to be given as one dimensional
/// `ndarray::Array1<f64>` vectors.
///
/// - `xdat`: x-data
/// - `sigx`: 1 sigma uncertainties of your x-data
/// - `ydat`: y-data
/// - `sigy`: 1 sigma uncertainties of your y-data
/// - `rho`: Optional list of correlation coefficients for between each individual x,y data point
/// - `fixpt`: Optional fix point through which to force the regression, given as a `Vec[x, y]`,
///         where `x` and `y` are the respective coordinates to force the fit through.
///
/// Except for `fixpt`, which must be a vector of length 2, all other vectors must contain,
/// if provided, the same number of values.
///
/// # Example:
///
/// ```
/// use ceresfit::Data;
/// use ndarray::array;
///
/// let xdat = array![1., 2., 3.];
/// let sigx = array![0.1, 0.02, 0.32];
/// let ydat = array![20., 25.4, 32.9];
/// let sigy = array![1.2, 2.4, 0.13];
/// let rho = None;
/// let fixpt = None;
///
/// let my_data = Data {
///     xdat,
///     sigx,
///     ydat,
///     sigy,
///     rho,
///     fixpt
/// };
///
/// let result = my_data.linear_fit().unwrap();
///
/// assert_eq!(result.slope, [6.371384342113318, 1.1441759334226203]);
/// assert_eq!(result.intercept, [13.491888793660642, 2.1663793263027946]);
/// assert_eq!(result.mswd, 0.15143729948155993);
/// ```
pub struct Data {
    pub xdat: Array1<f64>,
    pub sigx: Array1<f64>,
    pub ydat: Array1<f64>,
    pub sigy: Array1<f64>,
    pub rho: Option<Array1<f64>>,
    pub fixpt: Option<Array1<f64>>,
}

impl Data {
    pub fn linear_fit(&self) -> Result<LinearFit, String> {
        // Check input data
        let len_xdat = self.xdat.len();
        let len_sigx = self.sigx.len();
        let len_ydat = self.ydat.len();
        let len_sigy = self.sigy.len();

        if len_xdat != len_sigx || len_xdat != len_ydat || len_xdat != len_sigy {
            return Err("Length of xdat, ydat, and/or uncertainties do not match".to_owned());
        }

        if let Some(ref rho) = self.rho {
            if rho.len() != len_xdat {
                return Err("Length of rho does not match xdat".to_owned());
            }
        }

        if let Some(ref fixpt) = self.fixpt {
            if fixpt.len() != 2 {
                return Err("Length of fixpt must be of length 2".to_owned());
            }
        }

        // Create ndarrays to use in calculations
        let sigxy = match &self.rho {
            Some(rho) => rho * &self.sigx * &self.sigy,
            None => Array1::zeros(self.sigx.len()),
        };

        // Calculate all quantities for the linear fit
        let (slope, xbar, ybar, weights) = self.calculate_slope(&sigxy)?;
        let intercept = ybar - slope * xbar;
        let mswd = self.calculate_mswd(slope, intercept, &weights);
        let [slope_unc, intercept_unc] = match &self.fixpt {
            Some(fixpt) => self.unc_calc_fixpt(&sigxy, slope, xbar, ybar, &weights, fixpt),
            None => self.unc_calc_no_fixpt(&sigxy, slope, xbar, ybar, &weights),
        };

        let result = LinearFit {
            slope: [slope, slope_unc],
            intercept: [intercept, intercept_unc],
            mswd,
        };

        Ok(result)
    }

    pub fn uncertainty_band(
        &mut self,
        sigma: Option<f64>,
        bins: Option<usize>,
        xrange: Option<&Array1<f64>>,
    ) -> UncertaintyBand {
        let xrange_return = match xrange {
            Some(x) => x.clone(),
            None => {
                let num_bins = bins.unwrap_or(100);
                let min_x = *self
                    .xdat
                    .iter()
                    .min_by(|x, y| x.partial_cmp(y).unwrap())
                    .unwrap();
                let max_x = *self
                    .xdat
                    .iter()
                    .max_by(|x, y| x.partial_cmp(y).unwrap())
                    .unwrap();
                Array1::linspace(min_x, max_x, num_bins)
            }
        };

        let mut y_ub: Array1<f64> = Array1::zeros(xrange_return.len());

        // save the current xdat
        let xdat_save = self.xdat.clone();

        for (it, deltax) in xrange_return.iter().enumerate() {
            self.xdat = xdat_save.mapv(|x: f64| x - deltax);
            let result = self.linear_fit().unwrap();
            y_ub[it] = result.intercept[1];
        }

        y_ub *= sigma.unwrap_or(1.0);

        // write the saved xdat back
        self.xdat = xdat_save;
        let result = self.linear_fit().unwrap();

        let y_ub_min = &xrange_return * result.slope[0] + result.intercept[0] - &y_ub;
        let y_ub_max = &xrange_return * result.slope[0] + result.intercept[0] - &y_ub;

        UncertaintyBand {
            x: xrange_return,
            y_ub_min,
            y_ub_max,
        }
    }

    /// Calculate MSWD of the linear regression.
    fn calculate_mswd(&self, slope: f64, intercept: f64, weights: &Array1<f64>) -> f64 {
        let chi_square = (weights
            * (&self.ydat - slope * &self.xdat - intercept).mapv(|x: f64| x.powi(2)))
        .sum();
        let dof = match self.fixpt {
            Some(_) => self.xdat.len() - 1,
            None => self.xdat.len() - 2,
        };
        chi_square / dof as f64
    }

    /// Calculate the slope fully considering the uncertainties
    fn calculate_slope(&self, sigxy: &Array1<f64>) -> Result<(f64, f64, f64, Array1<f64>), String> {
        let sigx_sq = self.sigx.mapv(|x: f64| x.powi(2));
        let sigy_sq = self.sigy.mapv(|x: f64| x.powi(2));

        let calc_xbar = |weights: &Array1<f64>| -> f64 {
            match &self.fixpt {
                Some(f) => f[0],
                None => (weights * &self.xdat).sum() / weights.sum(),
            }
        };

        let calc_ybar = |weights: &Array1<f64>| -> f64 {
            match &self.fixpt {
                Some(f) => f[1],
                None => (weights * &self.ydat).sum() / weights.sum(),
            }
        };

        let calc_weights = |slp: f64| -> Array1<f64> {
            1.0 / (&sigy_sq + slp.powi(2) * &sigx_sq - 2.0 * slp * sigxy)
        };

        let iterate_slope = |slp: f64| -> f64 {
            let weights = calc_weights(slp);
            let weights_sq = &weights * &weights;
            let u_all = &self.xdat - calc_xbar(&weights);
            let v_all = &self.ydat - calc_ybar(&weights);

            (&weights_sq * &v_all * (&u_all * &sigy_sq + slp * &v_all * &sigx_sq - &v_all * sigxy))
                .sum()
                / (&weights_sq
                    * &u_all
                    * (&u_all * &sigy_sq + slp * &v_all * &sigx_sq - slp * &u_all * sigxy))
                    .sum()
        };

        let regression_limit = 1e-10; // fixme: change for Opt in struct - maybe
                                      // fixme: Can set this to zero for some, but not for everything! Check w/ all datasets.

        let mut iter_count: usize = 0;
        let mut slope_old = self.initial_guesses_slope()?;
        let mut slope_new = iterate_slope(slope_old);

        while (slope_old - slope_new).abs() > regression_limit {
            slope_old = slope_new;
            slope_new = iterate_slope(slope_new);
            iter_count += 1;

            if iter_count >= 1000000 {
                // fixme: define somewhere for user
                return Err("Reached maximum iter count. No slope found.".to_owned());
            }
        }

        let weights = calc_weights(slope_new);
        let xbar = calc_xbar(&weights);
        let ybar = calc_ybar(&weights);
        Ok((slope_new, xbar, ybar, weights))
    }

    /// Simple linear regression to calculate the slope and intercept of a line.
    fn initial_guesses_slope(&self) -> Result<f64, String> {
        let n = self.xdat.len();
        if n < 2 {
            return Err("Not enough data points".to_owned());
        }

        let sum_x = self.xdat.sum();
        let sum_y = self.ydat.sum();
        let sum_x2 = self.xdat.mapv(|x: f64| x.powi(2)).sum();
        let sum_xy = (&self.xdat * &self.ydat).sum();

        let slope = (n as f64 * sum_xy - sum_x * sum_y) / (n as f64 * sum_x2 - sum_x.powi(2));

        Ok(slope)
    }

    /// Uncertainty calculation with a fixed point
    fn unc_calc_fixpt(
        &self,
        sigxy: &Array1<f64>,
        slope: f64,
        xbar: f64,
        ybar: f64,
        weights: &Array1<f64>,
        fixpt: &Array1<f64>,
    ) -> [f64; 2] {
        let u_all = &self.xdat - xbar;
        let v_all = &self.ydat - ybar;

        let sigx_sq = self.sigx.mapv(|x: f64| x.powi(2));
        let sigy_sq = self.sigy.mapv(|x: f64| x.powi(2));
        let weights_sq = weights.mapv(|x: f64| x.powi(2));
        let u_all_sq = u_all.mapv(|x: f64| x.powi(2));
        let v_all_sq = v_all.mapv(|x: f64| x.powi(2));

        let dthdb = (&weights_sq
            * (2.0 * slope * (&u_all * &v_all * &sigx_sq - &u_all_sq * sigxy)
                + (&u_all_sq * &sigy_sq - &v_all_sq * &sigx_sq)))
            .sum()
            + 4.0
                * (weights.mapv(|x: f64| x.powi(3))
                    * (sigxy - slope * &sigx_sq)
                    * (slope.powi(2) * (&u_all * &v_all * &sigx_sq - &u_all_sq * sigxy)
                        + slope * (&u_all_sq * &sigy_sq - &v_all_sq * &sigx_sq)
                        - (&u_all * &v_all * &sigy_sq - &v_all_sq * sigxy)))
                    .sum();

        let calc_dtheta_dxi = |it: usize| -> f64 {
            weights_sq[it]
                * (slope.powi(2) * v_all[it] * sigx_sq[it]
                    - slope.powi(2) * 2.0 * u_all[it] * sigxy[it]
                    + 2.0 * slope * u_all[it] * sigy_sq[it]
                    - v_all[it] * sigy_sq[it])
        };

        let calc_dtheta_dyi = |it: usize| -> f64 {
            weights_sq[it]
                * (slope.powi(2) * u_all[it] * sigx_sq[it]
                    - 2.0 * slope * v_all[it] * sigx_sq[it]
                    - u_all[it] * sigy_sq[it]
                    + 2.0 * v_all[it] * sigxy[it])
        };

        let mut slope_unc_sq = 0.0;
        for (it, sigxi) in self.sigx.indexed_iter() {
            let sigyi = self.sigy[it];
            let sigxyi = sigxy[it];
            let dtheta_dxi = calc_dtheta_dxi(it);
            let dtheta_dyi = calc_dtheta_dyi(it);
            slope_unc_sq += dtheta_dxi.powi(2) * sigxi.powi(2)
                + dtheta_dyi.powi(2) * sigyi.powi(2)
                + 2.0 * sigxyi * dtheta_dxi * dtheta_dyi;
        }
        slope_unc_sq /= dthdb.powi(2);

        let intercept_unc_sq = fixpt[0].powi(2) * slope_unc_sq;

        [slope_unc_sq.sqrt(), intercept_unc_sq.sqrt()]
    }

    /// Returns the slope and intercept uncertainty as an array.

    /// Uncertainty calculation without a fixed point.
    ///
    /// Returns the slope and intercept uncertainties as an array.
    fn unc_calc_no_fixpt(
        &self,
        sigxy: &Array1<f64>,
        slope: f64,
        xbar: f64,
        ybar: f64,
        weights: &Array1<f64>,
    ) -> [f64; 2] {
        let u_all = &self.xdat - xbar;
        let v_all = &self.ydat - ybar;

        let sigx_sq = self.sigx.mapv(|x: f64| x.powi(2));
        let sigy_sq = self.sigy.mapv(|x: f64| x.powi(2));
        let weights_sq = weights.mapv(|x: f64| x.powi(2));
        let weights_sum = weights.sum();
        let u_all_sq = u_all.mapv(|x: f64| x.powi(2));
        let v_all_sq = v_all.mapv(|x: f64| x.powi(2));

        // sum over j in equation 60, last two lines
        let sum_j_u = (&weights_sq * &u_all * (sigxy - slope * &sigx_sq)).sum();
        let sum_j_v = (&weights_sq * &v_all * (sigxy - slope * &sigx_sq)).sum();

        // d(theta) / db derivative
        let dthdb = (&weights_sq
            * (2.0 * slope * (&u_all * &v_all * &sigx_sq - &u_all_sq * sigxy)
                + (&u_all_sq * &sigy_sq - &v_all_sq * &sigx_sq)))
            .sum()
            + 4.0
                * (weights.mapv(|x: f64| x.powi(3))
                    * (sigxy - slope * &sigx_sq)
                    * (slope.powi(2) * (&u_all * &v_all * &sigx_sq - &u_all_sq * sigxy)
                        + slope * (&u_all_sq * &sigy_sq - &v_all_sq * &sigx_sq)
                        - (&u_all * &v_all * &sigy_sq - &v_all_sq * sigxy)))
                    .sum()
            + 2.0
                * (&weights_sq
                    * (-slope.powi(2) * &u_all * &sigx_sq
                        + 2.0 * slope * &v_all * &sigx_sq
                        + &u_all * &sigy_sq
                        - 2.0 * &v_all * sigxy))
                    .sum()
                * sum_j_v
                / weights_sum
            + 2.0
                * (&weights_sq
                    * (-slope.powi(2) * &v_all * &sigx_sq + 2.0 * slope.powi(2) * &u_all * sigxy
                        - 2.0 * slope * &u_all * &sigy_sq
                        + &v_all * &sigy_sq))
                    .sum()
                * sum_j_u
                / weights_sum;

        let calc_dtheta_dxi = |it: usize| -> f64 {
            let mut sum_all = 0.0;
            for (jt, weight_jt) in weights.indexed_iter() {
                let kron = kron_delta(it, jt);
                sum_all += weight_jt.powi(2)
                    * (kron - weights[it] / weights_sum)
                    * (slope.powi(2) * v_all[jt] * sigx_sq[jt]
                        - slope.powi(2) * 2.0 * u_all[jt] * sigxy[jt]
                        + 2.0 * slope * u_all[jt] * sigy_sq[jt]
                        - v_all[jt] * sigy_sq[jt]);
            }
            sum_all
        };
        let calc_dtheta_dyi = |it: usize| -> f64 {
            let mut sum_all = 0.0;
            for (jt, weight_jt) in weights.indexed_iter() {
                let kron = kron_delta(it, jt);
                sum_all += weight_jt.powi(2)
                    * (kron - weights[it] / weights_sum)
                    * (slope.powi(2) * u_all[jt] * sigx_sq[jt]
                        - 2.0 * slope * v_all[jt] * sigx_sq[jt]
                        - u_all[jt] * sigy_sq[jt]
                        + 2.0 * v_all[jt] * sigxy[jt]);
            }
            sum_all
        };
        let calc_da_dxi = |it: usize| -> f64 {
            -slope * weights[it] / weights_sum - xbar * calc_dtheta_dxi(it) / dthdb
        };
        let calc_da_dyi =
            |it: usize| -> f64 { weights[it] / weights_sum - xbar * calc_dtheta_dyi(it) / dthdb };

        let mut slope_unc_sq = 0.0;
        for (it, sigxi) in self.sigx.indexed_iter() {
            let sigyi = self.sigy[it];
            let sigxyi = sigxy[it];
            let dtheta_dxi = calc_dtheta_dxi(it);
            let dtheta_dyi = calc_dtheta_dyi(it);
            slope_unc_sq += dtheta_dxi.powi(2) * sigxi.powi(2)
                + dtheta_dyi.powi(2) * sigyi.powi(2)
                + 2.0 * sigxyi * dtheta_dxi * dtheta_dyi;
        }
        slope_unc_sq /= dthdb.powi(2);

        let mut intercept_unc_sq = 0.0;
        for (it, sigxi) in self.sigx.indexed_iter() {
            let sigyi = self.sigy[it];
            let sigxyi = sigxy[it];
            let da_dxi = calc_da_dxi(it);
            let da_dyi = calc_da_dyi(it);
            intercept_unc_sq += da_dxi.powi(2) * sigxi.powi(2)
                + da_dyi.powi(2) * sigyi.powi(2)
                + 2.0 * sigxyi * da_dxi * da_dyi;
        }

        [slope_unc_sq.sqrt(), intercept_unc_sq.sqrt()]
    }
}

/// Kronecker delta function.
fn kron_delta(i: usize, j: usize) -> f64 {
    if i == j {
        1.0
    } else {
        0.0
    }
}

#[cfg(test)]
mod tests {
    use approx::assert_relative_eq;
    use ndarray::prelude::*;

    use super::*;

    #[test]
    fn test_initial_guesses() {
        let data = Data {
            xdat: array![1.0, 2.0, 3.0, 4.0, 5.0],
            sigx: array![0.0, 0.0, 0.0, 0.0, 0.0],
            ydat: array![2.0, 3.0, 4.0, 5.0, 6.0],
            sigy: array![0.0, 0.0, 0.0, 0.0, 0.0],
            rho: None,
            fixpt: None,
        };
        let test = data.initial_guesses_slope().unwrap();
        assert_eq!(test, 1.0);
    }

    #[test]
    fn test_initial_guesses_approximate() {
        let data = Data {
            xdat: array![1.0, 2.0, 3.0, 4.0, 5.0],
            sigx: array![0.0, 0.0, 0.0, 0.0, 0.0],
            ydat: array![2.1, 2.9, 4.2, 5.1, 5.7],
            sigy: array![0.0, 0.0, 0.0, 0.0, 0.0],
            rho: None,
            fixpt: None,
        };
        let test = data.initial_guesses_slope().unwrap();
        assert_relative_eq!(test, 0.94);
    }

    #[test]
    fn test_kron_delta() {
        assert_eq!(kron_delta(1, 1), 1.0);
        assert_eq!(kron_delta(1, 2), 0.0);
        assert_eq!(kron_delta(3, 2), 0.0);
    }

    #[test]
    fn test_uncertainty_band_sigma() {
        let mut data = Data {
            xdat: array![1.0, 2.0, 3.0, 4.0, 5.0],
            sigx: array![0.1, 0.1, 0.1, 0.1, 0.1],
            ydat: array![1.0, 2.0, 3.0, 4.0, 5.0],
            sigy: array![0.1, 0.1, 0.1, 0.1, 0.1],
            rho: None,
            fixpt: None,
        };
        let unc_exp = data.uncertainty_band(Some(1.0), None, None);
        let unc_rec = data.uncertainty_band(None, None, None);
        assert_eq!(unc_exp.x, unc_rec.x);
        assert_eq!(unc_exp.y_ub_min, unc_rec.y_ub_min);
        assert_eq!(unc_exp.y_ub_max, unc_rec.y_ub_max);
    }

    #[test]
    fn test_uncertainty_band_xrange() {
        let mut data = Data {
            xdat: array![1.0, 2.0, 3.0, 4.0, 5.0],
            sigx: array![0.1, 0.1, 0.1, 0.1, 0.1],
            ydat: array![1.0, 2.0, 3.0, 4.0, 5.0],
            sigy: array![0.1, 0.1, 0.1, 0.1, 0.1],
            rho: None,
            fixpt: None,
        };
        let xrange = array![1.0, 2.0, 3.0, 4.0, 5.0];
        let unc_exp = data.uncertainty_band(Some(1.0), None, Some(&xrange));
        assert_eq!(unc_exp.x, xrange);
        assert_eq!(unc_exp.y_ub_min.len(), xrange.len());
        assert_eq!(unc_exp.y_ub_max.len(), xrange.len());
    }

    #[test]
    fn test_uncertainty_band_bins() {
        let mut data = Data {
            xdat: array![1.0, 2.0, 3.0, 4.0, 5.0],
            sigx: array![0.1, 0.1, 0.1, 0.1, 0.1],
            ydat: array![1.0, 2.0, 3.0, 4.0, 5.0],
            sigy: array![0.1, 0.1, 0.1, 0.1, 0.1],
            rho: None,
            fixpt: None,
        };
        let unc_exp = data.uncertainty_band(Some(1.0), Some(10), None);
        assert_eq!(unc_exp.x.len(), 10);
        assert_eq!(unc_exp.y_ub_min.len(), 10);
        assert_eq!(unc_exp.y_ub_max.len(), 10);
    }

    // fixme: remove
    #[test]
    fn test_ndarray_tmp() {
        let arr = array![1.0, 2.0, 3.0, 3.3];
        let arr2 = &arr * &arr;
        let arr3 = arr.mapv(|x: f64| x.powi(2));
        assert_eq!(arr2, arr3);
        println!("{}", &arr2[0]);
    }
}
