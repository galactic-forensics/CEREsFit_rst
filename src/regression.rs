/// Module for linear regression calculations.

pub mod regression {
    use ndarray::prelude::*;
    use std::ops::Mul;

    use crate::{Data, LinearFit};

    pub fn linear_fit(data: &Data) -> Result<LinearFit, String> {
        // Check input data
        let len_xdat = data.xdat.len();
        let len_sigx = data.sigx.len();
        let len_ydat = data.ydat.len();
        let len_sigy = data.sigy.len();

        if len_xdat != len_sigx || len_xdat != len_ydat || len_xdat != len_sigy {
            return Err("Length of xdat, ydat, and/or uncertainties do not match".to_owned());
        }

        match data.rho {
            Some(ref rho) => {
                if rho.len() != len_xdat {
                    return Err("Length of rho does not match xdat".to_owned());
                }
            }
            None => {}
        }

        // Create ndarrays to use in calculations
        let xdat = Array1::from_vec(data.xdat.clone());
        let sigx = Array1::from_vec(data.sigx.clone());
        let ydat = Array1::from_vec(data.ydat.clone());
        let sigy = Array1::from_vec(data.sigy.clone());
        let sigxy = match &data.rho {
            Some(rho) => Array1::from_vec(rho.clone()) * &sigx * &sigy,
            None => Array1::zeros(sigx.len()),
        };
        let fixpt = match data.fixpt {
            Some(ref fixpt) => {
                if fixpt.len() != 2 {
                    return Err("Length of fixpt must be of length 2".to_owned());
                }
                Some(Array1::from_vec(fixpt.clone()))
            }
            None => None,
        };

        // Calculate all quantities for the linear fit
        let (slope, xbar, ybar, weights) =
            calculate_slope(&xdat, &sigx, &ydat, &sigy, &sigxy, &fixpt)?;
        let intercept = ybar - slope * xbar;
        let mswd = calculate_mswd(&xdat, &ydat, slope, intercept, &weights, &fixpt);
        let [slope_unc, intercept_unc] = match fixpt {
            Some(fixpt) => unc_calc_fixpt(
                &xdat, &sigx, &ydat, &sigy, &sigxy, slope, xbar, ybar, &weights, &fixpt),
            None => unc_calc_no_fixpt(
                &xdat, &sigx, &ydat, &sigy, &sigxy, slope, xbar, ybar, &weights,
            ),
        };

        let result = LinearFit {
            slope: [slope, slope_unc],
            intercept: [intercept, intercept_unc],
            mswd,
        };

        Ok(result)
    }

    /// Calculate MSWD of the linear regression.
    fn calculate_mswd(
        xdat: &Array1<f64>,
        ydat: &Array1<f64>,
        slope: f64,
        intercept: f64,
        weights: &Array1<f64>,
        fixpt: &Option<Array1<f64>>,
    ) -> f64 {
        let chi_square =
            (weights * (ydat - slope * xdat - intercept).mapv(|x: f64| x.powi(2))).sum();
        let dof = match fixpt {
            Some(_) => xdat.len() - 1,
            None => xdat.len() - 2,
        };
        chi_square / dof as f64
    }

    /// Calculate the slope fully considering the uncertainties
    fn calculate_slope(
        xdat: &Array1<f64>,
        sigx: &Array1<f64>,
        ydat: &Array1<f64>,
        sigy: &Array1<f64>,
        sigxy: &Array1<f64>,
        fixpt: &Option<Array<f64, Ix1>>,
    ) -> Result<(f64, f64, f64, Array1<f64>), String> {
        let sigx_sq = sigx.mapv(|x: f64| x.powi(2));
        let sigy_sq = sigy.mapv(|x: f64| x.powi(2));

        let calc_xbar = |weights: &Array1<f64>| -> f64 {
            match fixpt {
                Some(f) => f[0],
                None => (weights * xdat).sum() / weights.sum(),
            }
        };

        let calc_ybar = |weights: &Array1<f64>| -> f64 {
            match fixpt {
                Some(f) => f[1],
                None => (weights * ydat).sum() / weights.sum(),
            }
        };

        let calc_weights = |slp: f64| -> Array1<f64> {
            1.0 / (&sigy_sq + slp.powi(2) * &sigx_sq - 2.0 * slp * sigxy)
        };

        let iterate_slope = |slp: f64| -> f64 {
            let weights = calc_weights(slp);
            let weights_sq = &weights * &weights;
            let u_all = xdat - calc_xbar(&weights);
            let v_all = ydat - calc_ybar(&weights);

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
        let mut slope_old = initial_guesses_slope(xdat, ydat)?;
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
    fn initial_guesses_slope(xdat: &Array1<f64>, ydat: &Array1<f64>) -> Result<f64, String> {
        let n = xdat.len();
        if n < 2 {
            return Err("Not enough data points".to_owned());
        }

        let sum_x = xdat.sum();
        let sum_y = ydat.sum();
        let sum_x2 = xdat.mapv(|x: f64| x.powi(2)).sum();
        let sum_xy = xdat.mul(ydat).sum();

        let slope = (n as f64 * sum_xy - sum_x * sum_y) / (n as f64 * sum_x2 - sum_x.powi(2));

        Ok(slope)
    }

    /// Kronecker delta function.
    fn kron_delta(i: usize, j: usize) -> f64 {
        if i == j {
            1.0
        } else {
            0.0
        }
    }

    /// Uncertainty calculation with a fixed point
    ///
    fn unc_calc_fixpt(
        xdat: &Array1<f64>,
        sigx: &Array1<f64>,
        ydat: &Array1<f64>,
        sigy: &Array1<f64>,
        sigxy: &Array1<f64>,
        slope: f64,
        xbar: f64,
        ybar: f64,
        weights: &Array1<f64>,
        fixpt: &Array1<f64>,
    ) -> [f64; 2] {
        let u_all = xdat - xbar;
        let v_all = ydat - ybar;

        let sigx_sq = sigx.mapv(|x: f64| x.powi(2));
        let sigy_sq = sigy.mapv(|x: f64| x.powi(2));
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
            &weights_sq[it]
                * (slope.powi(2) * &v_all[it] * &sigx_sq[it]
                    - slope.powi(2) * 2.0 * &u_all[it] * sigxy[it]
                    + 2.0 * slope * &u_all[it] * &sigy_sq[it]
                    - &v_all[it] * &sigy_sq[it])
        };
        
        let calc_dtheta_dyi = |it: usize| -> f64 {
            &weights_sq[it]
                * (slope.powi(2) * &u_all[it] * &sigx_sq[it]
                    - 2.0 * slope * &v_all[it] * &sigx_sq[it]
                    - &u_all[it] * &sigy_sq[it]
                    + 2.0 * &v_all[it] * sigxy[it])
        };
        
        let mut slope_unc_sq = 0.0;
        for (it, sigxi) in sigx.indexed_iter() {
            let sigyi = sigy[it];
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
        xdat: &Array1<f64>,
        sigx: &Array1<f64>,
        ydat: &Array1<f64>,
        sigy: &Array1<f64>,
        sigxy: &Array1<f64>,
        slope: f64,
        xbar: f64,
        ybar: f64,
        weights: &Array1<f64>,
    ) -> [f64; 2] {
        let u_all = xdat - xbar;
        let v_all = ydat - ybar;

        let sigx_sq = sigx.mapv(|x: f64| x.powi(2));
        let sigy_sq = sigy.mapv(|x: f64| x.powi(2));
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
                * &sum_j_v
                / &weights_sum
            + 2.0
                * (&weights_sq
                    * (-slope.powi(2) * &v_all * &sigx_sq + 2.0 * slope.powi(2) * &u_all * sigxy
                        - 2.0 * slope * &u_all * &sigy_sq
                        + &v_all * &sigy_sq))
                    .sum()
                * &sum_j_u
                / &weights_sum;

        let calc_dtheta_dxi = |it: usize| -> f64 {
            let mut sum_all = 0.0;
            for (jt, weight_jt) in weights.indexed_iter() {
                let kron = kron_delta(it, jt);
                sum_all += weight_jt.powi(2)
                    * (kron - weights[it] / weights_sum)
                    * (slope.powi(2) * &v_all[jt] * &sigx_sq[jt]
                        - slope.powi(2) * 2.0 * &u_all[jt] * sigxy[jt]
                        + 2.0 * slope * &u_all[jt] * &sigy_sq[jt]
                        - &v_all[jt] * &sigy_sq[jt]);
            }
            sum_all
        };
        let calc_dtheta_dyi = |it: usize| -> f64 {
            let mut sum_all = 0.0;
            for (jt, weight_jt) in weights.indexed_iter() {
                let kron = kron_delta(it, jt);
                sum_all += weight_jt.powi(2)
                    * (kron - weights[it] / weights_sum)
                    * (slope.powi(2) * &u_all[jt] * &sigx_sq[jt]
                        - 2.0 * slope * &v_all[jt] * &sigx_sq[jt]
                        - &u_all[jt] * &sigy_sq[jt]
                        + 2.0 * &v_all[jt] * sigxy[jt]);
            }
            sum_all
        };
        let calc_da_dxi = |it: usize| -> f64 {
            -slope * weights[it] / weights_sum - xbar * calc_dtheta_dxi(it) / dthdb
        };
        let calc_da_dyi =
            |it: usize| -> f64 { weights[it] / weights_sum - xbar * calc_dtheta_dyi(it) / dthdb };

        let mut slope_unc_sq = 0.0;
        for (it, sigxi) in sigx.indexed_iter() {
            let sigyi = sigy[it];
            let sigxyi = sigxy[it];
            let dtheta_dxi = calc_dtheta_dxi(it);
            let dtheta_dyi = calc_dtheta_dyi(it);
            slope_unc_sq += dtheta_dxi.powi(2) * sigxi.powi(2)
                + dtheta_dyi.powi(2) * sigyi.powi(2)
                + 2.0 * sigxyi * dtheta_dxi * dtheta_dyi;
        }
        slope_unc_sq /= dthdb.powi(2);

        let mut intercept_unc_sq = 0.0;
        for (it, sigxi) in sigx.indexed_iter() {
            let sigyi = sigy[it];
            let sigxyi = sigxy[it];
            let da_dxi = calc_da_dxi(it);
            let da_dyi = calc_da_dyi(it);
            intercept_unc_sq += da_dxi.powi(2) * sigxi.powi(2)
                + da_dyi.powi(2) * sigyi.powi(2)
                + 2.0 * sigxyi * da_dxi * da_dyi;
        }

        [slope_unc_sq.sqrt(), intercept_unc_sq.sqrt()]
    }

    #[cfg(test)]
    mod tests {
        use approx::assert_relative_eq;
        use ndarray::prelude::*;

        use super::*;

        #[test]
        fn test_initial_guesses() {
            let xdat = array![1.0, 2.0, 3.0, 4.0, 5.0];
            let ydat = array![2.0, 3.0, 4.0, 5.0, 6.0];
            let test = initial_guesses_slope(&xdat, &ydat).unwrap();
            assert_eq!(test, 1.0);
        }

        #[test]
        fn test_initial_guesses_approximate() {
            let xdat = array![1.0, 2.0, 3.0, 4.0, 5.0];
            let ydat = array![2.1, 2.9, 4.2, 5.1, 5.7];
            let test = initial_guesses_slope(&xdat, &ydat).unwrap();
            assert_relative_eq!(test, 0.94);
        }

        #[test]
        fn test_kron_delta() {
            assert_eq!(kron_delta(1, 1), 1.0);
            assert_eq!(kron_delta(1, 2), 0.0);
            assert_eq!(kron_delta(3, 2), 0.0);
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
}
