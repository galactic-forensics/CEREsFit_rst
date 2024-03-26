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

        // Calculate the slope of the linear fit
        let slope = [
            calculate_slope(&xdat, &sigx, &ydat, &sigy, &sigxy, &fixpt)?,
            0.0,
        ];

        let result = LinearFit {
            slope,
            intercept: [0.0, 0.0],
            mswd: 0.0,
        };

        Ok(result)
    }

    /// Calculate the slope fully considering the uncertainties
    fn calculate_slope(
        xdat: &Array1<f64>,
        sigx: &Array1<f64>,
        ydat: &Array1<f64>,
        sigy: &Array1<f64>,
        sigxy: &Array1<f64>,
        fixpt: &Option<Array<f64, Ix1>>,
    ) -> Result<f64, String> {
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

        let regression_limit = 0.0; // fixme: change for Opt in struct - maybe

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

        println!("Iter count: {}", iter_count);

        Ok(slope_new)
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
