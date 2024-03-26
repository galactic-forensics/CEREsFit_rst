/// Module for linear regression calculations.

pub mod regression {
    use ndarray::{Array, Ix1};

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

        // Create nalgebra vectors to use in calculations
        let xdat = Array::from_vec(data.xdat.clone());
        let sigx = Array::from_vec(data.sigx.clone());
        let ydat = Array::from_vec(data.ydat.clone());
        let sigy = Array::from_vec(data.sigy.clone());
        let sigxy = match &data.rho {
            Some(rho) => {
                let rho = Array::from_vec(rho.clone());
                &rho * &sigx * &sigy
            }
            None => Array::zeros(sigx.len()),
        };
        let fixpt = match data.fixpt {
            Some(ref fixpt) => {
                if fixpt.len() != 2 {
                    return Err("Length of fixpt must be of length 2".to_owned());
                }
                Some(Array::from_vec(fixpt.clone()))
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
        xdat: &Array<f64, Ix1>,
        sigx: &Array<f64, Ix1>,
        ydat: &Array<f64, Ix1>,
        sigy: &Array<f64, Ix1>,
        sigxy: &Array<f64, Ix1>,
        fixpt: &Option<Array<f64, Ix1>>,
    ) -> Result<f64, String> {
        let sigx_sq = sigx * sigx;
        let sigy_sq = sigy * sigy;

        let calc_xbar = |weights: &Array<f64, Ix1>| -> f64 {
            match fixpt {
                Some(f) => *f.get(0).unwrap(),
                None => {
                    let sum_arr = weights * xdat;
                    sum_arr.sum() / weights.sum()
                }
            }
        };

        let calc_ybar = |weights: &Array<f64, Ix1>| -> f64 {
            match fixpt {
                Some(f) => *f.get(0).unwrap(),
                None => {
                    let sum_arr = weights * ydat;
                    sum_arr.sum() / weights.sum()
                }
            }
        };

        let calc_weights =
            |slp: f64| -> Array<f64, Ix1> { 1.0 / (&sigy_sq + slp * &sigx_sq - 2.0 * slp * sigxy) };

        let iterate_slope = |slp: f64| -> f64 {
            let weights = calc_weights(slp);
            let weights_sq = &weights * &weights;
            let u_all = xdat - calc_xbar(&weights);
            let v_all = ydat - calc_ybar(&weights);

            let nominator = &weights_sq
                * &v_all
                * (&u_all * &sigy_sq + slp * &v_all * &sigx_sq - &v_all * sigxy);
            let denominator = &weights_sq
                * &u_all
                * (&u_all * &sigy_sq + slp * &v_all * &sigx_sq - slp * &u_all * sigxy);

            nominator.sum() / denominator.sum()
        };

        let regression_limit = 1e-10; // fixme: change for Opt in struct

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
        Ok(slope_new)
    }

    /// Simple linear regression to calculate the slope and intercept of a line.
    fn initial_guesses_slope(
        xdat: &Array<f64, Ix1>,
        ydat: &Array<f64, Ix1>,
    ) -> Result<f64, String> {
        let n = xdat.len();
        if n < 2 {
            return Err("Not enough data points".to_owned());
        }

        let sum_x = xdat.iter().sum::<f64>();
        let sum_y = ydat.iter().sum::<f64>();
        let sum_x2 = xdat.iter().map(|x| x.powi(2)).sum::<f64>();
        let sum_xy = xdat
            .iter()
            .zip(ydat.iter())
            .map(|(x, y)| x * y)
            .sum::<f64>();

        let slope = (n as f64 * sum_xy - sum_x * sum_y) / (n as f64 * sum_x2 - sum_x.powi(2));

        Ok(slope)
    }

    fn kronecker_delta(i: Vec<usize>, j: Vec<usize>) -> Vec<usize> {
        i.iter()
            .zip(j.iter())
            .map(|(a, b)| if a == b { 1 } else { 0 })
            .collect()
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
        fn test_kronecker_delta() {
            let i = vec![1, 2, 3];
            let j = vec![1, 2, 4];
            let test = kronecker_delta(i, j);
            assert_eq!(test, vec![1, 1, 0]);
        }
    }
}
