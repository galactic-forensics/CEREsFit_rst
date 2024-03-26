extern crate nalgebra as na;

use std::fmt;

mod regression;

/// Data structure.
///
/// In order to perform a linear fit, you must package your data as a `Data` struct. The following
/// list provides an overview of the fields:
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
///
/// let xdat = vec![1., 2., 3.];
/// let sigx = vec![0.1, 0.02, 0.32];
/// let ydat = vec![20., 25.4, 32.9];
/// let sigy = vec![1.2, 2.4, 0.13];
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
pub struct Data {
    pub xdat: Vec<f64>,
    pub sigx: Vec<f64>,
    pub ydat: Vec<f64>,
    pub sigy: Vec<f64>,
    pub rho: Option<Vec<f64>>,
    pub fixpt: Option<Vec<f64>>,
}

/// LinearFit structure that holds the results.
///
/// Running the linear fit routine will, if successful, return a LinearFit structure that holds
/// the results of the calculations. LinearFit contains three fields: `slope`, `intercept`, and
/// `mswd`. `slope` and `intercept` contain each an array with two `f64` values representing the
/// value (item 0) and its uncertainty (item1). `mswd` holds the mean squared weighted deviation
/// of the linear regression.
///
/// # Example
///
/// todo
pub struct LinearFit {
    slope: [f64; 2],
    intercept: [f64; 2],
    mswd: f64,
}

impl fmt::Display for LinearFit {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "Slope: {} ± {}\nIntercept: {} ± {}\nMSWD: {}",
            self.slope[0], self.slope[1], self.intercept[0], self.intercept[1], self.mswd
        )
    }
}


#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn display_linear_fit() {
        let my_result = LinearFit {
            slope: [42.0, 0.2],
            intercept: [0.1, 1.2],
            mswd: 1.132,
        };
        assert_eq!(
            format!("{my_result}"),
            "Slope: 42 ± 0.2\nIntercept: 0.1 ± 1.2\nMSWD: 1.132"
        )
    }
}
