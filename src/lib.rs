use std::fmt;

mod regression;

pub use regression::Data;


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
/// ```
/// use ceresfit::Data;
/// use ndarray::prelude::*;
///
/// let my_data = Data {
///     xdat: array![1.0, 2.0, 3.0, 4.0, 5.0],
///     sigx: array![0.1, 0.1, 0.1, 0.1, 0.1],
///     ydat: array![1.0, 2.0, 3.0, 4.0, 5.0],
///     sigy: array![0.1, 0.1, 0.1, 0.1, 0.1],
///     rho: None,
///     fixpt: None,
/// };
///
/// let result = my_data.linear_fit().unwrap();
///
///
/// assert_eq!(result.slope[0], 1.0);
/// assert_eq!(result.intercept[0], 0.0);
/// 
/// // Pretty print the results
/// println!("{}", result);
/// ```
pub struct LinearFit {
    pub slope: [f64; 2],
    pub intercept: [f64; 2],
    pub mswd: f64,
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
