// Integration test for Set 1 data of paper
use approx::assert_relative_eq;
use ceresfit::{linear_fit, Data};

fn get_data() -> (Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>) {
    let xdat = vec![0.037, 0.035, 0.032, 0.04, 0.013, 0.038, 0.042, 0.03];
    let sigx = vec![
        0.00111, 0.00105, 0.00096, 0.0012, 0.00039, 0.00114, 0.00126, 0.0009,
    ];
    let ydat = vec![
        0.0008, 0.00084, 0.001, 0.00085, 0.0027, 0.00071, 0.00043, 0.0016,
    ];
    let sigy = vec![
        0.00008, 0.000084, 0.0001, 0.000085, 0.00027, 0.000071, 0.000043, 0.00016,
    ];
    let rho = vec![
        0.707106781186548,
        0.707106781186548,
        0.707106781186548,
        0.707106781186548,
        0.707106781186548,
        0.707106781186548,
        0.707106781186548,
        0.707106781186548,
    ];
    let fixpt = vec![0.01, 0.003];
    (xdat, sigx, ydat, sigy, rho, fixpt)
}

/// Test of set 1 dataset with correlated error bars, no forced intercept
#[test]
fn test_correlated_set1() {
    let (xdat, sigx, ydat, sigy, rho, _fixpt) = get_data();

    let slope_exp = [-0.0775157620599649, 0.0102712666882449];
    let intercept_exp = [0.00368416060914567, 0.00037524641619794];
    let mswd_exp = 1.04431697859327;

    let data = Data {
        xdat,
        sigx,
        ydat,
        sigy,
        rho: Some(rho),
        fixpt: None,
    };

    let result = linear_fit(&data).unwrap();

    assert_relative_eq!(slope_exp[0], result.slope[0], epsilon = 1e-10);
    // assert_relative_eq!(slope_exp[1], result.slope[1], epsilon = 1e-10);
    assert_relative_eq!(intercept_exp[0], result.intercept[0], epsilon = 1e-10);
    assert_relative_eq!(intercept_exp[1], result.intercept[1], epsilon = 1e-10);
    assert_relative_eq!(mswd_exp, result.mswd, epsilon = 1e-10);
}

/// Test of set 1 dataset with uncorrelated error bars, no forced intercept
#[test]
fn test_uncorrelated_set1() {
    let (xdat, sigx, ydat, sigy, _rho, _fixpt) = get_data();

    let slope_exp = [-0.0763849664508034, 0.00952058723778022];
    let intercept_exp = [0.00364110802078184, 0.000349049456994308];
    let mswd_exp = 1.73032125998902;

    let data = Data {
        xdat,
        sigx,
        ydat,
        sigy,
        rho: None,
        fixpt: None,
    };

    let result = linear_fit(&data).unwrap();

    assert_relative_eq!(slope_exp[0], result.slope[0], epsilon = 1e-10);
    assert_relative_eq!(slope_exp[1], result.slope[1], epsilon = 1e-10);
    assert_relative_eq!(intercept_exp[0], result.intercept[0], epsilon = 1e-10);
    assert_relative_eq!(intercept_exp[1], result.intercept[1], epsilon = 1e-10);
    assert_relative_eq!(mswd_exp, result.mswd, epsilon = 1e-10);
}

/// Test of set 1 dataset with correlated error bars and forced intercept
#[test]
fn test_correlated_with_fixed_point_set1() {
    let (xdat, sigx, ydat, sigy, rho, fixpt) = get_data();

    let slope_exp = [-0.0808424308784813, 0.00222209076123286];
    let intercept_exp = [0.00380842430878481, 2.22209076123286E-05];
    let mswd_exp = 0.911015001888297;

    let data = Data {
        xdat,
        sigx,
        ydat,
        sigy,
        rho: Some(rho),
        fixpt: Some(fixpt),
    };

    let result = linear_fit(&data).unwrap();

    assert_relative_eq!(slope_exp[0], result.slope[0], epsilon = 1e-10);
    assert_relative_eq!(slope_exp[1], result.slope[1], epsilon = 1e-10);
    assert_relative_eq!(intercept_exp[0], result.intercept[0], epsilon = 1e-10);
    assert_relative_eq!(intercept_exp[1], result.intercept[1], epsilon = 1e-10);
    assert_relative_eq!(mswd_exp, result.mswd, epsilon = 1e-10);
}
