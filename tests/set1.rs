// Integration test for Set 1 data of paper

use ceresfit::{Data, linear_fit};

/// Test of set 1 dataset with correlated error bars, no forced intercept
#[test]
fn test_correlated_set1() {
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

    let slope_exp = [-0.0775157620599649, 0.0];
    let slope_exp = [-0.0763849664508034, 0.0];
    let intercept_exp = [0.0, 0.0];
    let mswd_exp = 0.0;

    let data = Data {
        xdat,
        sigx,
        ydat,
        sigy,
        rho: None, //  Some(rho),
        fixpt: None,
    };

    let result = linear_fit(&data).unwrap();

    assert_eq!(slope_exp, result.slope);
    assert_eq!(intercept_exp, result.intercept);
    assert_eq!(mswd_exp, result.mswd);
}
