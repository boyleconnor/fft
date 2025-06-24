use crate::complex::Complex;

pub fn fast_fourier_transform(samples: &[Complex]) -> Vec<Complex> {
    fft_(&samples)
}

pub fn inverse_fast_fourier_transform(amplitudes: &[Complex]) -> Vec<Complex> {
    ifft_(&amplitudes).iter().map(|amplitude| *amplitude * Complex::real(1.0 / amplitudes.len() as f64)).collect()
}

fn fft_(samples: &[Complex]) -> Vec<Complex> {
    cooley_tukey(samples, -1.0)
}

fn ifft_(samples: &[Complex]) -> Vec<Complex> {
    cooley_tukey(samples, 1.0)
}

const PRIMES: [usize; 25] = [
    02, 03, 05, 07, 11,
    13, 17, 19, 23, 29,
    31, 37, 41, 43, 47,
    53, 59, 61, 67, 71,
    73, 79, 83, 89, 97
];

/// Perform Cooley-Tukey algorithm on `x`, a sequence of complex numbers. `sign` indicates the sign
/// of the exponent inside the integrand; it should be `-1.0` for the Fourier transform, and `1.0`
/// for the inverse Fourier transform.
fn cooley_tukey(x: &[Complex], sign: f64) -> Vec<Complex> {
    let n = x.len();
    if x.len() == 1 {
        vec![x[0].clone()]
    } else {
        let candidate_r_2 = PRIMES.iter().find(|r| (**r < n) && n % *r == 0);
        let r_2 = candidate_r_2.cloned().unwrap_or(n);
        let r_1 = n / r_2;
        let mut x_prime = vec![];
        // k_0 is a modulus class
        for k_0 in 0..r_2 {
            let elements_in_class = (0..r_1).map(|k_1| x[k_1 * r_2 + k_0]).collect::<Vec<_>>();
            x_prime.push(cooley_tukey(&elements_in_class, sign));
        }

        let mut y = vec![Complex::zero(); n];
        for j_0 in 0..r_1 {
            for j_1 in 0..r_2 {
                let x_idx = (j_1 * r_1) + j_0;
                for k_0 in 0..r_2 {
                    y[x_idx] += x_prime[k_0][j_0] * Complex::w((j_1 * r_1 + j_0) * k_0, n, sign)
                }
            }
        }

        y
    }
}

/// Perform Bluestein's algorithm on `x`, a sequence of Complex numbers
fn bluestein(x: &[Complex], sign: f64) -> Vec<Complex> {
    let chirp_range = (-(x.len() as isize) + 1)..(x.len() as isize);
    let x_q: Vec<Complex> = x.iter().enumerate().map(|(idx, v)| *v * Complex::cis(sign * std::f64::consts::PI * (idx as f64).powf(2.0) / x.len() as f64)).collect();
    let w_q: Vec<Complex> = chirp_range.map(|i| Complex::cis(-sign * std::f64::consts::PI * (i as f64).powf(2.0) / x.len() as f64)).collect();
    let convolution = fft_convolve(&x_q, &w_q);

    // fft_convolve(&x_q, &w_q)[0..x.len()].iter().enumerate().map(|(k, v)| *v * Complex::cis(sign * std::f64::consts::PI * (k as f64).powf(2.0) / x.len() as f64)).collect::<Vec<Complex>>()
    // NOTE: This x.len()-1..x.len()*2-1 is very important and easy to get wrong from the definition in
    convolution[x.len()-1..x.len()*2-1].iter().enumerate().map(|(k, v)| *v * Complex::cis(sign * std::f64::consts::PI * (k as f64).powf(2.0) / x.len() as f64)).collect::<Vec<Complex>>()
}

fn convolve(x: &[Complex], y: &[Complex]) -> Vec<Complex> {
    let output_size = x.len() + y.len() - 1;
    let mut output = vec![Complex::zero(); output_size];
    for i in 0..x.len() {
        for j in 0..y.len() {
            output[i+j] += x[i] * y[j];
        }
    }
    output
}

fn fft_convolve(x: &[Complex], y: &[Complex]) -> Vec<Complex> {
    let output_size = x.len() + y.len() - 1;
    let (x_, y_) = (pad_to_size(x, output_size), pad_to_size(y, output_size));
    // For Cooley-Tukey speedup, pad to next power of 2:
    let (x_, y_) = (pad_to_power_of_2(&x_), pad_to_power_of_2(&y_));
    assert!(x_.len() % 2 == 0);
    assert!(y_.len() % 2 == 0);
    // Note: it is necessary that (forward) FFT does not normalize the output, but iFFT does;
    // otherwise the output will be double-normalized
    let (f_x, f_y) = (fast_fourier_transform(&x_), fast_fourier_transform(&y_));
    let f_z: Vec<Complex> = f_x.iter().zip(f_y).map(|(a, b)| *a * b).collect();
    let mut output = inverse_fast_fourier_transform(&f_z);
    output.truncate(output_size);
    output
}

fn pad_to_power_of_2(samples: &[Complex]) -> Vec<Complex> {
    let bits = f64::log2(samples.len() as f64).ceil() as u32;
    let padded_size = 2usize.pow(bits);
    assert!(padded_size >= samples.len());
    // println!("Padding samples to size: {} = 2^{}", padded_size, bits);
    pad_to_size(samples, padded_size)
}

fn pad_to_size(samples: &[Complex], padded_size: usize) -> Vec<Complex> {
    let padded_samples = samples
        .iter()
        .cloned()
        .chain(vec![Complex::zero(); padded_size - samples.len()])
        .collect::<Vec<Complex>>();
    assert_eq!(padded_samples.len(), padded_size);
    padded_samples
}

#[test]
fn test_fft_convolve() {
    let arr1 = [12., 2., 9.].map(|x| Complex::real(x));
    let arr2 = [29., 4., 10.].map(|x| Complex::real(x));

    let convolution = fft_convolve(&arr1, &arr2);
    let expected_convolution: [Complex; 5] = [348., 106., 389.,  56.,  90.].map(|x| Complex::real(x));

    for (x, y) in convolution.iter().zip(expected_convolution) {
        assert!(x.euclidean_distance(y) < 0.01, "\nexpected: {:?}\nactual: {:?}", expected_convolution, convolution);
    }
}

#[test]
fn test_fft_convolve2() {
    let arr1 = [3, -2, 6, 21, 18].map(|x| Complex::real(x as f64));
    let arr2 = [-12, 5, 7].map(|x| Complex::real(x as f64));

    let convolution = fft_convolve(&arr1, &arr2);
    let expected_convolution = [-36, 39, -61, -236, -69, 237, 126].map(|x| Complex::real(x as f64));

    for (x, y) in convolution.iter().zip(expected_convolution) {
        assert!(x.euclidean_distance(y) < 0.01, "\nexpected: {:?}\nactual: {:?}", expected_convolution, convolution);
    }
}

#[test]
fn test_bluestein() {
    let signal = [0.2, -0.3, 0.4, -0.5, 0.2, -0.1, 0.7].map(|x| Complex::real(x));
    let ct_fft = cooley_tukey(&signal, -1.0);
    let bluestein_fft = bluestein(&signal, -1.0);
    println!("bluestein_fft: {bluestein_fft:#?}");

    for (x, y) in ct_fft.iter().zip(bluestein_fft.clone()) {
        assert!(x.euclidean_distance(y) < 0.01, "\nexpected: {:#?}\nactual: {:#?}", ct_fft, bluestein_fft);
    }
}
