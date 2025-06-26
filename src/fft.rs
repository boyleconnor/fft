use crate::complex::Complex;

pub fn fast_fourier_transform(samples: &[Complex]) -> Vec<Complex> {
    fft_(&samples)
}

pub fn inverse_fast_fourier_transform(amplitudes: &[Complex]) -> Vec<Complex> {
    ifft_(&amplitudes).iter().map(|amplitude| *amplitude * Complex::real(1.0 / amplitudes.len() as f64)).collect()
}

fn fft_(samples: &[Complex]) -> Vec<Complex> {
    frankenstein_fft(samples, -1.0)
}

fn ifft_(samples: &[Complex]) -> Vec<Complex> {
    frankenstein_fft(samples, 1.0)
}

const MAX_PRIME: usize = 150;

fn get_prime_factors(n: usize) -> (usize, Vec<usize>) {
    let mut remaining_factor = n;
    let mut prime_factors = vec![];
    for i in 2..=MAX_PRIME {
        if i > remaining_factor {
            break;
        }
        while remaining_factor % i == 0 {
            remaining_factor /= i;
            prime_factors.push(i);
        }
    }

    println!("factorization: ({}, {:?})", remaining_factor, prime_factors);
    (remaining_factor, prime_factors)
}

/// Perform Cooley-Tukey algorithm on `x`, a sequence of complex numbers, until no small primes
/// remain to divide the sample length, then perform Bluestein's algorithm. `sign` indicates the
/// sign of the exponent inside the integrand; it should be `-1.0` for the Fourier transform, and
/// `1.0` for the inverse Fourier transform.
fn frankenstein_fft(x: &[Complex], sign: f64) -> Vec<Complex> {
    let (remaining_factor, prime_factors) = get_prime_factors(x.len());
    let f_w_q = get_f_w_q(remaining_factor, sign);
    frankenstein_fft_(x, &prime_factors, &f_w_q, sign)
}

fn get_f_w_q(n: usize, sign: f64) -> Vec<Complex> {
    let chirp_range = (-(n as isize) + 1)..(n as isize);
    let w_q: Vec<Complex> = chirp_range.map(|i| Complex::cis(-sign * std::f64::consts::PI * (i as f64).powf(2.0) / n as f64)).collect();
    let padded_size = 2 * n - 1;
    let padded_samples = w_q
        .iter()
        .cloned()
        .chain(vec![Complex::zero(); padded_size - w_q.len()])
        .collect::<Vec<Complex>>();
    assert_eq!(padded_samples.len(), padded_size);
    let bits = f64::log2(padded_samples.len() as f64).ceil() as u32;
    let padded_size1 = 2usize.pow(bits);
    assert!(padded_size1 >= padded_samples.len());
    // println!("Padding samples to size: {} = 2^{}", padded_size, bits);
    let w_q_padded = padded_samples
        .iter()
        .cloned()
        .chain(vec![Complex::zero(); padded_size1 - padded_samples.len()])
        .collect::<Vec<Complex>>();
    assert_eq!(w_q_padded.len(), padded_size1);
    cooley_tukey2(&w_q_padded, -1.0)
}

fn frankenstein_fft_(x: &[Complex], prime_factors: &[usize], w_q_f: &[Complex], sign: f64) -> Vec<Complex> {
    let n = x.len();
    if n == 1 {
        vec![x[0].clone()]
    } else if prime_factors.len() > 0 {
        let r_2 = prime_factors[0].clone();
        let r_1 = n / r_2;
        let mut x_prime = vec![];
        // k_0 is a modulus class
        for k_0 in 0..r_2 {
            let elements_in_class = (0..r_1).map(|k_1| x[k_1 * r_2 + k_0]).collect::<Vec<_>>();
            x_prime.push(frankenstein_fft_(&elements_in_class, &prime_factors[1..], w_q_f, sign));
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
    } else {
        bluestein(x, w_q_f, sign)
    }
}

/// Perform Cooley-Tukey algorithm on `x`, a sequence of complex numbers. ONLY works for `x` where
/// `x.len()` is a power of 2 (incl. `1`). `sign` indicates the sign  of the exponent inside the
/// integrand; it should be `-1.0` for the Fourier transform, and `1.0` for the inverse Fourier
/// transform.
fn cooley_tukey2(x: &[Complex], sign: f64) -> Vec<Complex> {
    let n = x.len();
    if n == 1 {
        vec![x[0].clone()]
    } else {
        debug_assert!(n % 2 == 0);
        let r_1 = n / 2;
        let mut x_prime = vec![];
        for k_0 in 0..2 {
            let elements_in_class = (0..r_1).map(|k_1| x[k_1 * 2 + k_0]).collect::<Vec<_>>();
            x_prime.push(cooley_tukey2(&elements_in_class, sign));
        }

        let mut y = vec![Complex::zero(); n];
        for j_0 in 0..r_1 {
            for j_1 in 0..2 {
                let x_idx = (j_1 * r_1) + j_0;
                for k_0 in 0..2 {
                    y[x_idx] += x_prime[k_0][j_0] * Complex::w((j_1 * r_1 + j_0) * k_0, n, sign)
                }
            }
        }

        y
    }
}

/// Perform Bluestein's algorithm on `x`, a sequence of Complex numbers
fn bluestein(x: &[Complex], f_w_q: &[Complex], sign: f64) -> Vec<Complex> {
    let x_q: Vec<Complex> = x.iter().enumerate().map(|(idx, v)| *v * Complex::cis(sign * std::f64::consts::PI * (idx as f64).powf(2.0) / x.len() as f64)).collect();

    // Perform convolution:
    let (output_size, x_) = pad_for_convolution(&x_q);

    // Note: it is necessary that (forward) FFT does not normalize the output, but iFFT does;
    // otherwise the output will be double-normalized
    let f_x = cooley_tukey2(&x_, -1.0);
    let f_z: Vec<Complex> = f_x.iter().zip(f_w_q).map(|(a, b)| *a * *b).collect();
    let mut convolution: Vec<Complex> = cooley_tukey2(&f_z, 1.0).iter().map(|amplitude| *amplitude * Complex::real(1.0 / f_z.len() as f64)).collect();
    convolution.truncate(output_size);

    // fft_convolve(&x_q, &w_q)[0..x.len()].iter().enumerate().map(|(k, v)| *v * Complex::cis(sign * std::f64::consts::PI * (k as f64).powf(2.0) / x.len() as f64)).collect::<Vec<Complex>>()
    // NOTE: This x.len()-1..x.len()*2-1 is very important and easy to get wrong from the definition in
    convolution[x.len()-1..x.len()*2-1].iter().enumerate().map(|(k, v)| *v * Complex::cis(sign * std::f64::consts::PI * (k as f64).powf(2.0) / x.len() as f64)).collect::<Vec<Complex>>()
}

fn pad_for_convolution(x_q: &Vec<Complex>) -> (usize, Vec<Complex>) {
    let output_size = 2 * x_q.len() - 1;
    // For Cooley-Tukey speedup, pad to next power of 2:
    let min_padded_samples = x_q
        .iter()
        .cloned()
        .chain(vec![Complex::zero(); output_size - x_q.len()])
        .collect::<Vec<Complex>>();
    debug_assert_eq!(min_padded_samples.len(), output_size);
    let bits = f64::log2(min_padded_samples.len() as f64).ceil() as u32;
    let padded_size = 2usize.pow(bits);
    debug_assert!(padded_size >= min_padded_samples.len());
    let x_ = min_padded_samples
        .iter()
        .cloned()
        .chain(vec![Complex::zero(); padded_size - min_padded_samples.len()])
        .collect::<Vec<Complex>>();
    debug_assert_eq!(x_.len(), padded_size);
    (output_size, x_)
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
    let padded_samples = y
        .iter()
        .cloned()
        .chain(vec![Complex::zero(); output_size - y.len()])
        .collect::<Vec<Complex>>();
    assert_eq!(padded_samples.len(), output_size);
    let padded_samples1 = x
        .iter()
        .cloned()
        .chain(vec![Complex::zero(); output_size - x.len()])
        .collect::<Vec<Complex>>();
    assert_eq!(padded_samples1.len(), output_size);
    let (x_, y_) = (padded_samples1, padded_samples);
    // For Cooley-Tukey speedup, pad to next power of 2:
    let bits = f64::log2(y_.len() as f64).ceil() as u32;
    let padded_size = 2usize.pow(bits);
    assert!(padded_size >= y_.len());
    // println!("Padding samples to size: {} = 2^{}", padded_size, bits);
    let padded_samples2 = y_
        .iter()
        .cloned()
        .chain(vec![Complex::zero(); padded_size - y_.len()])
        .collect::<Vec<Complex>>();
    assert_eq!(padded_samples2.len(), padded_size);
    let bits1 = f64::log2(x_.len() as f64).ceil() as u32;
    let padded_size1 = 2usize.pow(bits1);
    assert!(padded_size1 >= x_.len());
    // println!("Padding samples to size: {} = 2^{}", padded_size, bits);
    let padded_samples3 = x_
        .iter()
        .cloned()
        .chain(vec![Complex::zero(); padded_size1 - x_.len()])
        .collect::<Vec<Complex>>();
    assert_eq!(padded_samples3.len(), padded_size1);
    let (x_, y_) = (padded_samples3, padded_samples2);

    // Note: it is necessary that (forward) FFT does not normalize the output, but iFFT does;
    // otherwise the output will be double-normalized
    let (f_x, f_y) = (cooley_tukey2(&x_, -1.0), cooley_tukey2(&y_, -1.0));
    let f_z: Vec<Complex> = f_x.iter().zip(f_y).map(|(a, b)| *a * b).collect();
    let mut output: Vec<Complex> = cooley_tukey2(&f_z, 1.0).iter().map(|amplitude| *amplitude * Complex::real(1.0 / f_z.len() as f64)).collect();
    output.truncate(output_size);
    output
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
    let signal = [0.2, -0.3, 0.4, -0.5, 0.2, -0.1, 0.7, 0.2].map(|x| Complex::real(x));
    let ct_fft = cooley_tukey2(&signal, -1.0);
    let f_w_q = get_f_w_q(signal.len(), -1.0);
    let bluestein_fft = bluestein(&signal, &f_w_q, -1.0);
    println!("bluestein_fft: {bluestein_fft:#?}");

    for (x, y) in ct_fft.iter().zip(bluestein_fft.clone()) {
        assert!(x.euclidean_distance(y) < 0.01, "\nexpected: {:#?}\nactual: {:#?}", ct_fft, bluestein_fft);
    }
}

#[test]
fn test_get_prime_factors1() {
    let (largest_factor, prime_factors) = get_prime_factors(123789136);
    let (expected_largest_factor, expected_factors) = (389 * 19889, vec![2, 2, 2, 2]);
    assert_eq!(prime_factors, expected_factors);
    assert_eq!(largest_factor, expected_largest_factor);
}

#[test]
fn test_get_prime_factors2() {
    let (largest_factor, prime_factors) = get_prime_factors(12);
    let (expected_largest_factor, expected_factors) = (1, vec![2, 2, 3]);
    assert_eq!(prime_factors, expected_factors);
    assert_eq!(largest_factor, expected_largest_factor);
}
