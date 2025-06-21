use crate::complex::Complex;

pub fn fast_fourier_transform(samples: &[Complex]) -> Vec<Complex> {
    fft_(&samples).iter().map(|amplitude| *amplitude * Complex::real(1.0 / samples.len() as f64)).collect()
}

pub fn inverse_fast_fourier_transform(amplitudes: &[Complex]) -> Vec<Complex> {
    ifft_(&amplitudes)
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