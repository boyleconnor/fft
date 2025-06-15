mod complex;

use std::fs::File;
use std::io::{BufReader, Write};
use hound;
use hound::WavReader;
use image;
use std;
use std::ops::Sub;
use crate::complex::Complex;

const IMAGE_WIDTH: u32 = 2048;
const IMAGE_HEIGHT: u32 = 1024;
const Y_SCALING_FACTOR: i16 = i16::MAX / (IMAGE_HEIGHT as i16);
const SAMPLE_ON_COLOR: image::Rgb<u8> = image::Rgb([u8::MAX, u8::MAX, u8::MAX]);
const MAX_VISUAL_AMPLITUDE: u32 = 100; // In pixels

fn main() {
    let mut reader = hound::WavReader::open("data/ode_to_joy.wav").unwrap();

    let samples: Vec<Complex> = reader.samples::<i16>().map(
        |sample| Complex { real: sample.unwrap() as f64 / i16::MAX as f64, imaginary: 0.0 }
    ).collect();
    println!("number of samples: {}", samples.len());
    println!("file spec: {:?}", reader.spec());
    let amplitudes = fast_fourier_transform(&pad_samples(&samples));


    let reconstructed_samples = inverse_fast_fourier_transform(&amplitudes);
    let mut writer = hound::WavWriter::create("outputs/reconstructed_ode_to_joy.wav", reader.spec()).unwrap();
    for sample in reconstructed_samples.iter().map(|sample| (sample.real * i16::MAX as f64) as i16) {
        writer.write_sample(sample).unwrap();
    }
    // draw_magnitudes(buckets);
    writer.finalize().unwrap();
}

fn dumb_inverse_fourier_transform(buckets: &[Complex]) -> Vec<Complex> {
    let mut reconstructed_samples = vec![Complex::zero(); buckets.len()];
    for (k, bucket_value) in buckets.iter().enumerate() {
        
        if buckets[k] == Complex::zero() {
            continue;
        }

        if k % 1_000 == 0 {
            println!("IFT: finished calculating buckets up to: {} / {}, (time = {:?})", k, buckets.len(), std::time::SystemTime::now());
        }
        
        let frequency = k as f64 / reconstructed_samples.len() as f64;

        for n in 0..reconstructed_samples.len() {
            // Complex arc distance, in radians
            let arc = 2.0 * std::f64::consts::PI * frequency * n as f64;
            reconstructed_samples[n] += Complex::unity_root(arc) * *bucket_value;
        }
    }
    reconstructed_samples
}

fn draw_magnitudes(magnitudes: Vec<(f64, f64)>) {
    let mut image = image::RgbImage::new(magnitudes.len() as u32, IMAGE_HEIGHT);
    let max_amplitude = magnitudes
        .iter()
        .map(|(real, imaginary)| (real.abs(), imaginary.abs()))
        .map(
            |(real, imaginary)| {
                std::cmp::max_by(real, imaginary, |a: &f64, b: &f64| a.total_cmp(b))
            }
        )
        .max_by(|a, b| a.total_cmp(b)).unwrap();
    for (period, (real, imaginary)) in magnitudes.iter().enumerate() {
        for (amplitude, center_y) in [
            (real, IMAGE_HEIGHT / 4),
            (imaginary, (IMAGE_HEIGHT * 3) / 4)
        ] {
            let discrete_amplitude = (amplitude / max_amplitude * MAX_VISUAL_AMPLITUDE as f64) as i32;
            let (lower, higher) = if (discrete_amplitude >= 0) { (0, discrete_amplitude) } else { (discrete_amplitude, 0) };
            for offset in lower..higher {
                image.put_pixel(period as u32, (center_y as i32 + offset) as u32, image::Rgb([u8::MAX; 3]))
            }
        }
    }
    image.save("outputs/fourier_transform.png").unwrap();
}

fn save_waveform(reader: &mut WavReader<BufReader<File>>) {
    let x_scaling_factor = reader.duration() / IMAGE_WIDTH;
    // let mut writer = hound::WavWriter::create("outputs/mono_output.wav", reader.spec()).unwrap();
    let mut image = image::RgbImage::new(IMAGE_WIDTH, IMAGE_HEIGHT);

    let samples = reader.samples::<i16>();
    for (idx, sample) in samples.enumerate() {
        if (idx as u32) % x_scaling_factor == 0 {
            let x = (idx as u32) / x_scaling_factor;
            for y in 0..(sample.unwrap() / Y_SCALING_FACTOR) {
                image.put_pixel(x, y as u32, SAMPLE_ON_COLOR);
            }
        }
    }
    image.save_with_format("outputs/image.png", image::ImageFormat::Png).unwrap()
}

fn dumb_fourier_transform(samples: &[Complex], min_k: usize) -> Vec<Complex> {
    let mut buckets = vec![Complex::zero(); samples.len()];
    for k in min_k..samples.len() {

        if k % 1_000 == 0 {
            println!("FT: finished calculating buckets up to: {} / {}, (time = {:?})", k, samples.len(), std::time::SystemTime::now());
        }
        
        let frequency = k as f64 / samples.len() as f64;

        for n in 0..samples.len() {
            // Complex arc distance, in radians
            let arc = 2.0 * std::f64::consts::PI * frequency * n as f64;
            buckets[k] += Complex::unity_root(-arc) * samples[n];
        }
        buckets[k] *= Complex { real: 1.0 / samples.len() as f64, imaginary: 0.0 };
    }

    buckets
}

fn fast_fourier_transform(samples: &[Complex]) -> Vec<Complex> {
    fft_(&samples).iter().map(|amplitude| *amplitude * Complex::real(1.0 / samples.len() as f64)).collect()
}

fn inverse_fast_fourier_transform(amplitudes: &[Complex]) -> Vec<Complex> {
    ifft_(&amplitudes)
}

fn pad_samples(samples: &[Complex]) -> Vec<Complex> {
    let bits = f64::log2(samples.len() as f64).ceil() as u32;
    let padded_size = 2usize.pow(bits);
    assert!(padded_size >= samples.len());
    println!("Padding samples to size: {} = 2^{}", padded_size, bits);
    let mut padded_samples = samples
        .iter()
        .cloned()
        .chain(vec![Complex::zero(); padded_size - samples.len()])
        .collect::<Vec<Complex>>();
    assert_eq!(padded_samples.len(), padded_size);
    padded_samples
}

fn fft_(samples: &[Complex]) -> Vec<Complex> {
    xfft_(samples, -1.0)
}
fn ifft_(samples: &[Complex]) -> Vec<Complex> {
    xfft_(samples, 1.0)
}

fn xfft_(samples: &[Complex], sign: f64) -> Vec<Complex> {
    if samples.len() == 1 {
        vec![samples[0].clone()]
    } else {
        let (mut odds, mut evens) = (vec![], vec![]);
        for (idx, sample) in samples.iter().enumerate() {
            if idx % 2 == 1 { odds.push(sample.clone()); } else { evens.push(sample.clone()); }
        }
        let (odd_fft, even_fft) = (xfft_(&odds, sign), xfft_(&evens, sign));


        let mut full_fft = vec![Complex::zero(); samples.len()];
        for i in 0..(samples.len() / 2) {
            // omega, i.e. i/nth root of unity:
            let omega_i = Complex::unity_root(sign * 2.0 * std::f64::consts::PI * i as f64 / samples.len() as f64);
            full_fft[i] = even_fft[i] + (omega_i * odd_fft[i]);
            full_fft[i + samples.len() / 2] = even_fft[i] - (omega_i * odd_fft[i]);
        }
        full_fft
    }
}
