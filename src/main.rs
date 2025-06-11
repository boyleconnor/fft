use std::fs::File;
use std::io::{BufReader, Write};
use hound;
use hound::WavReader;
use image;
use std;

const IMAGE_WIDTH: u32 = 2048;
const IMAGE_HEIGHT: u32 = 1024;
const Y_SCALING_FACTOR: i16 = i16::MAX / (IMAGE_HEIGHT as i16);
const SAMPLE_ON_COLOR: image::Rgb<u8> = image::Rgb([u8::MAX, u8::MAX, u8::MAX]);
const MAX_VISUAL_AMPLITUDE: u32 = 100; // In pixels

fn main() {
    let mut reader = hound::WavReader::open("data/ode_to_joy.wav").unwrap();

    let samples: Vec<f64> = reader.samples::<i16>().map(|sample| sample.unwrap() as f64 / i16::MAX as f64).collect();
    println!("number of samples: {}", samples.len());
    println!("file spec: {:?}", reader.spec());
    let buckets = dumb_fourier_transform(&samples, samples.len()-10_000);


    let reconstructed_samples = inverse_fourier_transform(&buckets);
    let mut writer = hound::WavWriter::create("outputs/reconstructed_ode_to_joy.wav", reader.spec()).unwrap();
    for sample in reconstructed_samples.iter().map(|sample| (sample * i16::MAX as f64) as i16) {
        writer.write_sample(sample).unwrap();
    }
    draw_magnitudes(buckets);
    writer.finalize().unwrap();
}

fn inverse_fourier_transform(buckets: &[(f64, f64)]) -> Vec<f64> {
    let mut reconstructed_samples = vec![0.0; buckets.len()];
    for (k, (real, imaginary)) in buckets.iter().enumerate() {
        
        if buckets[k] == (0.0f64, 0.0f64) {
            continue;
        }

        if k % 1_000 == 0 {
            println!("IFT: finished calculating buckets up to: {} / {}, (time = {:?})", k, buckets.len(), std::time::SystemTime::now());
        }
        
        let frequency = k as f64 / buckets.len() as f64;

        for n in 0..reconstructed_samples.len() {
            let (sin, cos) = f64::sin_cos(n as f64 * 2.0 * std::f64::consts::PI * frequency);
            reconstructed_samples[n] += (real * cos) + (imaginary * sin);
        }
    }
    reconstructed_samples.iter().map(|sample| sample / buckets.len() as f64).collect::<_>()
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

fn dumb_fourier_transform(samples: &[f64], min_k: usize) -> Vec<(f64, f64)> {
    let mut buckets = vec![(0.0, 0.0); samples.len()];
    for k in min_k..samples.len() {

        if k % 1_000 == 0 {
            println!("FT: finished calculating buckets up to: {} / {}, (time = {:?})", k, samples.len(), std::time::SystemTime::now());
        }
        
        let frequency = k as f64 / samples.len() as f64;

        let (mut imaginary, mut real) = (0.0f64, 0.0f64);
        for n in 0..samples.len() {
            let (sin, cos) = f64::sin_cos(n as f64 * 2.0f64 * std::f64::consts::PI * frequency);
            // Assume imaginary component is 0:
            imaginary += samples[n] * sin;
            real += samples[n] * cos;
        }
        buckets[k] = (real, imaginary);
    }

    buckets
}

fn fast_fourier_transform(samples: &[f64]) -> Vec<(f64, f64)> {
    let padded_samples = pad_samples(samples);

    fft_(&padded_samples)
}

fn inverse_fast_fourier_transform(buckets: &[(f64, f64)]) -> Vec<f64> {
    ifft_(&buckets).iter().map(|(real, _)| real).cloned().collect::<Vec<f64>>()
}

fn pad_samples(samples: &[f64]) -> Vec<(f64, f64)> {
    let bits = f64::log2(samples.len() as f64).ceil() as u32;
    let padded_size = 2usize.pow(bits);
    assert!(padded_size >= samples.len());
    println!("Padding samples to size: {}", padded_size);
    let mut padded_samples = samples
        .iter()
        .map(|sample| (sample.clone(), 0.0))
        .chain(vec![(0.0, 0.0); padded_size - samples.len()])
        .collect::<Vec<(f64, f64)>>();
    assert_eq!(padded_samples.len(), padded_size);
    padded_samples
}

// FIXME: These don't work:

fn fft_(samples: &[(f64, f64)]) -> Vec<(f64, f64)> {
    xfft_(samples, 1.0)
}
fn ifft_(samples: &[(f64, f64)]) -> Vec<(f64, f64)> {
    xfft_(samples, -1.0)
}

fn xfft_(samples: &[(f64, f64)], sign: f64) -> Vec<(f64, f64)> {
    if samples.len() == 1 {
        vec![samples[0].clone()]
    } else {
        let (mut odds, mut evens) = (vec![], vec![]);
        for (idx, sample) in samples.iter().enumerate() {
            if idx % 2 == 1 {
                odds.push(sample.clone());
            } else {
                evens.push(sample.clone());
            }
        }
        let (odd_fft, even_fft) = (fft_(&odds), fft_(&evens));

        let mut full_fft = vec![(0.0, 0.0); samples.len()];
        for i in 0..(samples.len() / 2) {
            // omega, i.e. i/nth root of unity:
            let (imaginary, real) = f64::sin_cos(2.0 * std::f64::consts::PI * i as f64 / samples.len() as f64);
            full_fft[i] = (odd_fft[i].0 + (real * odd_fft[i].0), even_fft[i].1 + (imaginary * odd_fft[i].1));
            full_fft[i + samples.len() / 2] = (even_fft[i].0 - (real * odd_fft[i].0), even_fft[i].1 - (imaginary * odd_fft[i].1));
        }
        full_fft
    }
}
