use std::fs::File;
use std::io::BufReader;
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
    let mut reader = hound::WavReader::open("data/test_mono.wav").unwrap();

    let samples: Vec<f64> = reader.samples::<i16>().map(|sample| sample.unwrap() as f64 / i16::MAX as f64).collect();
    println!("number of samples: {}", samples.len());
    println!("file spec: {:?}", reader.spec());
    let magnitudes = dumb_fourier_transform(&samples, 5_000);


    let reconstructed_samples = inverse_fourier_transform(&magnitudes, samples.len());
    println!("reconstructed samples: {:?}", &reconstructed_samples[40800..50800]);
    let mut writer = hound::WavWriter::create("outputs/reconstructed.wav", reader.spec()).unwrap();
    for sample in reconstructed_samples.iter().map(|sample| (sample * i16::MAX as f64) as i16) {
        writer.write_sample(sample).unwrap();
    }
    draw_magnitudes(magnitudes);
    writer.finalize().unwrap();
}

fn inverse_fourier_transform(magnitudes: &[(f64, f64)], sample_length: usize) -> Vec<f64> {
    let mut reconstructed_samples = vec![0.0; sample_length];
    for (period, (real, imaginary)) in magnitudes.iter().enumerate() {

        if period % 1_000 == 0 {
            println!("FT: finished calculating periods up to: {}, (time = {:?})", period, std::time::SystemTime::now());
        }

        for sample_idx in 0..reconstructed_samples.len() {
            if period != 0 {
                let (sin, cos) = f64::sin_cos(sample_idx as f64 * 2.0 * std::f64::consts::PI / period as f64);
                reconstructed_samples[sample_idx] += (real * cos) + (imaginary * sin);
            }
        }
    }
    reconstructed_samples.iter().map(|sample| sample / sample_length as f64).collect::<_>()
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

fn dumb_fourier_transform(samples: &[f64], max_period: usize) -> Vec<(f64, f64)> {
    let mut magnitudes = vec![(0.0, 0.0); max_period + 1];
    for period in 1..=max_period {

        if period % 1_000 == 0 {
            println!("FT: finished calculating periods up to: {}, (time = {:?})", period, std::time::SystemTime::now());
        }

        let (mut imaginary, mut real) = (0.0f64, 0.0f64);
        for i in 0..samples.len() {
            let (sin, cos) = f64::sin_cos(i as f64 * 2.0f64 * std::f64::consts::PI / period as f64);
            imaginary += samples[i] * sin;
            real += samples[i] * cos;
        }
        magnitudes[period] = (real, imaginary);
    }

    magnitudes
}
