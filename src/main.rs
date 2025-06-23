mod complex;
mod fft;

use std::time::Instant;
use hound;
use crate::complex::Complex;

fn main() {
    const INPUT_FILENAME: &'static str = "data/ode_to_joy.wav";
    println!("Reading from file: {}", INPUT_FILENAME);
    let mut reader = hound::WavReader::open(INPUT_FILENAME).unwrap();

    let samples: Vec<Complex> = reader.samples::<i16>().map(
        |sample| Complex { real: sample.unwrap() as f64 / i16::MAX as f64, imaginary: 0.0 }
    ).collect();
    println!("number of samples: {}", samples.len());
    println!("file spec: {:?}", reader.spec());

    println!("Starting FFT...");
    let start = Instant::now();
    let amplitudes = fft::fast_fourier_transform(&samples);
    let duration = start.elapsed();
    println!("FFT completed in: {:?}", duration);

    println!("Starting inverse FFT...");
    let start = Instant::now();
    let reconstructed_samples = fft::inverse_fast_fourier_transform(&amplitudes);
    let duration = start.elapsed();
    println!("inverse FFT completed in: {:?}", duration);

    const OUTPUT_FILENAME: &'static str = "outputs/reconstructed_ode_to_joy.wav";
    println!("Writing to file: {}", OUTPUT_FILENAME);
    let mut writer = hound::WavWriter::create(OUTPUT_FILENAME, reader.spec()).unwrap();
    for sample in reconstructed_samples.iter().map(|sample| (sample.real * i16::MAX as f64) as i16) {
        writer.write_sample(sample).unwrap();
    }
    // draw_magnitudes(buckets);
    writer.finalize().unwrap();
}
