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

fn main() {
    let mut reader = hound::WavReader::open("data/test_mono.wav").unwrap();

    let (sin_magnitudes, cos_magnitudes) = dumb_fourier_transform(&reader.samples().map(|sample| sample.unwrap()).collect::<Vec<i16>>(), 5_000);
    println!("sin_magnitudes: {:?}", sin_magnitudes);
    
    let mut image = image::RgbImage::new(sin_magnitudes.len() as u32, IMAGE_HEIGHT);
    for (idx, (sin_magnitude, cos_magnitude)) in sin_magnitudes.iter().zip(cos_magnitudes).enumerate() {
        for i in 0..(IMAGE_HEIGHT / 5) {
            image.put_pixel(idx as u32, IMAGE_HEIGHT / 4 + i, image::Rgb::from([(u8::MAX as f32 * sin_magnitude) as u8; 3]));
            image.put_pixel(idx as u32, (IMAGE_HEIGHT * 3) / 4 + i, image::Rgb::from([(u8::MAX as f32 * cos_magnitude) as u8; 3]));
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

fn dumb_fourier_transform(samples: &[i16], max_window_size: usize) -> (Vec<f32>, Vec<f32>) {
    let mut sin_magnitudes = vec![0.0; max_window_size + 1];
    let mut cos_magnitudes = vec![0.0; max_window_size + 1];
    for window_size in 2..=max_window_size {
        let mut sin_magnitude = 0.0;
        let mut cos_magnitude = 0.0;
        for i in 0..samples.len() {
            let (sin, cos) = f32::sin_cos(i as f32 * 2.0f32 * std::f32::consts::PI / window_size as f32);
            sin_magnitude += samples[i] as f32 * sin;
            cos_magnitude += samples[i] as f32 * cos;
        }
        sin_magnitudes[window_size] = sin_magnitude / (samples.len() as f32);
        cos_magnitudes[window_size] = cos_magnitude / (samples.len() as f32);
    }

    (sin_magnitudes, cos_magnitudes)
}
