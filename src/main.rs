use hound;

fn main() {
    let mut reader = hound::WavReader::open("data/test_mono.wav").unwrap();
    let mut writer = hound::WavWriter::create("outputs/mono_output.wav", reader.spec()).unwrap();

    let samples = reader.samples::<i16>();
    for (idx, sample) in samples.enumerate() {
        if idx % 2 == 0 {
            writer.write_sample(sample.unwrap().saturating_mul(40)).unwrap();
        }
    }
}
