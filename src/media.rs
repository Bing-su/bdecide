//! Accept local media or decoded buffers, e.g. CLI paths and application RGB/PCM data.
use camino::Utf8PathBuf;
use image::RgbImage;
use serde::{Deserialize, Serialize};
use symphonia::core::codecs::audio::AudioDecoderOptions;
use symphonia::core::formats::TrackType;
use symphonia::core::formats::probe::Hint;
use symphonia::core::io::MediaSourceStream;

use crate::{Error, Result};

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged, deny_unknown_fields)]
pub enum ImageInput {
    Path(Utf8PathBuf),
    Rgb {
        width: u32,
        height: u32,
        pixels: Vec<u8>,
    },
}

impl ImageInput {
    pub(crate) fn validate(&self) -> Result<()> {
        if let Self::Rgb {
            width,
            height,
            pixels,
        } = self
        {
            // Keep the byte count exact even for malformed dimensions, e.g. u32::MAX by u32::MAX.
            let size = u128::from(*width) * u128::from(*height) * 3;
            if *width == 0 || *height == 0 || size != pixels.len() as u128 {
                return Err(Error::InvalidRequest(
                    "RGB pixels must contain width * height * 3 bytes".into(),
                ));
            }
        }
        Ok(())
    }

    pub(crate) fn decode(&self) -> Result<RgbImage> {
        self.validate()?;
        match self {
            Self::Path(path) => image::ImageReader::open(path)
                .map_err(|source| Error::Io {
                    path: path.clone(),
                    source,
                })?
                .with_guessed_format()
                .map_err(|source| Error::Io {
                    path: path.clone(),
                    source,
                })?
                .decode()
                .map(|image| image.to_rgb8())
                .map_err(|error| Error::InvalidRequest(format!("image {path}: {error}"))),
            Self::Rgb {
                width,
                height,
                pixels,
            } => RgbImage::from_raw(*width, *height, pixels.clone())
                .ok_or_else(|| Error::InvalidRequest("invalid RGB buffer".into())),
        }
    }
}

/// Audio is mono at 16 kHz; float samples use [-1, 1], e.g. `AudioInput::Samples`.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged, deny_unknown_fields)]
pub enum AudioInput {
    Path(Utf8PathBuf),
    Samples { samples: Vec<f32>, sample_rate: u32 },
    Pcm16 { pcm16: Vec<i16>, sample_rate: u32 },
}

impl AudioInput {
    pub(crate) fn validate(&self) -> Result<()> {
        match self {
            Self::Samples {
                samples,
                sample_rate,
            } => {
                check_rate(*sample_rate)?;
                if samples.is_empty() || samples.iter().any(|x| !x.is_finite() || x.abs() > 1.0) {
                    return Err(Error::InvalidRequest(
                        "audio samples must be nonempty, finite and in [-1, 1]".into(),
                    ));
                }
            }
            Self::Pcm16 { pcm16, sample_rate } => {
                check_rate(*sample_rate)?;
                if pcm16.is_empty() {
                    return Err(Error::InvalidRequest(
                        "audio samples must not be empty".into(),
                    ));
                }
            }
            Self::Path(_) => {}
        }
        Ok(())
    }

    pub(crate) fn decode(&self) -> Result<Vec<f32>> {
        self.validate()?;
        let mut samples = match self {
            Self::Samples { samples, .. } => samples.iter().take(480_000).copied().collect(),
            Self::Pcm16 { pcm16, .. } => pcm16
                .iter()
                .take(480_000)
                .map(|&x| f32::from(x) / 32768.0)
                .collect(),
            Self::Path(path) => decode_audio(path)?,
        };
        if samples.is_empty() || samples.iter().any(|x| !x.is_finite() || x.abs() > 1.0) {
            return Err(Error::InvalidRequest(
                "audio samples must be nonempty, finite and in [-1, 1]".into(),
            ));
        }
        // Match the trained frontend's clip limits, e.g. pad 100 ms to 500 ms.
        samples.resize(samples.len().max(8000), 0.0);
        Ok(samples)
    }
}

fn decode_audio(path: &camino::Utf8Path) -> Result<Vec<f32>> {
    let file = std::fs::File::open(path).map_err(|source| Error::Io {
        path: path.into(),
        source,
    })?;
    let error = |error| Error::InvalidRequest(format!("audio {path}: {error}"));
    let mut hint = Hint::new();
    if let Some(extension) = path.extension() {
        hint.with_extension(extension);
    }
    let mut format = symphonia::default::get_probe()
        .probe(
            &hint,
            MediaSourceStream::new(Box::new(file), Default::default()),
            Default::default(),
            Default::default(),
        )
        .map_err(error)?;
    let track = format
        .default_track(TrackType::Audio)
        .ok_or_else(|| Error::InvalidRequest("no audio track".into()))?;
    let params = track
        .codec_params
        .as_ref()
        .and_then(|params| params.audio())
        .ok_or_else(|| Error::InvalidRequest("missing audio codec parameters".into()))?;
    let mut decoder = symphonia::default::get_codecs()
        .make_audio_decoder(params, &AudioDecoderOptions::default())
        .map_err(error)?;
    let id = track.id;
    let mut samples = Vec::new();
    while samples.len() < 480_000 {
        let Some(packet) = format.next_packet().map_err(error)? else {
            break;
        };
        if packet.track_id != id {
            continue;
        }
        // Fail on corrupt packets rather than silently changing the speech the model hears.
        let buffer = decoder.decode(&packet).map_err(error)?;
        check_rate(buffer.spec().rate())?;
        if buffer.spec().channels().count() != 1 {
            return Err(Error::InvalidRequest("audio must be mono".into()));
        }
        let mut decoded = vec![0.0_f32; buffer.samples_interleaved()];
        buffer.copy_to_slice_interleaved(&mut decoded);
        samples.extend(decoded.into_iter().take(480_000 - samples.len()));
    }
    Ok(samples)
}

fn check_rate(rate: u32) -> Result<()> {
    if rate != 16_000 {
        return Err(Error::InvalidRequest(
            "audio sample_rate must be 16000 Hz".into(),
        ));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use rstest::rstest;

    use super::*;

    #[test]
    fn rejects_overflowing_rgb_dimensions() {
        // Reject malformed dimensions before decoding, e.g. RGB byte counts larger than u64::MAX.
        assert!(matches!(
            ImageInput::Rgb {
                width: u32::MAX,
                height: u32::MAX,
                pixels: Vec::new(),
            }
            .validate(),
            Err(Error::InvalidRequest(_))
        ));
    }

    #[rstest]
    #[case::empty(&[], 16000)]
    #[case::nonfinite(&[f32::NAN], 16000)]
    #[case::out_of_range(&[1.1], 16000)]
    #[case::sample_rate(&[0.0], 48000)]
    fn rejects_invalid_audio_buffers(#[case] samples: &[f32], #[case] sample_rate: u32) {
        assert!(matches!(
            AudioInput::Samples {
                samples: samples.to_vec(),
                sample_rate
            }
            .validate(),
            Err(Error::InvalidRequest(_))
        ));
    }

    #[test]
    fn validates_buffers_and_decodes_pcm_wave() {
        let root =
            camino::Utf8Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/tiny-d1-omni");
        let values: serde_json::Value =
            serde_json::from_str(include_str!("../tests/fixtures/tiny-d1-omni/mel.json"))
                .expect("frontend reference");
        let pcm: Vec<i16> = values
            .get("samples")
            .and_then(serde_json::Value::as_array)
            .expect("PCM samples")
            .iter()
            .map(|x| i16::try_from(x.as_i64().expect("PCM integer")).expect("PCM16 range"))
            .collect();
        let expected = AudioInput::Pcm16 {
            pcm16: pcm,
            sample_rate: 16000,
        }
        .decode()
        .expect("PCM16 samples");
        let decoded = AudioInput::Path(root.join("speech.wav"))
            .decode()
            .expect("Symphonia WAV decode");
        assert_eq!(decoded, expected);
        let short = AudioInput::Samples {
            samples: vec![0.5],
            sample_rate: 16000,
        }
        .decode()
        .expect("short clip");
        assert_eq!(short.len(), 8000);
        assert_eq!(short.first(), Some(&0.5));
        let long = AudioInput::Samples {
            samples: vec![0.0; 480001],
            sample_rate: 16000,
        }
        .decode()
        .expect("30 second clip cap");
        assert_eq!(long.len(), 480000);
        assert!(matches!(
            ImageInput::Rgb {
                width: 2,
                height: 2,
                pixels: vec![0; 3]
            }
            .validate(),
            Err(Error::InvalidRequest(_))
        ));
        serde_json::from_value::<AudioInput>(
            serde_json::json!({"samples":[0.0],"sample_rate":16000,"channels":2}),
        )
        .expect_err("unknown audio fields");
        AudioInput::Path(root.join("config.json"))
            .decode()
            .expect_err("unsupported file must not become silence");
    }
}
