#![cfg(test)]

use std::fs;

use approx::abs_diff_eq;
use bdecide::hub::ModelSource;
use bdecide::{
    AutoModel,
    DecisionModel,
    Device,
    LoadOptions,
    Question,
    Qwen3_5ForCausalLM,
    Qwen3_5TextConfig,
    Qwen3_5TextModel,
    Request,
};
#[cfg(feature = "cpu")]
use bdecide::{
    Error,
    Qwen3_5Config,
    Truncation,
    hub::{HubOptions, Token},
};
#[cfg(feature = "cpu")]
use burn::backend::Flex;
#[cfg(feature = "wgpu")]
use burn::backend::Wgpu;
use burn::tensor::backend::Backend;
use burn::tensor::{Int, Tensor, TensorData};
use camino::{Utf8Path, Utf8PathBuf};
#[cfg(feature = "cpu")]
use indexmap::IndexMap;
use rstest::rstest;
#[cfg(feature = "cpu")]
use serde_json::Map;
use serde_json::Value;
#[cfg(feature = "cpu")]
use serde_json::json;
use tempfile::tempdir;

#[rstest]
#[case("tiny-vev-4b", true, 2560)]
#[case("tiny-vev-9b", false, 4096)]
#[case("tiny-wald", true, 2560)]
fn reads_pinned_release_dimensions(
    #[case] variant: &str,
    #[case] tied: bool,
    #[case] hidden: usize,
) {
    let directory = tempdir().unwrap();
    let root = Utf8Path::from_path(directory.path()).unwrap();
    fs::copy(
        fixture(variant).join("release-config.json"),
        root.join("config.json"),
    )
    .unwrap();
    let config = Qwen3_5TextConfig::from_pretrained(root).unwrap();
    assert_eq!(config.tie_word_embeddings, tied);
    assert_eq!(config.model_type, "qwen3_5_text");
    assert_eq!(config.hidden_size, hidden);
}

// Exercise the public vocabulary output while comparing selected upstream answer rows.
// For example decision adapters use A/B, but forward still returns the entire vocabulary.
fn selected_logits<B: Backend>(
    model: &Qwen3_5ForCausalLM<B>,
    input: &[u32],
    answers: &[u32],
    device: &B::Device,
) -> Vec<f32> {
    let ids = Tensor::from_data(TensorData::new(input.to_vec(), [1, input.len()]), device);
    let output = model.forward(ids, 1).unwrap();
    let answer_ids =
        Tensor::<B, 1, Int>::from_data(TensorData::new(answers.to_vec(), [answers.len()]), device);
    output
        .logits
        .select(2, answer_ids)
        .into_data()
        .to_vec()
        .unwrap()
}

fn assert_tensor<B: Backend>(actual: Tensor<B, 3>, expected: &Value) {
    let expected: Vec<Vec<Vec<f32>>> = serde_json::from_value(expected.clone()).unwrap();
    assert_eq!(
        actual.dims(),
        [expected.len(), expected[0].len(), expected[0][0].len()]
    );
    let values = actual.into_data().to_vec::<f32>().unwrap();
    for (actual, expected) in values
        .into_iter()
        .zip(expected.into_iter().flatten().flatten())
    {
        assert!(
            abs_diff_eq!(actual, expected, epsilon = 5e-4),
            "{actual} != {expected}"
        );
    }
}

fn verify_native_forward<B: Backend>(
    model: &Qwen3_5ForCausalLM<B>,
    root: &Utf8Path,
    reference: &Value,
    device: &B::Device,
) {
    let native = &reference["causal_lm"];
    let inputs: Vec<Vec<u32>> = serde_json::from_value(native["input_ids"].clone()).unwrap();
    let shape = [inputs.len(), inputs[0].len()];
    let ids = Tensor::from_data(
        TensorData::new(inputs.into_iter().flatten().collect::<Vec<_>>(), shape),
        device,
    );
    let tokens: Vec<u32> = serde_json::from_value(native["token_ids"].clone()).unwrap();
    let tokens =
        Tensor::<B, 1, Int>::from_data(TensorData::new(tokens.clone(), [tokens.len()]), device);
    // The keep count slices sequence positions, never vocabulary rows; zero means every position.
    // For example keeping more positions than exist returns the full sequence, like Python slicing.
    for keep in [0, 1, 2, 8] {
        let output = model.forward(ids.clone(), keep).unwrap();
        let start = if keep == 0 {
            0
        } else {
            shape[1].saturating_sub(keep)
        };
        assert_eq!(
            output.logits.dims(),
            [shape[0], shape[1] - start, model.config.vocab_size]
        );
        let expected = Value::Array(
            native["logits"]
                .as_array()
                .unwrap()
                .iter()
                .map(|batch| Value::Array(batch.as_array().unwrap()[start..].to_vec()))
                .collect(),
        );
        assert_tensor(output.logits.select(2, tokens.clone()), &expected);
    }
    let base = Qwen3_5TextModel::<B>::from_pretrained(root, device).unwrap();
    assert_eq!(base.config.model_type, "qwen3_5_text");
    assert_tensor(base.forward(ids.clone()), &native["last_hidden_state"]);
    let defaults = model.forward_builder().input_ids(ids).call().unwrap();
    assert_tensor(defaults.logits.select(2, tokens), &native["logits"]);
}

fn fixture(variant: &str) -> Utf8PathBuf {
    Utf8Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures")
        .join(variant)
}

fn compare(actual: &Value, expected: &Value) {
    match expected {
        Value::Object(fields) => {
            for (key, value) in fields {
                compare(&actual[key], value);
            }
        }
        Value::Number(number) => assert!(
            abs_diff_eq!(
                actual.as_f64().unwrap(),
                number.as_f64().unwrap(),
                epsilon = 5e-4
            ),
            "{actual} != {expected}"
        ),
        _ => assert_eq!(actual, expected),
    }
}

fn verify<B: Backend>(variant: &str, device: Device, backend_device: &B::Device) {
    let root = fixture(variant);
    let model = AutoModel::from_pretrained(LoadOptions {
        source: ModelSource::Local(root.clone()),
        device,
    })
    .expect("reference fixtures and predictions must be valid");
    let expected_family = if variant == "tiny-wald" {
        "wald"
    } else {
        "vev"
    };
    assert_eq!(model.metadata().architecture, expected_family);
    let reference: Value = serde_json::from_slice(&fs::read(root.join("reference.json")).unwrap())
        .expect("reference fixtures and predictions must be valid");
    let backbone = Qwen3_5ForCausalLM::<B>::from_pretrained(&root, backend_device)
        .expect("reference fixtures and predictions must be valid");
    verify_native_forward(&backbone, &root, &reference, backend_device);
    // All seven boundary cases run on CPU. Keep software-GPU parity focused on
    // the mixed choice/score/noul request; large option counts exercise the same readout.
    let case_limit = if matches!(device, Device::Cpu) {
        usize::MAX
    } else {
        1
    };
    for case in reference["cases"]
        .as_array()
        .expect("reference fixtures and predictions must be valid")
        .iter()
        .take(case_limit)
    {
        let request: Request = serde_json::from_value(case["request"].clone())
            .expect("reference fixtures and predictions must be valid");
        let response = model
            .predict(&request)
            .expect("reference fixtures and predictions must be valid");
        let actual = serde_json::to_value(&response)
            .expect("reference fixtures and predictions must be valid");
        // Upstream answer parity needs identical prompt text; JSON spelling may change tokens.
        // Still check every upstream token sequence below, including structured-state cases.
        let mut text_only = request.state.is_string();
        for (id, question) in &request.questions {
            let text_criteria = match question {
                Question::Choice { criteria, .. } | Question::Noul { criteria, .. } => criteria
                    .values()
                    .all(|value| value.is_string() || value.is_null()),
                Question::Score { criteria, .. } => criteria.iter().all(Value::is_string),
            };
            if request.state.is_string() && text_criteria {
                compare(&actual["answers"][id], &case["answers"][id]);
            }
            text_only &= text_criteria;
        }
        assert_eq!(response.answers.len(), request.questions.len());
        assert_eq!(response.usage.output_tokens, 0);
        let mut input_tokens = 0;
        for row in case["rows"]
            .as_array()
            .expect("reference fixtures and predictions must be valid")
        {
            let input_ids: Vec<u32> = serde_json::from_value(row["input_ids"].clone())
                .expect("reference fixtures and predictions must be valid");
            let answer_ids: Vec<u32> = serde_json::from_value(row["answer_ids"].clone())
                .expect("reference fixtures and predictions must be valid");
            let expected: Vec<f32> = serde_json::from_value(row["logits"].clone())
                .expect("reference fixtures and predictions must be valid");
            let logits = selected_logits(&backbone, &input_ids, &answer_ids, backend_device);
            for (actual, expected) in logits.into_iter().zip(expected) {
                assert!(
                    abs_diff_eq!(actual, expected, epsilon = 5e-4),
                    "{variant}: {actual} != {expected}"
                );
            }
            input_tokens += input_ids.len();
        }
        if text_only {
            assert_eq!(response.usage.input_tokens, input_tokens);
        }
    }
}

#[cfg(feature = "cpu")]
#[rstest]
#[case("tiny-vev-4b")]
#[case("tiny-vev-9b")]
#[case("tiny-wald")]
fn cpu_matches_upstream_text_decisions(#[case] variant: &str) {
    verify::<Flex>(variant, Device::Cpu, &Default::default());
}

#[cfg(feature = "wgpu")]
#[rstest]
#[case("tiny-vev-4b")]
#[case("tiny-vev-9b")]
#[case("tiny-wald")]
#[ignore = "requires a wgpu adapter"]
fn wgpu_matches_upstream_text_decisions(#[case] variant: &str) {
    verify::<Wgpu<f32, i32>>(variant, Device::Wgpu, &Default::default());
}

#[cfg(feature = "cpu")]
#[rstest]
#[case("tiny-vev-4b")]
#[case("tiny-wald")]
fn validates_budgets_and_reports_state_loss(#[case] variant: &str) {
    let model = AutoModel::from_pretrained(LoadOptions {
        source: ModelSource::Local(fixture(variant)),
        device: Device::Cpu,
    })
    .unwrap();
    let mut request: Request = serde_json::from_value(json!({"state":"alpha ".repeat(1000), "questions":{"q":{"type":"noul","instructions":"cancel?"}}})).unwrap();
    request.options.max_len = Some(700);
    assert!(matches!(
        model.predict(&request),
        Err(Error::InvalidRequest(_))
    ));
    request.options.truncation = Truncation::Truncate;
    let response = model.predict(&request).unwrap();
    assert_eq!(response.usage.input_tokens, 700);
    assert!(response.usage.truncated);
    assert!(response.usage.state_tokens_dropped > 0);
    assert_eq!(response.usage.truncated_questions, ["q"]);
    request.options.max_len = Some(4);
    assert!(matches!(
        model.predict(&request),
        Err(Error::InvalidRequest(_))
    ));
    request.options.max_len = None;
    request.options.head_max_len = Some(16);
    assert!(matches!(
        model.predict(&request),
        Err(Error::InvalidRequest(_))
    ));
    request.options.head_max_len = None;
    request.questions.clear();
    assert!(matches!(
        model.predict(&request),
        Err(Error::InvalidRequest(_))
    ));
}

#[cfg(feature = "cpu")]
#[rstest]
#[case("tiny-vev-4b")]
#[case("tiny-vev-9b")]
#[case("tiny-wald")]
#[tokio::test]
async fn hub_protocol_artifacts_stay_pinned_and_load_offline(#[case] variant: &str) {
    use wiremock::matchers::{method, path};
    use wiremock::{Mock, MockServer, ResponseTemplate};
    let server = MockServer::start().await;
    let cache = tempdir().unwrap();
    let sha = "1234567890123456789012345678901234567890";
    let mut files = vec!["config.json", "tokenizer.json", "tokenizer_config.json"];
    if variant == "tiny-wald" {
        files.extend([
            "serving.json",
            "temperature.json",
            "model.safetensors.index.json",
            "model-00001-of-00002.safetensors",
            "model-00002-of-00002.safetensors",
        ]);
    } else {
        files.extend(["vev.json", "model.safetensors"]);
    }
    for file in files {
        let revision = if file == "config.json" { "main" } else { sha };
        let response = ResponseTemplate::new(200)
            .insert_header("X-Repo-Commit", sha)
            .insert_header("ETag", format!("\"{}\"", file.replace('.', "-")))
            .set_body_bytes(fs::read(fixture(variant).join(file)).unwrap());
        for verb in ["HEAD", "GET"] {
            Mock::given(method(verb))
                .and(path(format!(
                    "/test/renamed/resolve/{revision}/nested/{file}"
                )))
                .respond_with(response.clone())
                .expect(1)
                .mount(&server)
                .await;
        }
    }
    let mut absent = vec![
        ("main", "rl_agent_config.json"),
        (sha, "joint_head_config.json"),
    ];
    absent.push((
        sha,
        if variant == "tiny-wald" {
            "vev.json"
        } else {
            "model.safetensors.index.json"
        },
    ));
    for (revision, file) in absent {
        Mock::given(method("HEAD"))
            .and(path(format!(
                "/test/renamed/resolve/{revision}/nested/{file}"
            )))
            .respond_with(ResponseTemplate::new(404).insert_header("X-Error-Code", "EntryNotFound"))
            .expect(1)
            .mount(&server)
            .await;
    }
    let mut options = HubOptions::new("test/renamed");
    options.endpoint = Some(server.uri());
    options.cache_dir = Some(Utf8Path::from_path(cache.path()).unwrap().into());
    options.subfolder = Some("nested".into());
    options.token = Token::Anonymous;
    let request: Request = serde_json::from_value(
        json!({"state":"alpha", "questions":{"q":{"type":"noul","instructions":"cancel?"}}}),
    )
    .unwrap();
    let load = |options| {
        AutoModel::from_pretrained(LoadOptions {
            source: ModelSource::Hub(options),
            device: Device::Cpu,
        })
        .unwrap()
    };
    let online = serde_json::to_value(load(options.clone()).predict(&request).unwrap()).unwrap();
    assert_eq!(online["metadata"]["commit_sha"], sha);
    assert_eq!(online["metadata"]["revision"], "main");
    options.local_files_only = true;
    assert_eq!(
        online,
        serde_json::to_value(load(options).predict(&request).unwrap()).unwrap()
    );
    server.verify().await;
}

#[cfg(feature = "cpu")]
#[rstest]
#[case::head_only("head_only")]
#[case::both_equal("equal")]
#[case::both_different("different")]
#[case::text_only_prefix("flat")]
#[case::missing_tied_pair("missing")]
#[case::non_finite("nan")]
#[case::wrong_dtype("dtype")]
#[case::missing_untied_head("untied_missing")]
fn transformers_embedding_aliases_preserve_strict_loading(#[case] variant: &str) {
    let fixture_name = if variant == "untied_missing" {
        "tiny-vev-9b"
    } else {
        "tiny-vev-4b"
    };
    let directory = tempdir().unwrap();
    let root = Utf8Path::from_path(directory.path()).unwrap();
    fs::copy(
        fixture(fixture_name).join("config.json"),
        root.join("config.json"),
    )
    .unwrap();
    let bytes = fs::read(fixture(fixture_name).join("model.safetensors")).unwrap();
    let header_len = usize::try_from(u64::from_le_bytes(bytes[..8].try_into().unwrap())).unwrap();
    let header: Value = serde_json::from_slice(&bytes[8..8 + header_len]).unwrap();
    let body = &bytes[8 + header_len..];
    let mut tensors: IndexMap<String, (Value, Vec<u8>)> = header
        .as_object()
        .unwrap()
        .iter()
        .filter(|(name, _)| *name != "__metadata__")
        .map(|(name, metadata)| {
            let start = usize::try_from(metadata["data_offsets"][0].as_u64().unwrap()).unwrap();
            let end = usize::try_from(metadata["data_offsets"][1].as_u64().unwrap()).unwrap();
            (name.clone(), (metadata.clone(), body[start..end].to_vec()))
        })
        .collect();
    let embed = "model.language_model.embed_tokens.weight";
    match variant {
        "head_only" => {
            let data = tensors.shift_remove(embed).unwrap();
            tensors.insert("lm_head.weight".into(), data);
        }
        "equal" | "different" => {
            let mut data = tensors[embed].clone();
            if variant == "different" {
                data.1[..2].copy_from_slice(&[0x80, 0x3f]);
            }
            tensors.insert("lm_head.weight".into(), data);
        }
        "flat" => {
            tensors = tensors
                .into_iter()
                .map(|(name, data)| (name.replace("model.language_model.", "model."), data))
                .collect();
        }
        "missing" => {
            tensors.shift_remove(embed);
        }
        "nan" => {
            tensors.get_mut(embed).unwrap().1[..2].copy_from_slice(&[0xc0, 0x7f]);
        }
        "dtype" => {
            tensors.get_mut(embed).unwrap().0["dtype"] = json!("I16");
        }
        "untied_missing" => {
            tensors.shift_remove("lm_head.weight");
        }
        _ => panic!("unknown checkpoint test variant: {variant}"),
    }
    let mut data = Vec::new();
    let mut header = Map::new();
    for (name, (mut metadata, bytes)) in tensors {
        metadata["data_offsets"] = json!([data.len(), data.len() + bytes.len()]);
        header.insert(name, metadata);
        data.extend(bytes);
    }
    let mut header = serde_json::to_vec(&header).unwrap();
    header.resize(header.len().next_multiple_of(8), b' ');
    let mut output = (header.len() as u64).to_le_bytes().to_vec();
    output.extend(header);
    output.extend(data);
    fs::write(root.join("model.safetensors"), output).unwrap();
    let load = || Qwen3_5ForCausalLM::<Flex>::from_pretrained(root, &Default::default());
    if matches!(variant, "missing" | "nan" | "dtype" | "untied_missing") {
        assert!(matches!(load(), Err(Error::Weights(_))));
        return;
    }
    let actual = selected_logits(&load().unwrap(), &[2, 3, 4], &[0, 5], &Default::default());
    let expected = if variant == "different" {
        // Transformers keeps both unequal tensors even with tie_word_embeddings=true.
        // Loading the same values with an explicit untied config must produce identical logits.
        let mut config: Value =
            serde_json::from_slice(&fs::read(root.join("config.json")).unwrap()).unwrap();
        config["tie_word_embeddings"] = json!(false);
        config["text_config"]["tie_word_embeddings"] = json!(false);
        fs::write(
            root.join("config.json"),
            serde_json::to_vec(&config).unwrap(),
        )
        .unwrap();
        load().unwrap()
    } else {
        Qwen3_5ForCausalLM::<Flex>::from_pretrained(&fixture(fixture_name), &Default::default())
            .unwrap()
    };
    let expected = selected_logits(&expected, &[2, 3, 4], &[0, 5], &Default::default());
    for (actual, expected) in actual.into_iter().zip(expected) {
        assert!(abs_diff_eq!(actual, expected, epsilon = 1e-6));
    }
}

#[cfg(feature = "cpu")]
#[rstest]
#[case("tiny-vev-4b", false, true)]
#[case("tiny-vev-9b", true, false)]
fn causal_lm_extracts_text_config_without_inheriting_wrapper_tying(
    #[case] variant: &str,
    #[case] root_tied: bool,
    #[case] text_tied: bool,
) {
    let directory = tempdir().unwrap();
    let root = Utf8Path::from_path(directory.path()).unwrap();
    let mut config: Value =
        serde_json::from_slice(&fs::read(fixture(variant).join("config.json")).unwrap()).unwrap();
    config["tie_word_embeddings"] = json!(root_tied);
    fs::write(
        root.join("config.json"),
        serde_json::to_vec(&config).unwrap(),
    )
    .unwrap();
    fs::copy(
        fixture(variant).join("model.safetensors"),
        root.join("model.safetensors"),
    )
    .unwrap();
    let text = Qwen3_5TextConfig::from_pretrained(root).unwrap();
    let wrapper = Qwen3_5Config::from_pretrained(root).unwrap();
    assert_eq!(text.tie_word_embeddings, text_tied);
    assert_eq!(wrapper.tie_word_embeddings, root_tied);
    // Wrapper and text settings are independent, e.g. constructing a wrapper keeps its false default.
    assert!(!Qwen3_5Config::new(text).tie_word_embeddings);
    let model = Qwen3_5ForCausalLM::<Flex>::from_pretrained(root, &Default::default()).unwrap();
    assert_eq!(model.config.tie_word_embeddings, text_tied);
    assert_eq!(
        model.get_input_embeddings().weight.id == model.get_output_embeddings().weight.id,
        text_tied
    );
}
