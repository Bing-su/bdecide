#![cfg(feature = "cpu")]
use bdecide::{
    AutoModel, DecisionModel, Device, Error, LoadOptions, Request,
    hub::{HubOptions, ModelSource},
};
use camino::{Utf8Path, Utf8PathBuf};
use rstest::rstest;
use serde_json::{Value, json};
use wiremock::{
    Mock, MockServer, ResponseTemplate,
    matchers::{method, path},
};

fn fixture(variant: &str) -> Utf8PathBuf {
    Utf8Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures")
        .join(variant)
}
fn request() -> Request {
    serde_json::from_value(
        json!({"state":"alpha", "questions":{"q":{"type":"noul","instructions":"cancel?"}}}),
    )
    .expect("Clef test request must be valid")
}

#[rstest]
#[case("tiny-clef")]
#[case("tiny-clef-flash")]
fn auto_model_dispatches_from_artifacts_and_reuses_weights(#[case] variant: &str) {
    let model = AutoModel::from_pretrained(LoadOptions {
        source: ModelSource::Local(fixture(variant)),
        device: Device::Cpu,
    })
    .unwrap();
    assert_eq!(model.metadata().architecture, "clef");
    let first = serde_json::to_value(model.predict(&request()).unwrap()).unwrap();
    let second = serde_json::to_value(model.predict(&request()).unwrap()).unwrap();
    assert_eq!(first, second);
    assert_eq!(first["usage"]["output_tokens"], 0);
}

#[tokio::test]
async fn sharded_hub_load_is_pinned_and_works_offline() {
    let server = MockServer::start().await;
    let cache = tempfile::tempdir().unwrap();
    let sha = "1234567890123456789012345678901234567890";
    let files = [
        "config.json",
        "joint_head_config.json",
        "joint_head.safetensors",
        "tokenizer.json",
        "tokenizer_config.json",
        "model.safetensors.index.json",
        "model-00001-of-00002.safetensors",
        "model-00002-of-00002.safetensors",
    ];
    // A renamed repository and nested subfolder must still load as Clef. Resolve
    // only config at main; every head/tokenizer/index/shard is pinned to its SHA.
    for (index, file) in files.iter().enumerate() {
        let revision = if index == 0 { "main" } else { sha };
        let response = ResponseTemplate::new(200)
            .insert_header("X-Repo-Commit", sha)
            .insert_header("ETag", format!("\"{}\"", file.replace('.', "-")))
            .set_body_bytes(std::fs::read(fixture("tiny-clef").join(file)).unwrap());
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
    Mock::given(method("HEAD"))
        .and(path(
            "/test/renamed/resolve/main/nested/rl_agent_config.json",
        ))
        .respond_with(ResponseTemplate::new(404).insert_header("X-Error-Code", "EntryNotFound"))
        .expect(1)
        .mount(&server)
        .await;
    let mut options = HubOptions::new("test/renamed");
    options.endpoint = Some(server.uri());
    options.cache_dir = Some(Utf8PathBuf::from_path_buf(cache.path().to_owned()).unwrap());
    options.subfolder = Some("nested".into());
    options.token = bdecide::hub::Token::Anonymous;
    let model = AutoModel::from_pretrained(LoadOptions {
        source: ModelSource::Hub(options.clone()),
        device: Device::Cpu,
    })
    .unwrap();
    let online = serde_json::to_value(model.predict(&request()).unwrap()).unwrap();
    assert_eq!(online["metadata"]["commit_sha"], sha);
    assert_eq!(online["metadata"]["revision"], "main");
    options.local_files_only = true;
    let model = AutoModel::from_pretrained(LoadOptions {
        source: ModelSource::Hub(options),
        device: Device::Cpu,
    })
    .unwrap();
    assert_eq!(
        online,
        serde_json::to_value(model.predict(&request()).unwrap()).unwrap()
    );
    assert!(
        server
            .received_requests()
            .await
            .unwrap()
            .iter()
            .all(|request| !request.headers.contains_key("authorization"))
    );
    server.verify().await;
}

#[test]
fn invalid_head_and_unsafe_index_fail_before_inference() {
    let directory = tempfile::tempdir().unwrap();
    let root = Utf8Path::from_path(directory.path()).unwrap();
    for file in [
        "config.json",
        "joint_head_config.json",
        "joint_head.safetensors",
        "tokenizer.json",
        "tokenizer_config.json",
        "model.safetensors.index.json",
        "model-00001-of-00002.safetensors",
        "model-00002-of-00002.safetensors",
    ] {
        std::fs::copy(fixture("tiny-clef").join(file), root.join(file)).unwrap();
    }
    let load = || {
        AutoModel::from_pretrained(LoadOptions {
            source: ModelSource::Local(root.into()),
            device: Device::Cpu,
        })
    };
    std::fs::write(
        root.join("model.safetensors.index.json"),
        r#"{"weight_map":{"lm_head.weight":"../../outside.safetensors"}}"#,
    )
    .unwrap();
    assert!(matches!(load(), Err(Error::InvalidCheckpoint(_))));
    std::fs::copy(
        fixture("tiny-clef").join("model.safetensors.index.json"),
        root.join("model.safetensors.index.json"),
    )
    .unwrap();
    let mut config: Value =
        serde_json::from_slice(&std::fs::read(root.join("joint_head_config.json")).unwrap())
            .unwrap();
    config["width"] = json!(32);
    std::fs::write(
        root.join("joint_head_config.json"),
        serde_json::to_vec(&config).unwrap(),
    )
    .unwrap();
    assert!(matches!(load(), Err(Error::Weights(_))));
    std::fs::copy(
        fixture("tiny-clef").join("joint_head_config.json"),
        root.join("joint_head_config.json"),
    )
    .unwrap();
    // An index may name existing files while omitting half the actual parameters.
    // Loading only shard one must fail instead of retaining random layer weights.
    let mut index: Value =
        serde_json::from_slice(&std::fs::read(root.join("model.safetensors.index.json")).unwrap())
            .unwrap();
    for file in index["weight_map"].as_object_mut().unwrap().values_mut() {
        *file = json!("model-00001-of-00002.safetensors");
    }
    std::fs::write(
        root.join("model.safetensors.index.json"),
        serde_json::to_vec(&index).unwrap(),
    )
    .unwrap();
    assert!(matches!(load(), Err(Error::Weights(_))));
    std::fs::copy(
        fixture("tiny-clef").join("model.safetensors.index.json"),
        root.join("model.safetensors.index.json"),
    )
    .unwrap();
    std::fs::remove_file(root.join("model-00002-of-00002.safetensors")).unwrap();
    assert!(matches!(load(), Err(Error::MissingArtifact(_))));
}
