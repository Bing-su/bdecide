use std::fs;
#[cfg(feature = "cpu")]
use std::io::Write;
use std::process::Command;
#[cfg(feature = "cpu")]
use std::process::Stdio;

#[cfg(feature = "cpu")]
use bdecide::{AutoModel, Device, Error, LoadOptions, hub::ModelSource};
#[cfg(feature = "cpu")]
use camino::{Utf8Path, Utf8PathBuf};
use rstest::rstest;
use serde_json::{Value, json};
use tempfile::{NamedTempFile, tempdir};
use usage::test::command;

#[rstest]
#[case::help("--help", "Evaluate typed questions")]
#[case::short_help("-h", "predict")]
#[case::version("--version", env!("CARGO_PKG_VERSION"))]
#[case::short_version("-V", env!("CARGO_PKG_VERSION"))]
fn help_and_version_exit_successfully_without_a_model(#[case] arg: &str, #[case] text: &str) {
    let output = command!("bdecide", arg).assert_success();
    assert_eq!(output.stderr_text(), "");
    assert!(output.stdout_text().contains(text));
}

#[rstest]
#[case::long("--help")]
#[case::short("-h")]
fn predict_help_explains_input_without_loading_a_model(#[case] arg: &str) {
    let output = command!("bdecide", "predict", arg).assert_success();
    assert_eq!(output.stderr_text(), "");
    assert!(output.stdout_text().contains("--model"));
    assert!(output.stdout_text().contains("--jsonl"));
    assert!(output.stdout_text().contains("wgpu"));
    assert!(!output.stdout_text().contains("vulkan"));
}

#[test]
#[cfg(not(feature = "wgpu"))]
fn disabled_wgpu_reports_a_device_error_before_resolving_the_model() {
    let input = NamedTempFile::new().expect("input file should be created");
    fs::write(
        input.path(),
        r#"{"state":"alpha","questions":{"q":{"type":"noul","instructions":"cancel?"}}}"#,
    )
    .expect("input file should be written");
    // A missing checkpoint proves the feature error precedes model I/O.
    let output = command!(
        "bdecide",
        "predict".as_ref(),
        "--model".as_ref(),
        "./missing-checkpoint".as_ref(),
        "--device".as_ref(),
        "wgpu".as_ref(),
        "--input".as_ref(),
        input.path().as_os_str(),
    );
    assert_eq!(output.status.code(), Some(1));
    assert_eq!(output.stderr_text(), "");
    let response: Value = serde_json::from_slice(&output.stdout).expect("failure should be JSON");
    assert_eq!(response["error"]["kind"], "device");
    assert!(
        response["error"]["message"]
            .as_str()
            .expect("failure message should be text")
            .contains("--features wgpu")
    );
}

#[test]
fn missing_subcommand_exits_with_a_parse_error() {
    let output = command!("bdecide");
    assert_eq!(output.status.code(), Some(2));
    assert_eq!(output.stdout_text(), "");
    assert!(output.stderr_text().contains("predict"));
}

#[test]
fn missing_model_exits_before_reading_input() {
    let output = command!("bdecide", "predict");
    assert_eq!(output.status.code(), Some(2));
    assert_eq!(output.stdout_text(), "");
    assert!(output.stderr_text().contains("--model"));
}

#[test]
fn invalid_json_returns_a_machine_readable_failure_without_loading_a_model() {
    let input = NamedTempFile::new().expect("input file should be created");
    fs::write(input.path(), "not-json").expect("input file should be written");
    let output = command!(
        "bdecide",
        "predict".as_ref(),
        "--model".as_ref(),
        "unused/model".as_ref(),
        "--input".as_ref(),
        input.path().as_os_str(),
    );
    assert_eq!(output.status.code(), Some(1));
    assert_eq!(output.stderr_text(), "");
    let response: Value = serde_json::from_slice(&output.stdout).expect("failure should be JSON");
    assert_eq!(response["error"]["kind"], "invalid_request");
    assert!(response["error"].get("line").is_none());
}

#[cfg(feature = "cpu")]
#[rstest]
#[case::laya("rl_agent_config.json")]
#[case::encoder("encoder/config.json")]
#[case::tokenizer("tokenizer/tokenizer_config.json")]
fn malformed_checkpoint_json_reports_a_model_error(#[case] artifact: &str) {
    let directory = tempdir().unwrap();
    let root = directory.path();
    let fixture = Utf8Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/tiny-laya");
    // Copy only this model's required artifacts, then damage one JSON file, e.g. encoder/config.json.
    for file in [
        "rl_agent_config.json",
        "encoder/config.json",
        "model.safetensors",
        "tokenizer/tokenizer.json",
        "tokenizer/tokenizer_config.json",
    ] {
        let target = root.join(file);
        fs::create_dir_all(target.parent().unwrap()).unwrap();
        fs::copy(fixture.join(file), target).unwrap();
    }
    fs::write(root.join(artifact), "{").unwrap();
    let input = root.join("request.json");
    fs::write(&input, r#"{"state":"alpha","questions":{}}"#).unwrap();
    let output = Command::new(env!("CARGO_BIN_EXE_bdecide"))
        .args(["predict", "--model"])
        .arg(root)
        .arg("--input")
        .arg(input)
        .output()
        .unwrap();
    assert_eq!(output.status.code(), Some(1));
    assert!(output.stderr.is_empty());
    let response: Value = serde_json::from_slice(&output.stdout).unwrap();
    assert_eq!(response["error"]["kind"], "model");
    assert!(
        response["error"]["message"]
            .as_str()
            .unwrap()
            .contains(artifact)
    );
}

#[cfg(all(feature = "cpu", unix))]
#[test]
fn explicit_anonymous_cache_does_not_require_utf8_home() {
    use std::ffi::OsString;
    use std::os::unix::ffi::OsStringExt;

    let directory = tempdir().unwrap();
    let input = directory.path().join("request.json");
    fs::write(&input, r#"{"state":"alpha","questions":{}}"#).unwrap();
    // Keep the invalid home in a child process; a chosen cache and anonymous auth never need it.
    let output = Command::new(env!("CARGO_BIN_EXE_bdecide"))
        .args(["predict", "--model", "test/laya", "--cache-dir"])
        .arg(directory.path())
        .args(["--anonymous", "--local-files-only", "--input"])
        .arg(input)
        .env("HOME", OsString::from_vec(b"/unused/\xff".to_vec()))
        .env_remove("HF_HOME")
        .env_remove("HF_HUB_CACHE")
        .env_remove("XDG_CACHE_HOME")
        .output()
        .unwrap();
    assert_eq!(output.status.code(), Some(1));
    assert!(output.stderr.is_empty());
    let response: Value = serde_json::from_slice(&output.stdout).unwrap();
    // An offline cache miss proves resolution reached the explicit cache without network I/O.
    assert_eq!(response["error"]["kind"], "hub");
    assert!(
        response["error"]["message"]
            .as_str()
            .unwrap()
            .contains("local cache")
    );
}

#[rstest]
#[case::zero_max_len("max_len", 0, "max_len must be at least 4")]
#[case::short_max_len("max_len", 3, "max_len must be at least 4")]
#[case::zero_head_max_len("head_max_len", 0, "head_max_len must be at least 16")]
#[case::short_head_max_len("head_max_len", 15, "head_max_len must be at least 16")]
fn invalid_budgets_fail_before_model_loading(
    #[case] option: &str,
    #[case] value: usize,
    #[case] message: &str,
    #[values(false, true)] jsonl: bool,
) {
    let directory = tempdir().expect("temporary directory should be created");
    let missing_model = directory.path().join("missing-checkpoint");
    let input = directory.path().join("request.json");
    let request = json!({
        "state": "alpha",
        "questions": {},
        "options": {option: value},
    });
    fs::write(&input, request.to_string()).expect("input file should be written");
    // The absent checkpoint proves invalid budgets precede model I/O, e.g. max_len=3.
    // Explicit truncation must not bypass validation, even for an empty question set.
    let mut command = Command::new(env!("CARGO_BIN_EXE_bdecide"));
    command
        .arg("predict")
        .arg("--model")
        .arg(&missing_model)
        .arg("--input")
        .arg(&input)
        .arg("--truncate");
    if jsonl {
        command.arg("--jsonl");
    }
    let output = command.output().expect("CLI should run");
    assert_eq!(output.status.code(), Some(1));
    assert!(output.stderr.is_empty());
    let response: Value = serde_json::from_slice(&output.stdout).expect("failure should be JSON");
    assert_eq!(response["error"]["kind"], "invalid_request");
    assert_eq!(
        response["error"]["message"],
        format!("invalid request: {message}")
    );
    if jsonl {
        assert_eq!(response["error"]["line"], 1);
    } else {
        assert!(response["error"].get("line").is_none());
    }
}

#[test]
fn missing_input_reports_an_io_failure_on_stderr() {
    let directory = tempdir().expect("temporary directory should be created");
    let missing = directory.path().join("missing.json");
    let output = command!(
        "bdecide",
        "predict".as_ref(),
        "--model".as_ref(),
        "unused/model".as_ref(),
        "--input".as_ref(),
        missing.as_os_str(),
    );
    assert_eq!(output.status.code(), Some(1));
    assert_eq!(output.stdout_text(), "");
    assert!(output.stderr_text().starts_with("bdecide: "));
}

#[cfg(feature = "cpu")]
#[rstest]
#[case::max_len("max_len")]
#[case::head_max_len("head_max_len")]
fn budgets_exceeding_model_positions_are_rejected(#[case] option: &str) {
    let root = Utf8Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/tiny-laya");
    let config: Value =
        serde_json::from_str(include_str!("fixtures/tiny-laya/encoder/config.json"))
            .expect("fixture config should be JSON");
    let positions = config["max_position_embeddings"]
        .as_u64()
        .expect("fixture positions should be an integer");
    let request = json!({
        "state": "alpha",
        "questions": {},
        "options": {option: positions + 1},
    });
    let input = NamedTempFile::new().expect("input file should be created");
    fs::write(input.path(), request.to_string()).expect("input file should be written");
    // Model-specific limits still apply to empty batches, e.g. positions+1 is invalid.
    let output = command!(
        "bdecide",
        "predict".as_ref(),
        "--model".as_ref(),
        root.as_os_str(),
        "--input".as_ref(),
        input.path().as_os_str(),
    );
    assert_eq!(output.status.code(), Some(1));
    assert_eq!(output.stderr_text(), "");
    let response: Value = serde_json::from_slice(&output.stdout).expect("failure should be JSON");
    assert_eq!(response["error"]["kind"], "invalid_request");
    assert_eq!(
        response["error"]["message"],
        format!(
            "invalid request: max_len must be 4..={positions} and head_max_len 16..={positions}"
        )
    );
}

#[cfg(feature = "cpu")]
#[rstest]
#[case::default_device(&[])]
#[case::explicit_cpu_stdin(&["--device", "cpu", "--input", "-"])]
fn jsonl_continues_after_errors_and_keeps_stdout_machine_readable(#[case] extra: &[&str]) {
    let root = Utf8Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/tiny-laya");
    // Exercise piped stdin (e.g. requests.jsonl); command! only captures output.
    let mut child = Command::new(env!("CARGO_BIN_EXE_bdecide"))
        .args(["predict", "--model", root.as_str(), "--jsonl"])
        .args(extra)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .unwrap();
    let request =
        json!({"state":"alpha","questions":{"q":{"type":"noul","instructions":"cancel?"}}});
    let bad = json!({"state":null,"questions":{}});
    // An overflowing RGB size must produce an error row and leave later requests readable.
    let bad_image = json!({
        "state": null, "questions": {},
        "images": [{"width": u32::MAX, "height": u32::MAX, "pixels": []}],
    });
    let mut input = child.stdin.take().unwrap();
    writeln!(input, "not-json\n{request}\n{bad}\n{bad_image}\n{request}").unwrap();
    drop(input);
    let output = child.wait_with_output().unwrap();
    assert_eq!(output.status.code(), Some(1));
    assert!(
        output.stderr.is_empty(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let rows: Vec<Value> = String::from_utf8(output.stdout)
        .unwrap()
        .lines()
        .map(|line| serde_json::from_str(line).unwrap())
        .collect();
    assert_eq!(rows.len(), 5);
    assert_eq!(rows[0]["error"]["line"], 1);
    assert_eq!(rows[2]["error"]["kind"], "invalid_request");
    assert_eq!(rows[3]["error"]["kind"], "invalid_request");
    assert_eq!(rows[3]["error"]["line"], 4);
    assert_eq!(
        rows[3]["error"]["message"],
        "invalid request: RGB pixels must contain width * height * 3 bytes"
    );
    assert_eq!(rows[1], rows[4]);
    assert_eq!(rows[1]["metadata"]["device"], "cpu");
}

#[test]
#[cfg(feature = "cpu")]
fn file_input_supports_auto_device_and_explicit_truncation() {
    let root = Utf8Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/tiny-laya");
    let mut input = NamedTempFile::new().unwrap();
    write!(
        input,
        "{}",
        json!({
            "state": "alpha ".repeat(100),
            "questions": {"q": {"type": "noul", "instructions": "cancel?"}},
        })
    )
    .unwrap();
    let output = command!(
        "bdecide",
        "predict".as_ref(),
        "--model".as_ref(),
        root.as_os_str(),
        "--input".as_ref(),
        input.path().as_os_str(),
        "--device".as_ref(),
        "auto".as_ref(),
        "--truncate".as_ref(),
    )
    .assert_success();
    assert_eq!(output.stderr_text(), "");
    let response: Value = serde_json::from_slice(&output.stdout).unwrap();
    assert_eq!(response["metadata"]["architecture"], "laya");
    assert_eq!(response["usage"]["truncated"], true);
    assert!(response["answers"]["q"]["noul"].is_number());
}

#[test]
#[cfg(feature = "cpu")]
fn local_checkpoint_requires_every_artifact() {
    let directory = tempdir().unwrap();
    let options = LoadOptions {
        source: ModelSource::Local(Utf8PathBuf::try_from(directory.path().to_path_buf()).unwrap()),
        device: Device::Cpu,
    };
    assert!(matches!(
        AutoModel::from_pretrained(options),
        Err(Error::MissingArtifact(_))
    ));
}
