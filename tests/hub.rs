#![cfg(feature = "cpu")]
use std::{
    fs,
    io::Write,
    process::{Command, Output, Stdio},
};

use camino::{Utf8Path, Utf8PathBuf};
use rstest::rstest;
use serde_json::{Value, json};
use tempfile::tempdir;
use wiremock::{
    Mock, MockServer, ResponseTemplate,
    matchers::{method, path},
};

const SHA: &str = "1234567890123456789012345678901234567890";
const FILES: [&str; 5] = [
    "rl_agent_config.json",
    "encoder/config.json",
    "model.safetensors",
    "tokenizer/tokenizer.json",
    "tokenizer/tokenizer_config.json",
];

// Serve exact Hub resolve URLs to verify revision and credential behavior without
// external requests, e.g. main for config and its commit SHA for all other files.
async fn hub_server(prefix: &str) -> MockServer {
    let server = MockServer::start().await;
    for (index, file) in FILES.iter().enumerate() {
        let revision = if index == 0 { "main" } else { SHA };
        let body = fs::read(fixture().join(file)).expect("Hub test setup must succeed");
        let name = Utf8Path::new(file)
            .file_name()
            .expect("artifact has a name");
        let etag = format!("\"{}\"", name.replace('.', "-"));
        let response = ResponseTemplate::new(200)
            .insert_header("X-Repo-Commit", SHA)
            .insert_header("ETag", etag)
            .set_body_bytes(body);
        for verb in ["HEAD", "GET"] {
            Mock::given(method(verb))
                .and(path(format!(
                    "/test/laya/resolve/{revision}/{prefix}{file}"
                )))
                .respond_with(response.clone())
                // Initial and forced loads each fetch every artifact exactly once.
                .expect(2)
                .mount(&server)
                .await;
        }
    }
    server
}

fn fixture() -> Utf8PathBuf {
    Utf8Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/tiny-laya")
}

fn invoke(cache: &Utf8Path, endpoint: &str, subfolder: &str, extra: &[&str]) -> Output {
    // Keep Hub settings per child, e.g. HF_TOKEN; command! has no environment overrides.
    let mut child = Command::new(env!("CARGO_BIN_EXE_bdecide"))
        .args([
            "predict",
            "--model",
            "test/laya",
            "--subfolder",
            subfolder,
            "--cache-dir",
            cache.as_str(),
            "--anonymous",
        ])
        .args(extra)
        .env("HF_ENDPOINT", endpoint)
        .env("HF_TOKEN", "must-not-be-sent")
        .env("HF_HUB_OFFLINE", "0")
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .expect("Hub test setup must succeed");
    writeln!(
        child.stdin.take().expect("child stdin is piped"),
        "{}",
        json!({"state":"alpha","questions":{"q":{"type":"noul","instructions":"cancel?"}}})
    )
    .expect("Hub test setup must succeed");
    child
        .wait_with_output()
        .expect("Hub test setup must succeed")
}

// Exercise names that resemble cache directories or globs, e.g. snapshots and v[1].
#[rstest]
#[case::root("")]
#[case::nested("nested")]
#[case::cache_name("snapshots")]
#[case::nested_cache_name("nested/snapshots/variant")]
#[case::glob_name("v[1]")]
#[tokio::test]
async fn hub_pins_files_and_supports_python_cache_and_anonymous_auth(#[case] subfolder: &str) {
    let directory = tempdir().expect("Hub test setup must succeed");
    // Exercise Unicode and spaces through CLI parsing and Hub cache I/O.
    let cache = Utf8Path::from_path(directory.path())
        .expect("temporary path must be UTF-8")
        .join("모델 캐시");
    let prefix = if subfolder.is_empty() {
        String::new()
    } else {
        format!("{subfolder}/")
    };
    let server = hub_server(&prefix).await;
    let endpoint = server.uri();
    let online = invoke(&cache, &endpoint, subfolder, &[]);
    assert!(
        online.status.success(),
        "{} {}",
        String::from_utf8_lossy(&online.stdout),
        String::from_utf8_lossy(&online.stderr)
    );
    let response: Value =
        serde_json::from_slice(&online.stdout).expect("Hub test setup must succeed");
    assert_eq!(response["metadata"]["commit_sha"], SHA);
    assert_eq!(response["metadata"]["subfolder"], subfolder);
    let requests = server.received_requests().await.unwrap();
    assert!(!requests.is_empty());
    assert!(
        requests
            .iter()
            .all(|r| !r.headers.contains_key("authorization"))
    );
    // Known artifacts need only resolve URLs, e.g. no repository tree listing.
    assert!(requests.iter().all(|r| !r.url.path().starts_with("/api/")));
    assert!(
        requests
            .iter()
            .filter(|r| r.url.path().contains("/resolve/main/"))
            .all(|r| r.url.path().ends_with("rl_agent_config.json"))
    );
    assert!(
        requests
            .iter()
            .any(|r| r.url.path() == format!("/test/laya/resolve/{SHA}/{prefix}model.safetensors"))
    );
    let offline = invoke(&cache, &endpoint, subfolder, &["--local-files-only"]);
    assert!(
        offline.status.success(),
        "{}",
        String::from_utf8_lossy(&offline.stdout)
    );
    let pinned = invoke(&cache, &endpoint, subfolder, &["--revision", SHA]);
    assert!(pinned.status.success());
    assert_eq!(
        server.received_requests().await.unwrap().len(),
        requests.len()
    );
    fs::remove_file(cache.join(format!(
        "models--test--laya/snapshots/{SHA}/{prefix}tokenizer/tokenizer.json"
    )))
    .expect("Hub test setup must succeed");
    let incomplete = invoke(&cache, &endpoint, subfolder, &["--local-files-only"]);
    assert!(!incomplete.status.success());
    assert!(String::from_utf8_lossy(&incomplete.stdout).contains("tokenizer/tokenizer.json"));
    let conflicting = invoke(
        &cache,
        &endpoint,
        subfolder,
        &["--local-files-only", "--force-download"],
    );
    assert!(!conflicting.status.success());
    assert_eq!(
        server.received_requests().await.unwrap().len(),
        requests.len()
    );

    // A forced load must refresh all artifacts, e.g. even already cached weights.
    let forced = invoke(&cache, &endpoint, subfolder, &["--force-download"]);
    assert!(
        forced.status.success(),
        "{} {}",
        String::from_utf8_lossy(&forced.stdout),
        String::from_utf8_lossy(&forced.stderr)
    );
    let refreshed = server.received_requests().await.unwrap();
    assert!(
        refreshed
            .iter()
            .all(|r| !r.headers.contains_key("authorization"))
    );
    let refreshed = refreshed
        .get(requests.len()..)
        .expect("earlier requests remain recorded");
    for (index, file) in FILES.iter().enumerate() {
        let revision = if index == 0 { "main" } else { SHA };
        let target = format!("/test/laya/resolve/{revision}/{prefix}{file}");
        assert_eq!(
            refreshed
                .iter()
                .filter(|r| r.method == "GET" && r.url.path() == target)
                .count(),
            1,
            "each artifact must be downloaded once: {target}"
        );
    }
    server.verify().await;
}
