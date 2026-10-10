use std::fs::File;
use std::io::{self, BufRead, BufReader, BufWriter, ErrorKind, Write};
use std::process::ExitCode;

#[cfg(any(feature = "cpu", feature = "wgpu"))]
use bdecide::Device;
use bdecide::hub::{HubOptions, ModelSource, Token};
use bdecide::{
    AutoModel,
    DecisionModel,
    Error,
    LoadOptions,
    Request,
    Response,
    Result,
    Truncation,
};
use camino::Utf8PathBuf;
use serde::Serialize;
use usage::{Args, Cli, Run, Subcommands, ValueEnum};

/// Evaluate typed questions with a locally executed decision model.
#[derive(Cli, Debug)]
#[usage(bin = "bdecide", version, unknown_flags = "error", run)]
pub(crate) struct Bdecide {
    #[usage(subcommand)]
    command: Commands,
}

#[derive(Subcommands, Debug)]
#[usage(run)]
pub(crate) enum Commands {
    Predict(Predict),
}

/// Evaluate typed questions using a model and return JSON or JSONL responses.
#[derive(Args, Debug)]
#[usage(
    args_override_self = false,
    after_help = "Example: bdecide predict --model convaiinnovations/laya-multilingual --input request.json\nStream: bdecide predict --model ./checkpoint --jsonl < requests.jsonl"
)]
pub(crate) struct Predict {
    /// Hugging Face owner/repo or an existing local checkpoint directory.
    #[usage(long)]
    model: String,
    /// Read requests from this file; omit or use '-' for stdin.
    #[usage(long)]
    input: Option<Utf8PathBuf>,
    /// Return one response or error per input line while reusing the model.
    #[usage(long)]
    jsonl: bool,
    #[usage(long, value_enum, default = "cpu")]
    device: CliDevice,
    #[usage(long)]
    revision: Option<String>,
    #[usage(long)]
    subfolder: Option<String>,
    #[usage(long)]
    cache_dir: Option<Utf8PathBuf>,
    #[usage(long)]
    local_files_only: bool,
    #[usage(long)]
    force_download: bool,
    #[usage(long, conflicts = "require_token")]
    anonymous: bool,
    #[usage(long)]
    require_token: bool,
    /// Explicitly allow the model's reference truncation policy.
    #[usage(long)]
    truncate: bool,
}

// Keep CLI value metadata at the binary boundary; library devices stay parser-independent.
#[derive(ValueEnum, Debug)]
enum CliDevice {
    Cpu,
    Wgpu,
    Auto,
}

#[derive(Serialize)]
struct Failure {
    error: FailureDetail,
}
#[derive(Serialize)]
struct FailureDetail {
    kind: &'static str,
    message: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    line: Option<usize>,
}

impl Run for Predict {
    type Output = ExitCode;

    fn run(self) -> Self::Output {
        match execute(self) {
            Ok(false) => ExitCode::SUCCESS,
            Ok(true) => ExitCode::FAILURE,
            Err(error) if error.kind() == ErrorKind::BrokenPipe => ExitCode::SUCCESS,
            Err(error) => {
                eprintln!("bdecide: {error}");
                ExitCode::FAILURE
            }
        }
    }
}

fn execute(args: Predict) -> io::Result<bool> {
    let mut input: Box<dyn BufRead> = match &args.input {
        Some(path) if path.as_str() != "-" => Box::new(BufReader::new(File::open(path)?)),
        _ => Box::new(BufReader::new(io::stdin())),
    };
    let mut output = BufWriter::new(io::stdout().lock());
    let mut model = None;
    let mut had_errors = false;
    if args.jsonl {
        for (index, line) in input.lines().enumerate() {
            had_errors |= respond(&args, &mut model, &line?, Some(index + 1), &mut output)?;
            output.flush()?;
        }
    } else {
        let mut text = String::new();
        input.read_to_string(&mut text)?;
        had_errors = respond(&args, &mut model, &text, None, &mut output)?;
    }
    output.flush()?;
    Ok(had_errors)
}

fn respond(
    args: &Predict,
    model: &mut Option<AutoModel>,
    text: &str,
    line: Option<usize>,
    output: &mut impl Write,
) -> io::Result<bool> {
    let failed = match predict_request(args, model, text) {
        Ok(response) => {
            serde_json::to_writer(&mut *output, &response)?;
            false
        }
        Err(error) => {
            let kind = match error {
                Error::Json(_) | Error::InvalidRequest(_) => "invalid_request",
                Error::Device(_) => "device",
                Error::Hub(_) => "hub",
                _ => "model",
            };
            serde_json::to_writer(
                &mut *output,
                &Failure {
                    error: FailureDetail {
                        kind,
                        message: error.to_string(),
                        line,
                    },
                },
            )?;
            true
        }
    };
    writeln!(output)?;
    Ok(failed)
}

fn predict_request(args: &Predict, model: &mut Option<AutoModel>, text: &str) -> Result<Response> {
    let mut request: Request = serde_json::from_str(text)?;
    if args.truncate {
        request.options.truncation = Truncation::Truncate;
    }
    // Validate before loading so malformed JSONL lines never trigger downloads.
    request.validate()?;
    let model = match model {
        Some(model) => model,
        None => model.insert(AutoModel::from_pretrained(load_options(args)?)?),
    };
    model.system_one(&request)
}

fn load_options(args: &Predict) -> Result<LoadOptions> {
    let device = match args.device {
        CliDevice::Cpu => {
            #[cfg(feature = "cpu")]
            {
                Some(Device::flex())
            }
            #[cfg(not(feature = "cpu"))]
            return Err(Error::Device(
                "CPU backend is disabled; enable cpu or select wgpu".into(),
            ));
        }
        CliDevice::Wgpu => {
            #[cfg(feature = "wgpu")]
            {
                Some(Device::wgpu(Default::default()))
            }
            #[cfg(not(feature = "wgpu"))]
            return Err(Error::Device("rebuild with --features wgpu".into()));
        }
        CliDevice::Auto => None,
    };
    let path = Utf8PathBuf::from(&args.model);
    let source = if path.is_dir() || path.is_absolute() || args.model.starts_with('.') {
        ModelSource::Local(match &args.subfolder {
            Some(folder) => path.join(folder),
            None => path,
        })
    } else {
        let mut options = HubOptions::new(&args.model);
        options.revision = args.revision.clone();
        options.subfolder = args.subfolder.clone();
        options.cache_dir = args.cache_dir.clone();
        options.local_files_only = args.local_files_only;
        options.force_download = args.force_download;
        options.token = if args.anonymous {
            Token::Anonymous
        } else if args.require_token {
            Token::Required
        } else {
            Token::Auto
        };
        ModelSource::Hub(options)
    };
    Ok(LoadOptions { source, device })
}

#[cfg(test)]
mod tests {
    use rstest::rstest;
    use usage::test::{self as harness, Outcome, Page};

    use super::*;

    #[test]
    #[cfg(feature = "cpu")]
    fn predict_defaults_to_cpu_and_automatic_auth() {
        let words = harness::argv(["predict", "--model", "owner/repo"]);
        let cli = harness::parse(Bdecide::spec(), &words.words(), Bdecide::parse_from)
            .expect("predict arguments should parse");
        let Commands::Predict(args) = cli.command;
        assert!(args.input.is_none());
        assert!(!args.jsonl);
        assert!(!args.truncate);
        let options = load_options(&args).unwrap();
        assert_eq!(options.device, Some(Device::flex()));
        let ModelSource::Hub(hub) = options.source else {
            panic!("owner/repo should select the Hub");
        };
        assert!(matches!(hub.token, Token::Auto));
    }

    #[rstest]
    #[case::anonymous("--anonymous", Token::Anonymous)]
    #[case::required("--require-token", Token::Required)]
    fn predict_preserves_hub_options(#[case] auth: &str, #[case] token: Token) {
        let words = harness::argv([
            "predict",
            "--model",
            "owner/repo",
            "--device",
            "wgpu",
            "--revision",
            "commit",
            "--subfolder",
            "nested",
            "--cache-dir",
            "cache",
            "--local-files-only",
            "--force-download",
            "--jsonl",
            "--truncate",
            auth,
        ]);
        let cli = harness::parse(Bdecide::spec(), &words.words(), Bdecide::parse_from)
            .expect("predict arguments should parse");
        let Commands::Predict(args) = cli.command;
        assert!(args.jsonl);
        assert!(args.truncate);
        #[cfg(not(feature = "wgpu"))]
        assert!(matches!(load_options(&args), Err(Error::Device(_))));
        #[cfg(feature = "wgpu")]
        assert_eq!(
            load_options(&args).unwrap().device,
            Some(Device::wgpu(Default::default()))
        );
        // Inspect Hub policy without creating a disabled backend, e.g. CPU-only builds.
        let args = Predict {
            device: CliDevice::Auto,
            ..args
        };
        let options = load_options(&args).unwrap();
        assert!(options.device.is_none());
        let ModelSource::Hub(hub) = options.source else {
            panic!("owner/repo should select the Hub");
        };
        assert_eq!(hub.revision.as_deref(), Some("commit"));
        assert_eq!(hub.subfolder.as_deref(), Some("nested"));
        assert_eq!(hub.cache_dir, Some(Utf8PathBuf::from("cache")));
        assert!(hub.local_files_only);
        assert!(hub.force_download);
        assert!(matches!(
            (hub.token, token),
            (Token::Anonymous, Token::Anonymous) | (Token::Required, Token::Required)
        ));
    }

    #[test]
    fn predict_resolves_local_subfolder_and_input() {
        let words = harness::argv([
            "predict",
            "--model",
            "./checkpoint",
            "--subfolder",
            "nested",
            "--device",
            "auto",
            "--input",
            "requests.jsonl",
        ]);
        let cli = harness::parse(Bdecide::spec(), &words.words(), Bdecide::parse_from)
            .expect("predict arguments should parse");
        let Commands::Predict(args) = cli.command;
        assert_eq!(args.input, Some(Utf8PathBuf::from("requests.jsonl")));
        let options = load_options(&args).unwrap();
        assert!(options.device.is_none());
        let ModelSource::Local(path) = options.source else {
            panic!("./checkpoint should select a local directory");
        };
        assert_eq!(path, Utf8PathBuf::from("./checkpoint").join("nested"));
    }

    #[test]
    #[cfg(not(feature = "cpu"))]
    fn predict_rejects_disabled_cpu_before_model_io() {
        // Preserve the CLI's explicit default, e.g. a WGPU-only binary still rejects --device cpu.
        let words = harness::argv(["predict", "--model", "./missing-checkpoint"]);
        let cli = harness::parse(Bdecide::spec(), &words.words(), Bdecide::parse_from)
            .expect("predict arguments should parse");
        let Commands::Predict(args) = cli.command;
        assert!(matches!(load_options(&args), Err(Error::Device(_))));
    }

    #[rstest]
    #[case::missing_command(&[], "predict")]
    #[case::unknown_command(&["unknown"], "unknown")]
    #[case::missing_model(&["predict"], "--model")]
    #[case::invalid_device(&["predict", "--model", "./checkpoint", "--device", "invalid"], "invalid")]
    #[case::unknown_flag(&["predict", "--model", "./checkpoint", "--unknown"], "--unknown")]
    #[case::repeated_model(&["predict", "--model", "./one", "--model", "./two"], "--model")]
    #[case::missing_value(&["predict", "--model"], "--model")]
    #[case::conflicting_tokens(&["predict", "--model", "./checkpoint", "--anonymous", "--require-token"], "--anonymous")]
    #[case::reversed_conflict(&["predict", "--model", "./checkpoint", "--require-token", "--anonymous"], "--anonymous")]
    fn invalid_arguments_report_parse_errors(#[case] args: &[&str], #[case] text: &str) {
        let words = harness::argv(args.iter().copied());
        let outcome = harness::outcome(Bdecide::spec(), &words.words(), Bdecide::parse_from);
        let Outcome::Failed(printed) = outcome else {
            panic!("invalid arguments should fail: {outcome:?}");
        };
        assert_eq!(printed.code, 2);
        assert!(printed.stderr);
        assert!(printed.text.contains(text), "{}", printed.text);
    }

    #[test]
    fn help_lists_only_the_predict_command_and_its_examples() {
        let tree = harness::help_tree(Bdecide::spec(), Page::Long);
        let headers: Vec<_> = tree
            .lines()
            .filter(|line| line.starts_with("=== "))
            .collect();
        assert_eq!(headers, ["=== bdecide ===", "=== bdecide predict ==="]);
        assert!(tree.contains("bdecide predict --model"));
        assert!(tree.contains("--jsonl < requests.jsonl"));
    }
}
