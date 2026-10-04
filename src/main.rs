#![recursion_limit = "256"]
mod cli;

use std::process::ExitCode;
use usage::Run;

fn main() -> ExitCode {
    cli::Bdecide::parse().run()
}
