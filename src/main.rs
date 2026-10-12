#![recursion_limit = "256"]
mod cli;

use std::process::ExitCode;

use usage_rs::Run;

use crate::cli::Bdecide;

fn main() -> ExitCode {
    Bdecide::parse().run()
}
