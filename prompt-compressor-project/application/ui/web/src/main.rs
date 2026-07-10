use std::path::PathBuf;

use anyhow::Result;
use clap::Parser;
use prompt_compressor_local_ui::{run_server, ServerOptions};

#[derive(Debug, Parser)]
#[command(name = "prompt-compressor-local-ui")]
#[command(about = "Development local UI for Prompt Compressor")]
struct Args {
    #[arg(long, value_name = "HOST", default_value = "127.0.0.1")]
    host: String,

    #[arg(long, value_name = "PORT", default_value_t = 8787)]
    port: u16,

    #[arg(long, value_name = "DIR")]
    settings_dir: Option<PathBuf>,
}

fn main() -> Result<()> {
    let args = Args::parse();
    run_server(ServerOptions {
        host: args.host,
        port: args.port,
        settings_dir: args.settings_dir,
    })
}
