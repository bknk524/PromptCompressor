use clap::Parser;

#[derive(Debug, Parser)]
#[command(name = "prompt-compressor-mcp-server")]
#[command(about = "MCP server scaffold for Prompt Compressor")]
struct Args {
    #[arg(long)]
    print_tools: bool,
}

fn main() {
    let args = Args::parse();

    if args.print_tools {
        let tools = serde_json::json!([
            {
                "name": "compress_prompt",
                "description": "Compress a natural-language prompt into a Codex-oriented prompt."
            },
            {
                "name": "estimate_tokens",
                "description": "Estimate token counts before and after compression."
            },
            {
                "name": "list_profiles",
                "description": "List configured profiles such as standard and code_focused."
            },
            {
                "name": "compare_profiles",
                "description": "Compare multiple profiles against the same input."
            }
        ]);
        println!("{tools}");
        return;
    }

    eprintln!("stdio MCP transport is not implemented yet.");
    eprintln!("This binary is a scaffold entry point for the next implementation step.");
}
