#[tokio::main]
async fn main() -> std::io::Result<()> {
    sid_isnt_done::builtin_tools::run_sid_editor_tool().await
}
