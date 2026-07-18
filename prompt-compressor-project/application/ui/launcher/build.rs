fn main() {
    println!("cargo:rerun-if-env-changed=TRIMPROMPT_BUILD_ID");
    #[cfg(windows)]
    {
        let mut resource = winresource::WindowsResource::new();
        resource.set_icon("../desktop/assets/TrimPrompt.ico");
        resource.set("FileDescription", "TrimPrompt launcher");
        resource.set("ProductName", "TrimPrompt");
        resource
            .compile()
            .expect("failed to embed Windows application icon");
    }
}
