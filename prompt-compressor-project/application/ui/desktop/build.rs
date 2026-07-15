fn main() {
    #[cfg(windows)]
    {
        let mut resource = winresource::WindowsResource::new();
        resource.set_icon("assets/TrimPrompt.ico");
        resource.set("FileDescription", "TrimPrompt");
        resource.set("ProductName", "TrimPrompt");
        resource
            .compile()
            .expect("failed to embed Windows application icon");
    }
}
