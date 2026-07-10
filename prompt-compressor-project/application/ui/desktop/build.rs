fn main() {
    #[cfg(windows)]
    {
        let mut resource = winresource::WindowsResource::new();
        resource.set_icon("assets/PromptC.ico");
        resource.set("FileDescription", "Prompt Compressor");
        resource.set("ProductName", "Prompt Compressor");
        resource
            .compile()
            .expect("failed to embed Windows application icon");
    }
}
