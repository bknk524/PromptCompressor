fn main() {
    println!("cargo:rerun-if-env-changed=TRIMPROMPT_BUILD_ID");
    #[cfg(windows)]
    {
        let mut resource = winresource::WindowsResource::new();
        resource.set_icon("assets/TrimPrompt.ico");
        resource.set("FileDescription", "TrimPrompt");
        resource.set("ProductName", "TrimPrompt");
        resource.set_manifest(
            r#"<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<assembly xmlns="urn:schemas-microsoft-com:asm.v1" manifestVersion="1.0">
  <dependency>
    <dependentAssembly>
      <assemblyIdentity type="win32" name="Microsoft.Windows.Common-Controls" version="6.0.0.0" processorArchitecture="*" publicKeyToken="6595b64144ccf1df" language="*" />
    </dependentAssembly>
  </dependency>
</assembly>"#,
        );
        resource
            .compile()
            .expect("failed to embed Windows application icon");
    }
}
