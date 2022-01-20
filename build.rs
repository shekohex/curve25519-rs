use std::env;

fn main() -> Result<(), ()> {
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-changed=./src/util_helpers.h");

    let target = env::var("TARGET").unwrap();
    let host = env::var("HOST").unwrap();
    if target.contains("msvc") && host.contains("windows") {
        let mut config = cc::Build::new();
        config.file("src/util_helpers.asm");
        if target.contains("x86_64") {
            config.define("X64", None);
        }
        config.compile("util_helpers");
    }
    else
    {
        cc::Build::new()
            .file("./src/util_helpers.c")
            .compile("util_helpers");
    }
    Ok(())
}
