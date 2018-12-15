fn main() -> Result<(), ()> {
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-changed=./src/util_helpers.h");
    cc::Build::new()
        .file("./src/util_helpers.c")
        .compile("util_helpers");
    Ok(())
}
