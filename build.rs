use std::env;

fn main() {
    build_gpu_merge_kernel();

    let os = env::var("CARGO_CFG_TARGET_OS").expect("Unable to get TARGET_OS");
    match os.as_str() {
        "linux" | "windows" => {
            if let Some(lib_path) = env::var_os("DEP_TCH_LIBTORCH_LIB") {
                println!(
                    "cargo:rustc-link-arg=-Wl,-rpath={}",
                    lib_path.to_string_lossy()
                );
            }
            println!("cargo:rustc-link-arg=-Wl,--no-as-needed");
            // println!("cargo:rustc-link-arg=-Wl,--copy-dt-needed-entries");
            println!("cargo:rustc-link-arg=-ltorch");
        },
        _ => {},
    }
}

fn build_gpu_merge_kernel() {
    println!("cargo:rerun-if-changed=rust/search/gpu_merge_kernel.cpp");
    println!("cargo:rerun-if-env-changed=LIBTORCH");
    println!("cargo:rerun-if-env-changed=LIBTORCH_CXX11_ABI");
    println!("cargo:rerun-if-env-changed=TORCH_CXX11_ABI");

    let libtorch_path = match env::var("LIBTORCH") {
        Ok(path) => path,
        Err(_) => {
            eprintln!("[build] LIBTORCH not set; skipping gpu_merge_kernel compilation.");
            return;
        },
    };

    let include_main = format!("{}/include", libtorch_path);
    let include_api = format!("{}/include/torch/csrc/api/include", libtorch_path);

    let mut builder = cc::Build::new();
    builder
        .cpp(true)
        .flag_if_supported("-std=c++17")
        .include(&include_main)
        .include(&include_api)
        .file("rust/search/gpu_merge_kernel.cpp");

    // Respect libtorch ABI flag.
    if let Ok(abi) = env::var("LIBTORCH_CXX11_ABI")
        .or_else(|_| env::var("TORCH_CXX11_ABI"))
        .or_else(|_| env::var("CXX11_ABI"))
    {
        builder.define("_GLIBCXX_USE_CXX11_ABI", Some(abi.as_str()));
    }

    builder.compile("xtr_gpu_merge");
}
