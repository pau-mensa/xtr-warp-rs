use std::env;
use std::path::{Path, PathBuf};
use std::process::Command;

fn main() {
    // Declare the custom cfg we flip from build_cuda_shim so downstream
    // `#[cfg(xtr_has_cuda_shim)]` stops emitting a rustc lint.
    println!("cargo:rustc-check-cfg=cfg(xtr_has_cuda_shim)");

    let os = env::var("CARGO_CFG_TARGET_OS").expect("Unable to get TARGET_OS");

    // Existing libtorch linkage (unchanged from previous build.rs).
    match os.as_str() {
        "linux" | "windows" => {
            if let Some(lib_path) = env::var_os("DEP_TCH_LIBTORCH_LIB") {
                println!(
                    "cargo:rustc-link-arg=-Wl,-rpath={}",
                    lib_path.to_string_lossy()
                );
            }
            println!("cargo:rustc-link-arg=-Wl,--no-as-needed");
            println!("cargo:rustc-link-arg=-ltorch");
        },
        _ => {},
    }

    // Build the CUDA stream FFI shim (Linux-only; CUDA is not expected on
    // macOS builds of this project). Requires libtorch headers + a cuda
    // runtime header. The shim itself only calls into the C++ API —
    // cuda_runtime.h is dragged in transitively by c10 headers, so we
    // just need CUDA's include path.
    if os == "linux" {
        if let Err(e) = build_cuda_shim() {
            println!(
                "cargo:warning=CUDA stream shim build skipped: {}. Multi-stream Phase-3 will be unavailable.",
                e
            );
        }
    }

    println!("cargo:rerun-if-changed=shim/cuda_stream_shim.cpp");
    println!("cargo:rerun-if-changed=build.rs");
}

fn build_cuda_shim() -> Result<(), String> {
    // libtorch include/lib — pulled from torch-sys env via DEP_TCH_*.
    let torch_lib = env::var_os("DEP_TCH_LIBTORCH_LIB")
        .ok_or("DEP_TCH_LIBTORCH_LIB not set (torch-sys didn't populate it)".to_string())?;
    let torch_lib_path = PathBuf::from(&torch_lib);
    // libtorch include dir sits next to lib/, as a sibling.
    let torch_root = torch_lib_path
        .parent()
        .ok_or("DEP_TCH_LIBTORCH_LIB has no parent".to_string())?;
    let torch_include = torch_root.join("include");
    let torch_api_include = torch_include.join("torch/csrc/api/include");

    if !torch_include.join("c10/cuda/CUDAStream.h").exists() {
        return Err(format!(
            "c10/cuda/CUDAStream.h not found under {:?} — is this a CUDA-enabled libtorch?",
            torch_include
        ));
    }

    // Hunt down a usable cuda_runtime.h. Preference order:
    //   1) venv nvidia/cuda_runtime/include  (pip-installed CUDA)
    //   2) /usr/local/cuda/include
    //   3) /opt/cuda/include
    let cuda_include = find_cuda_include()
        .ok_or("no cuda_runtime.h found in common locations".to_string())?;

    // PyTorch 2.x wheels since ~2.6 use the CXX11 ABI (=1). Older wheels
    // use =0. Ask python at build time to avoid guessing.
    let abi_flag = detect_cxx11_abi();

    let mut build = cc::Build::new();
    build
        .cpp(true)
        .std("c++17")
        .flag_if_supported("-Wno-unused-parameter")
        .flag_if_supported("-Wno-deprecated-declarations")
        .define("_GLIBCXX_USE_CXX11_ABI", abi_flag.as_str())
        .file("shim/cuda_stream_shim.cpp")
        .include(&torch_include)
        .include(&torch_api_include)
        .include(&cuda_include);

    build
        .try_compile("xtr_cuda_shim")
        .map_err(|e| format!("cc build failed: {}", e))?;

    // We statically link the shim into the cdylib via cc, but the shim
    // calls into libc10_cuda's C++ API, which we must link against.
    println!("cargo:rustc-link-search=native={}", torch_lib_path.display());
    println!("cargo:rustc-link-lib=c10_cuda");
    println!("cargo:rustc-link-lib=c10");
    println!("cargo:rustc-cfg=xtr_has_cuda_shim");

    Ok(())
}

fn find_cuda_include() -> Option<PathBuf> {
    // Preference order:
    //   1) triton's bundled CUDA headers (has both cuda_runtime.h AND
    //      crt/host_defines.h laid out as a full CUDA install)
    //   2) system /usr/local/cuda or /opt/cuda
    //   3) nvidia.cuda_runtime pip wheel (NOTE: has cuda_runtime.h but
    //      missing crt/ subdir, so usually won't compile standalone)
    let py = env::var("PYTHON").unwrap_or_else(|_| "python".into());
    let ask = |cmd: &str| -> Option<PathBuf> {
        let out = Command::new(&py).args(["-c", cmd]).output().ok()?;
        if !out.status.success() {
            return None;
        }
        let s = String::from_utf8_lossy(&out.stdout).trim().to_string();
        if s.is_empty() {
            None
        } else {
            Some(PathBuf::from(s))
        }
    };
    let mut candidates: Vec<PathBuf> = Vec::new();
    if let Some(p) = ask(
        "import triton.backends.nvidia as m, os; \
         print(os.path.join(os.path.dirname(m.__file__), 'include'))",
    ) {
        candidates.push(p);
    }
    candidates.push(PathBuf::from("/usr/local/cuda/include"));
    candidates.push(PathBuf::from("/opt/cuda/include"));
    if let Some(p) = ask(
        "import nvidia.cuda_runtime as m, os; \
         print(os.path.join(os.path.dirname(m.__file__), 'include'))",
    ) {
        candidates.push(p);
    }
    for c in candidates {
        // Must have cuda_runtime.h AND crt/host_defines.h (the latter is
        // transitively included by cuda_runtime_api.h).
        if c.join("cuda_runtime.h").exists() && c.join("crt/host_defines.h").exists() {
            return Some(c);
        }
    }
    None
}

fn detect_cxx11_abi() -> String {
    // Default to CXX11 ABI enabled (modern torch wheels).
    let default = "1".to_string();
    let python = env::var("PYTHON").unwrap_or_else(|_| "python".into());
    let out = Command::new(python)
        .args([
            "-c",
            "import torch; print(int(torch._C._GLIBCXX_USE_CXX11_ABI))",
        ])
        .output();
    if let Ok(o) = out {
        if o.status.success() {
            let s = String::from_utf8_lossy(&o.stdout).trim().to_string();
            if s == "0" || s == "1" {
                return s;
            }
        }
    }
    default
}

#[allow(dead_code)]
fn _path_exists(p: &Path) -> bool {
    p.exists()
}
