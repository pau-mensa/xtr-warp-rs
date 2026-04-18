pub mod cuda_stream;
pub mod residual_codec;
pub mod types;

use indicatif::{ProgressBar, ProgressDrawTarget, ProgressStyle};

/// Create a progress bar that is either visible or hidden (no-op).
pub fn maybe_progress(show: bool, len: u64, msg: &str) -> ProgressBar {
    if show {
        let bar = ProgressBar::hidden();
        bar.set_length(len);
        bar.set_style(
            ProgressStyle::default_bar()
                .template("{msg} {wide_bar} {pos}/{len}")
                .unwrap(),
        );
        bar.set_message(msg.to_owned());
        bar.set_draw_target(ProgressDrawTarget::stderr());
        bar
    } else {
        ProgressBar::hidden()
    }
}
