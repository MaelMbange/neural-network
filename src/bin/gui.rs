// Declare the history and gui modules from their canonical locations under src/,
// without touching src/lib.rs.
#[path = "../history/mod.rs"]
mod history;

#[path = "../gui/mod.rs"]
mod gui_module;

fn main() -> eframe::Result<()> {
    let options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default()
            .with_title("RNA — Neural-network trainer")
            .with_inner_size([1100.0, 700.0]),
        ..Default::default()
    };

    eframe::run_native(
        "RNA Trainer",
        options,
        Box::new(|cc| Ok(Box::new(gui_module::app::GuiApp::new(cc)))),
    )
}
