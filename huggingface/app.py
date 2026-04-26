import gradio as gr
import subprocess
import threading
import os

def run_training():
    print("Starting background training process...", flush=True)
    subprocess.run(["python3", "-u", "train_colab_aniket.py"])
    print("Training finished!", flush=True)

def start_ui():
    with gr.Blocks() as demo:
        gr.Markdown("# 🚀 DebugZero GRPO Training Space")
        gr.Markdown(
            "Click the button below to asynchronously dispatch the Reinforcement Learning script directly onto the A100 GPU natively within this container."
        )

        with gr.Row():
            btn = gr.Button("Launch Training Run", variant="primary")

        status = gr.Textbox(
            label="Status Logging",
            placeholder="Standby..."
        )

        def launch():
            threading.Thread(target=run_training, daemon=True).start()
            return (
                "✅ Training launched successfully in background terminal!\n"
                "Check the Logs tab to watch tqdm progress and metric plotting."
            )

        btn.click(launch, outputs=[status])

        gr.Markdown("## 📊 Training Outputs")

        if os.path.exists("metrics.png"):
            gr.Image(value="metrics.png", label="Metrics Preview")
            gr.File(value="metrics.png", label="Download metrics.png")
        else:
            gr.Markdown("`metrics.png` not found yet. Run training first.")

        if os.path.exists("debugzero_model.zip"):
            gr.File(
                value="debugzero_model.zip",
                label="Download Trained Model (.zip)"
            )
        elif os.path.exists("debugzero_model"):
            gr.Markdown(
                "`debugzero_model/` exists. Zip it in training script to download easily."
            )

    print("\n[DebugZero] Initiating Gradio Web Server on port 7860...", flush=True)

    demo.launch(
        server_name="0.0.0.0",
        server_port=7860
    )

if __name__ == "__main__":
    print("[DebugZero] Secure Container Boot Sequence Started.", flush=True)
    start_ui()