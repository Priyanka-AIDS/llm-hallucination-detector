import gradio as gr  # type: ignore
import requests  # type: ignore

def detect_hallucination(prompt, response):
    try:
        res = requests.post(
            "http://localhost:8000/detect",
            json={
                "prompt": prompt,
                "response": response
            }
        )

        data = res.json()

        highlighted = [
            (response, "Hallucinated" if data["is_hallucinated"] else None)
        ]

        return (
            highlighted,
            data["hallucination_score"],
            data["explanation"]
        )

    except Exception as e:
        return (
            [(response, None)],
            0,
            f"Error: {str(e)}"
        )


with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🔍 RUC-Detect")
    gr.Markdown("Real-time hallucination detection system")

    with gr.Row():
        with gr.Column():
            prompt_in = gr.Textbox(label="Prompt", lines=3)
            response_in = gr.Textbox(label="LLM Response", lines=4)
            btn = gr.Button("Analyze")

        with gr.Column():
            span_out = gr.HighlightedText(color_map={"Hallucinated": "red"})
            score_out = gr.Number(label="Score")
            explanation_out = gr.Textbox(label="Explanation")

    btn.click(
        fn=detect_hallucination,
        inputs=[prompt_in, response_in],
        outputs=[span_out, score_out, explanation_out]
    )


if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)