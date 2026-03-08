# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "openai",
#   "playwright",
#   "streamlit",
# ]
# ///
# Run with: `uv run streamlit run code_execution.py`
# Install Chromium once first: `uv run --with playwright python -m playwright install chromium`
# Requires `OPENAI_API_KEY` in the environment or Streamlit secrets.

"""Streamlit app for running a browser automation agent with the Responses API."""

from __future__ import annotations

import base64
import json
import os
import traceback
from typing import Any

from openai import OpenAI
from playwright.sync_api import sync_playwright
import streamlit as st

DEFAULT_PROMPT = (
    "Go to Hacker News, click on the most interesting link "
    "(be prepared to justify your choice), take a screenshot, "
    "and give me a critique of the visual layout."
)
DEFAULT_MODEL = "gpt-5.4"


def _message_text(item: Any) -> str:
    try:
        parts = getattr(item, "content", None)
        if isinstance(parts, list) and parts:
            out: list[str] = []
            for part in parts:
                text = getattr(part, "text", None)
                if isinstance(text, str) and text:
                    out.append(text)
            if out:
                return "\n".join(out)
    except Exception:
        pass
    return str(item)


def _load_openai_api_key() -> str | None:
    api_key = os.getenv("OPENAI_API_KEY")
    if isinstance(api_key, str) and api_key.strip():
        return api_key.strip()

    try:
        secret_value = st.secrets.get("OPENAI_API_KEY")
    except Exception:
        return None

    if isinstance(secret_value, str) and secret_value.strip():
        return secret_value.strip()
    return None


def _create_client() -> OpenAI:
    api_key = _load_openai_api_key()
    if not api_key:
        raise RuntimeError(
            "OPENAI_API_KEY is not configured. Set it in the environment or Streamlit secrets."
        )
    return OpenAI(api_key=api_key)


def _tool_specs() -> list[dict[str, Any]]:
    return [
        {
            "type": "function",
            "name": "exec_py",
            "description": "Execute provided interactive Python in a persistent runtime context.",
            "parameters": {
                "type": "object",
                "properties": {
                    "code": {
                        "type": "string",
                        "description": (
                            "Python code to execute. Write small snippets. "
                            "State persists across tool calls via globals(). "
                            "This runtime uses Playwright's sync Python API, so call methods directly "
                            "without await. Do not use asyncio.run(...), threads, or create your own "
                            "event loop. You can use ONLY these prebound objects/helpers: "
                            "log(x) for text output, display(base64_png_string) for image output, "
                            "browser (Playwright browser), context (viewport 1440x900), page "
                            "(already created). Be concise with log(x): do not send large base64 "
                            "payloads, screenshots, buffers, page HTML, or other large blobs through "
                            "log(). If you create an image or screenshot, pass the base64 PNG string "
                            "to display(). Do not write screenshots or image data to disk. Do not "
                            "assume extra globals or helpers are available unless they are explicitly "
                            "listed here. Do not close browser/context/page unless explicitly asked."
                        ),
                    }
                },
                "required": ["code"],
                "additionalProperties": False,
            },
        },
        {
            "type": "function",
            "name": "ask_user",
            "description": "Ask the user a clarification question and wait for their response.",
            "parameters": {
                "type": "object",
                "properties": {
                    "question": {
                        "type": "string",
                        "description": "The exact question to show the user.",
                    }
                },
                "required": ["question"],
                "additionalProperties": False,
            },
        },
    ]


def _init_state() -> None:
    defaults: dict[str, Any] = {
        "ce_playwright": None,
        "ce_browser": None,
        "ce_context": None,
        "ce_page": None,
        "ce_runtime_globals": None,
        "ce_response_id": None,
        "ce_next_input": None,
        "ce_pending_inputs": [],
        "ce_pending_question": None,
        "ce_pending_call_id": None,
        "ce_events": [],
        "ce_status": "idle",
        "ce_final_answer": "",
        "ce_last_message": "",
        "ce_steps_remaining": 0,
        "ce_prompt_value": DEFAULT_PROMPT,
        "ce_model_value": DEFAULT_MODEL,
        "ce_headless": True,
        "ce_max_steps": 20,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def _append_event(event_type: str, **payload: Any) -> None:
    st.session_state["ce_events"].append({"type": event_type, **payload})


def _close_runtime() -> None:
    for key in ("ce_page", "ce_context", "ce_browser"):
        obj = st.session_state.get(key)
        if obj is not None:
            try:
                obj.close()
            except Exception:
                pass
    playwright_driver = st.session_state.get("ce_playwright")
    if playwright_driver is not None:
        try:
            playwright_driver.stop()
        except Exception:
            pass

    st.session_state["ce_playwright"] = None
    st.session_state["ce_browser"] = None
    st.session_state["ce_context"] = None
    st.session_state["ce_page"] = None
    st.session_state["ce_runtime_globals"] = None


def _reset_agent_state(close_runtime: bool) -> None:
    if close_runtime:
        _close_runtime()

    st.session_state["ce_response_id"] = None
    st.session_state["ce_next_input"] = None
    st.session_state["ce_pending_inputs"] = []
    st.session_state["ce_pending_question"] = None
    st.session_state["ce_pending_call_id"] = None
    st.session_state["ce_events"] = []
    st.session_state["ce_status"] = "idle"
    st.session_state["ce_final_answer"] = ""
    st.session_state["ce_last_message"] = ""
    st.session_state["ce_steps_remaining"] = 0


def _ensure_runtime(headless: bool) -> None:
    if st.session_state.get("ce_browser") is not None:
        return

    playwright_driver = sync_playwright().start()
    browser = playwright_driver.chromium.launch(
        headless=headless,
        args=["--window-size=1440,900"],
    )
    context = browser.new_context(viewport={"width": 1440, "height": 900})
    page = context.new_page()

    st.session_state["ce_playwright"] = playwright_driver
    st.session_state["ce_browser"] = browser
    st.session_state["ce_context"] = context
    st.session_state["ce_page"] = page
    st.session_state["ce_runtime_globals"] = {
        "__builtins__": __builtins__,
        "browser": browser,
        "context": context,
        "page": page,
    }


def _execute_python_tool(code: str) -> list[dict[str, Any]]:
    py_output: list[dict[str, Any]] = []
    runtime_globals = st.session_state["ce_runtime_globals"]
    runtime_globals["browser"] = st.session_state["ce_browser"]
    runtime_globals["context"] = st.session_state["ce_context"]
    runtime_globals["page"] = st.session_state["ce_page"]

    def log(*items: Any) -> None:
        text = " ".join(str(item) for item in items)
        py_output.append({"type": "input_text", "text": text[:5000]})

    def display(base64_image: str) -> None:
        py_output.append(
            {
                "type": "input_image",
                "image_url": f"data:image/png;base64,{base64_image}",
                "detail": "original",
            }
        )

    runtime_globals["log"] = log
    runtime_globals["display"] = display
    _append_event("code", text=code)

    wrapped = (
        "def __codex_exec__():\n"
        + "".join(
            f"    {line}\n" if line else "    \n"
            for line in (code or "pass").splitlines()
        )
    )

    try:
        exec(wrapped, runtime_globals, runtime_globals)
        runtime_globals["__codex_exec__"]()
    except Exception:
        log(traceback.format_exc())

    for item in py_output:
        if item.get("type") == "input_text":
            _append_event("tool_log", text=item.get("text", ""))
        elif item.get("type") == "input_image":
            _append_event("tool_image", image_url=item.get("image_url", ""))

    return py_output[:]


def _handle_response(resp: Any) -> tuple[list[dict[str, Any]] | None, bool]:
    had_tool_call = False
    latest_phase: str | None = None
    pending_outputs: list[dict[str, Any]] = []

    for item in resp.output:
        item_type = getattr(item, "type", None)

        if item_type == "function_call" and getattr(item, "name", None) == "exec_py":
            had_tool_call = True
            raw_args = getattr(item, "arguments", "{}") or "{}"
            try:
                args = json.loads(raw_args)
            except json.JSONDecodeError:
                args = {}
            code = args.get("code", "") if isinstance(args, dict) else ""

            pending_outputs.append(
                {
                    "type": "function_call_output",
                    "call_id": getattr(item, "call_id", None),
                    "output": _execute_python_tool(code),
                }
            )
            continue

        if item_type == "function_call" and getattr(item, "name", None) == "ask_user":
            had_tool_call = True
            raw_args = getattr(item, "arguments", "{}") or "{}"
            try:
                args = json.loads(raw_args)
            except json.JSONDecodeError:
                args = {}
            question = (
                args.get("question", "Please provide more information.")
                if isinstance(args, dict)
                else "Please provide more information."
            )

            st.session_state["ce_pending_question"] = question
            st.session_state["ce_pending_call_id"] = getattr(item, "call_id", None)
            st.session_state["ce_pending_inputs"] = pending_outputs[:]
            st.session_state["ce_status"] = "waiting_for_user"
            _append_event("question", text=question)
            return None, False

        if item_type == "message":
            text = _message_text(item).strip()
            phase = getattr(item, "phase", None)
            if text:
                st.session_state["ce_last_message"] = text
                if phase == "final_answer":
                    st.session_state["ce_final_answer"] = text
                _append_event("assistant", text=text, phase=phase)
            if isinstance(phase, str) or phase is None:
                latest_phase = phase
            continue

        if item_type == "output_item.done":
            phase = getattr(item, "phase", None)
            if isinstance(phase, str) or phase is None:
                latest_phase = phase

    if pending_outputs:
        return pending_outputs, False

    if not had_tool_call:
        if latest_phase == "final_answer" and st.session_state["ce_last_message"]:
            st.session_state["ce_final_answer"] = st.session_state["ce_last_message"]
        st.session_state["ce_status"] = "completed"
        return None, True

    return None, False


def _run_agent_steps() -> None:
    try:
        client = _create_client()

        while st.session_state["ce_steps_remaining"] > 0:
            current_input = st.session_state.get("ce_next_input")
            if not current_input:
                if st.session_state.get("ce_pending_question"):
                    return
                st.session_state["ce_status"] = "completed"
                return

            request: dict[str, Any] = {
                "model": st.session_state["ce_model_value"],
                "tools": _tool_specs(),
                "input": current_input,
            }
            previous_response_id = st.session_state.get("ce_response_id")
            if previous_response_id:
                request["previous_response_id"] = previous_response_id

            resp = client.responses.create(**request)
            st.session_state["ce_response_id"] = resp.id
            st.session_state["ce_steps_remaining"] -= 1

            next_input, finished = _handle_response(resp)
            if st.session_state["ce_status"] == "waiting_for_user":
                st.session_state["ce_next_input"] = None
                return
            if finished:
                st.session_state["ce_next_input"] = None
                return
            if next_input:
                st.session_state["ce_next_input"] = next_input
                continue

            st.session_state["ce_next_input"] = None
            st.session_state["ce_status"] = "completed"
            return

        if st.session_state["ce_status"] == "running":
            st.session_state["ce_status"] = "max_steps"
            _append_event("status", text="Stopped after reaching the max step limit.")
    except Exception:
        st.session_state["ce_status"] = "error"
        error_text = traceback.format_exc()
        _append_event("error", text=error_text)


def _start_run(prompt: str, max_steps: int, model: str, headless: bool) -> None:
    _reset_agent_state(close_runtime=True)
    _ensure_runtime(headless=headless)

    st.session_state["ce_prompt_value"] = prompt
    st.session_state["ce_model_value"] = model
    st.session_state["ce_headless"] = headless
    st.session_state["ce_max_steps"] = max_steps
    st.session_state["ce_steps_remaining"] = max_steps
    st.session_state["ce_status"] = "running"
    st.session_state["ce_next_input"] = [{"role": "user", "content": prompt}]

    _run_agent_steps()


def _continue_after_user_input(answer: str) -> None:
    pending_call_id = st.session_state.get("ce_pending_call_id")
    if not pending_call_id:
        return

    pending_inputs = list(st.session_state.get("ce_pending_inputs") or [])
    pending_inputs.append(
        {
            "type": "function_call_output",
            "call_id": pending_call_id,
            "output": answer,
        }
    )

    _append_event("user", text=answer)
    st.session_state["ce_pending_question"] = None
    st.session_state["ce_pending_call_id"] = None
    st.session_state["ce_pending_inputs"] = []
    st.session_state["ce_status"] = "running"
    st.session_state["ce_next_input"] = pending_inputs

    _run_agent_steps()


def _run_more_steps(extra_steps: int) -> None:
    st.session_state["ce_steps_remaining"] = extra_steps
    st.session_state["ce_status"] = "running"
    _run_agent_steps()


def _status_label() -> str:
    status = st.session_state.get("ce_status", "idle")
    labels = {
        "idle": "Idle",
        "running": "Running",
        "waiting_for_user": "Waiting for user input",
        "completed": "Completed",
        "max_steps": "Max steps reached",
        "error": "Error",
    }
    return labels.get(status, status)


def _render_event(event: dict[str, Any], index: int) -> None:
    event_type = event.get("type")

    if event_type == "assistant":
        st.markdown(event.get("text", ""))
        return

    if event_type == "user":
        st.caption("User response")
        st.write(event.get("text", ""))
        return

    if event_type == "question":
        st.info(event.get("text", ""))
        return

    if event_type == "status":
        st.caption(event.get("text", ""))
        return

    if event_type == "error":
        st.error(event.get("text", ""))
        return

    if event_type == "code":
        with st.expander(f"Executed Python #{index + 1}", expanded=False):
            st.code(event.get("text", ""), language="python")
        return

    if event_type == "tool_log":
        st.code(event.get("text", ""), language="text")
        return

    if event_type == "tool_image":
        image_url = event.get("image_url", "")
        if image_url.startswith("data:image/png;base64,"):
            payload = image_url.split(",", 1)[1]
            try:
                st.image(base64.b64decode(payload), width="stretch")
                return
            except Exception:
                pass
        st.caption("Image output unavailable")


def main() -> None:
    st.set_page_config(page_title="Code Execution Agent", layout="wide")
    _init_state()

    st.title("Code Execution Agent")
    st.write("Run a browser automation agent backed by Playwright and the OpenAI Responses API.")

    with st.form("ce_run_form"):
        prompt = st.text_area(
            "Prompt",
            value=st.session_state.get("ce_prompt_value", DEFAULT_PROMPT),
            height=140,
        )
        col1, col2, col3 = st.columns(3)
        with col1:
            model = st.text_input(
                "Model",
                value=st.session_state.get("ce_model_value", DEFAULT_MODEL),
            )
        with col2:
            max_steps = int(
                st.number_input(
                    "Max steps",
                    min_value=1,
                    max_value=100,
                    value=int(st.session_state.get("ce_max_steps", 20)),
                    step=1,
                )
            )
        with col3:
            headless = st.checkbox(
                "Headless browser",
                value=bool(st.session_state.get("ce_headless", True)),
                help="Recommended for Streamlit runs.",
            )
        start_run = st.form_submit_button("Start Run", type="primary")

    action_col1, action_col2 = st.columns(2)
    with action_col1:
        continue_more = st.button(
            "Run 5 More Steps",
            disabled=st.session_state.get("ce_status") not in {"max_steps", "running"},
        )
    with action_col2:
        reset_run = st.button("Reset Session")

    if start_run:
        with st.spinner("Running agent..."):
            _start_run(
                prompt=prompt.strip() or DEFAULT_PROMPT,
                max_steps=max_steps,
                model=model.strip() or DEFAULT_MODEL,
                headless=headless,
            )

    if continue_more:
        with st.spinner("Running more steps..."):
            _run_more_steps(5)

    if reset_run:
        _reset_agent_state(close_runtime=True)
        st.rerun()

    st.caption(f"Status: {_status_label()}")

    page = st.session_state.get("ce_page")
    if page is not None:
        try:
            current_url = page.url
        except Exception:
            current_url = ""
        if current_url:
            st.caption(f"Current page: {current_url}")

    pending_question = st.session_state.get("ce_pending_question")
    if pending_question:
        st.subheader("Clarification Needed")
        st.info(pending_question)
        with st.form("ce_question_form"):
            answer = st.text_input("Your answer")
            submit_answer = st.form_submit_button("Continue")
        if submit_answer:
            with st.spinner("Continuing agent run..."):
                _continue_after_user_input(answer.strip())

    final_answer = st.session_state.get("ce_final_answer", "").strip()
    if final_answer:
        st.subheader("Final Answer")
        st.markdown(final_answer)

    if st.session_state.get("ce_events"):
        st.subheader("Run Output")
        for index, event in enumerate(st.session_state["ce_events"]):
            _render_event(event, index)


if __name__ == "__main__":
    main()
