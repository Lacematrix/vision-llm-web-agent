"""Centralized prompt templates managed by LangChain prompt utilities."""

from __future__ import annotations

import json
from typing import Any, Dict, List

from langchain_core.prompts import PromptTemplate


TOOL_DESCRIPTION_TEMPLATE = PromptTemplate.from_template(
    """{name}: {description}\n{parameters_block}"""
)

AGENT_SYSTEM_PROMPT_TEMPLATE = PromptTemplate.from_template(
    """You are an autonomous web agent. Your job is to complete tasks by controlling a web browser and using available tools.

**Available Tools:**
{tool_descriptions}

**Input Format:**
You will receive:
- Screenshot (if available)
- Current state in JSON format with: round, screenshot_available, dom_summary, instruction
- Tool execution results in JSON format: {{"tool_execution": "tool_name", "result": "result_text"}}

**Response Format (MUST be valid JSON):**

To use a tool:
```json
{{
    "thought": "What I'm doing and why",
    "tool": "tool_name",
    "parameters": {{"param": "value"}},
    "next": "If task is not fully complete, what to do next"
}}
```

When task is complete:
```json
{{
    "thought": "Summary of what was accomplished",
    "status": "complete",
    "result": "Final answer for the user"
}}
```

**Rules:**
1. Respond ONLY with valid JSON (start with {{, end with }})
2. Call ONE tool at a time
3. **CRITICAL: Trust DOM over screenshot** - If an element is not in dom_summary, it's NOT clickable, even if you see it in the screenshot
4. Use specific CSS selectors for click/type actions
5. If there is a CAPTCHA, try another site, DO NOT try to solve the CAPTCHA.
6. **Error Recovery:** If actions fail 2+ times, try different approaches - never repeat the exact same action more than 2 times
7. Call download_pdf for pdf download.
8. **File Paths:** For all file operations (download_pdf, pdf_extract_text, pdf_extract_images, save_image, write_text), provide ONLY the filename (e.g., "abc.pdf", "output.txt"), NOT directory paths. The system will automatically save files to artifacts/ directory in a single level (artifacts/filename).

**Preferences:**
1. Prefer arXiv for academic and technical reports.
"""
)

CURRENT_STATE_MESSAGE_TEMPLATE = PromptTemplate.from_template(
    """Current State:
```json
{state_json}
```"""
)

JSON_RETRY_FEEDBACK_TEMPLATE = PromptTemplate.from_template(
    """ERROR: Your response was not valid JSON. Please respond with valid JSON format only. Use the exact format specified in the system prompt.

Your previous response:
{previous_response}"""
)

TEXT_SUMMARY_TEMPLATE = PromptTemplate.from_template(
    """Please provide a concise summary of the following text.
The summary should be clear, informative, and capture the main points.
Keep it under {max_length} characters.

Text to summarize:
{text_to_summarize}

Summary:"""
)

CONTEXT_COMPRESSION_SUMMARY_TEMPLATE = PromptTemplate.from_template(
    """Summarize the following conversation history for future continuity.
Keep key facts, constraints, completed actions, failures, and pending next steps.
Use concise bullet-like sentences, and keep the result under {max_length} characters.

Conversation history:
{history_text}

Compressed summary:"""
)

CONTEXT_SUMMARY_HISTORY_MESSAGE_TEMPLATE = PromptTemplate.from_template(
    """Context summary of earlier conversation:
{summary}

Use this summary as prior context and continue from the latest messages."""
)

DOM_FILTER_SYSTEM_TEMPLATE = PromptTemplate.from_template(
    """You are an HTML element filter helping a downstream web agent. Share only the most relevant interactive elements such as search inputs, navigation links, and buttons. Keep div elements with nav/search semantics when useful. The user will provide a question and a structured element list. Return the indexes of up to max = {max_elements} elements in the format ```json [1,3,5]``` and nothing else."""
)

DOM_FILTER_USER_TEMPLATE = PromptTemplate.from_template(
    """{user_prompt}

Here are the <{tag}> elements on the page:
{input_prompt}"""
)

LOCAL_MULTI_STEP_REMINDER_TEMPLATE = PromptTemplate.from_template(
    """

**CRITICAL - MULTI-STEP TASK DETECTED:**
Your original task requires multiple steps. You MUST complete ALL steps before marking as complete:
1. **FIRST:** Extract ALL images with pdf_extract_images(file_name=\"report.pdf\", output_dir=\"extracted_images\") WITHOUT page_num
2. **SECOND:** Save all extracted images using save_image for each image
3. **THIRD:** Find and interpret the first image (if task says 'interpret the first image')
4. **ONLY THEN:** Mark status as \"complete\" after ALL steps are done!

**DO NOT skip steps!** Check your original task requirements carefully!"""
)

LOCAL_FILE_PROCESSING_INSTRUCTION_TEMPLATE = PromptTemplate.from_template(
    """You are currently in LOCAL FILE PROCESSING mode. Use pdf_extract_text, pdf_extract_images, save_image, write_text, and ocr_image_to_text tools to process the downloaded PDF files. These tools work on local files in the artifacts/ directory. Do NOT use web browser tools (click, type_text, etc.) in this mode. DO NOT download PDF again - it's already downloaded!{multi_step_reminder}"""
)

WEB_BROWSING_INSTRUCTION_TEMPLATE = PromptTemplate.from_template(
    """Analyze the current state and decide the next action. Respond with valid JSON."""
)

LOCAL_FILE_PROCESSING_NOTE_TEMPLATE = PromptTemplate.from_template(
    """You are in LOCAL FILE PROCESSING mode. Use pdf_extract_text, pdf_extract_images, ocr_image_to_text tools. Ignore screenshot/DOM."""
)

PDF_DETECTED_INSTRUCTION_TEMPLATE = PromptTemplate.from_template(
    """CRITICAL: PDF PAGE DETECTED! The current page is a PDF file (URL: {pdf_url}). You MUST download it using download_pdf(url=\"{pdf_url}\", file_name=\"report.pdf\") before processing. DO NOT try to scroll or interact with the PDF in the browser - download it first!"""
)

FORCED_DOM_SUMMARY_MESSAGE_TEMPLATE = PromptTemplate.from_template(
    """FORCED ACTION: dom_summary was automatically called because '{action_key}' has failed {failure_count} times. Here is the DOM summary:
{dom_result}

You MUST use the selectors from the INPUT FIELDS section above. DO NOT repeat the blocked action!"""
)

INTERVENTION_REPEAT_TOOL_TEMPLATE = PromptTemplate.from_template(
    """INTERVENTION: You have called '{tool_name}' with the same parameters {repeat_count} times. The result was already provided. You MUST proceed to the next step:
{extra_guidance}"""
)

FAILED_ACTION_INTERVENTION_TEMPLATE = PromptTemplate.from_template(
    """INTERVENTION: The action '{tool_name}' with parameters {parameters_json} has failed {failure_count} times. You MUST call 'dom_summary' tool to find the correct selector before trying again. DO NOT repeat the same failed action! If this action fails {force_dom_summary_threshold} times, it will be automatically blocked."""
)

ACTION_BLOCKED_TEMPLATE = PromptTemplate.from_template(
    """ACTION BLOCKED: Your requested action '{tool_name}' with parameters {parameters_json} has been blocked because it has failed {failure_count} times. The system has automatically replaced it with a dom_summary call. You MUST use the selectors from the DOM summary result. DO NOT attempt the blocked action again!"""
)

DOWNLOAD_PDF_ALREADY_DONE_TEMPLATE = PromptTemplate.from_template(
    """PDF already downloaded! You are in LOCAL FILE PROCESSING mode. Available files: {available_files}. Use pdf_extract_text/pdf_extract_images tools to process the existing PDF. DO NOT download again!"""
)

CONTEXT_SWITCH_MESSAGE_TEMPLATE = PromptTemplate.from_template(
    """PDF file '{file_name}' has been successfully downloaded to local artifacts directory. You should now use local file processing tools (pdf_extract_text, pdf_extract_images, ocr_image_to_text, save_image, write_text) to process this file. These tools work on local files and do NOT require web browser operations. Ignore any screenshot/DOM information when processing local files."""
)

DOWNLOAD_CONTEXT_SWITCH_NOTE_TEMPLATE = PromptTemplate.from_template(
    """PDF file '{file_name}' has been successfully downloaded to local artifacts directory. You should now use local file processing tools (pdf_extract_text, pdf_extract_images, ocr_image_to_text) to process this file. These tools work on local files and do NOT require web browser operations. Ignore any screenshot/DOM information when processing local files.

IMPORTANT - For tasks requiring specific figures (e.g., 'interpret Figure 1'):
1. FIRST extract text from PDF (pdf_extract_text without page_num) to search for 'Figure 1' in the text and find which page it's on
2. THEN extract images ONLY from that specific page (use page_num parameter)
3. DO NOT extract images from page 1 or all pages before finding where Figure 1 is located!"""
)

HISTORY_SUMMARY_REQUEST_TEMPLATE = PromptTemplate.from_template(
    """Please provide a concise summary of the following conversation history:

{summary}

Summary:"""
)

HISTORY_SUMMARIZE_INSTRUCTION_TEMPLATE = PromptTemplate.from_template(
    """Summarize the conversation history concisely."""
)


def build_tool_descriptions(available_tools: List[Dict[str, Any]]) -> str:
    lines: List[str] = []
    for tool in available_tools:
        parameters_block = ""
        if tool.get("parameters"):
            parameters_block = (
                "Parameters: "
                + json.dumps(tool["parameters"], ensure_ascii=False, indent=2)
            )
        lines.append(
            TOOL_DESCRIPTION_TEMPLATE.format(
                name=tool["name"],
                description=tool["description"],
                parameters_block=parameters_block,
            )
        )
    return "\n".join(lines).strip()


def build_agent_system_prompt(available_tools: List[Dict[str, Any]]) -> str:
    return AGENT_SYSTEM_PROMPT_TEMPLATE.format(
        tool_descriptions=build_tool_descriptions(available_tools)
    )


def build_current_state_message(state_json: str) -> str:
    return CURRENT_STATE_MESSAGE_TEMPLATE.format(state_json=state_json)


def build_json_retry_feedback(previous_response: str) -> str:
    return JSON_RETRY_FEEDBACK_TEMPLATE.format(previous_response=previous_response)


def build_text_summary_prompt(text_to_summarize: str, max_length: int) -> str:
    return TEXT_SUMMARY_TEMPLATE.format(
        text_to_summarize=text_to_summarize,
        max_length=max_length,
    )


def build_context_compression_prompt(history_text: str, max_length: int) -> str:
    return CONTEXT_COMPRESSION_SUMMARY_TEMPLATE.format(
        history_text=history_text,
        max_length=max_length,
    )


def build_context_summary_history_message(summary: str) -> str:
    return CONTEXT_SUMMARY_HISTORY_MESSAGE_TEMPLATE.format(summary=summary)


def build_dom_filter_system_prompt(max_elements: int) -> str:
    return DOM_FILTER_SYSTEM_TEMPLATE.format(max_elements=max_elements)


def build_dom_filter_user_prompt(user_prompt: str, tag: str, input_prompt: str) -> str:
    return DOM_FILTER_USER_TEMPLATE.format(
        user_prompt=user_prompt,
        tag=tag,
        input_prompt=input_prompt,
    )


def build_local_multi_step_reminder() -> str:
    return LOCAL_MULTI_STEP_REMINDER_TEMPLATE.format()


def build_local_file_processing_instruction(multi_step_reminder: str) -> str:
    return LOCAL_FILE_PROCESSING_INSTRUCTION_TEMPLATE.format(
        multi_step_reminder=multi_step_reminder,
    )


def build_web_browsing_instruction() -> str:
    return WEB_BROWSING_INSTRUCTION_TEMPLATE.format()


def build_local_file_processing_note() -> str:
    return LOCAL_FILE_PROCESSING_NOTE_TEMPLATE.format()


def build_pdf_detected_instruction(pdf_url: str) -> str:
    return PDF_DETECTED_INSTRUCTION_TEMPLATE.format(pdf_url=pdf_url)


def build_forced_dom_summary_message(action_key: str, failure_count: int, dom_result: str) -> str:
    return FORCED_DOM_SUMMARY_MESSAGE_TEMPLATE.format(
        action_key=action_key,
        failure_count=failure_count,
        dom_result=dom_result,
    )


def build_intervention_repeat_tool_message(tool_name: str, repeat_count: int, extra_guidance: str) -> str:
    return INTERVENTION_REPEAT_TOOL_TEMPLATE.format(
        tool_name=tool_name,
        repeat_count=repeat_count,
        extra_guidance=extra_guidance,
    )


def build_failed_action_intervention(tool_name: str, parameters_json: str, failure_count: int, force_dom_summary_threshold: int) -> str:
    return FAILED_ACTION_INTERVENTION_TEMPLATE.format(
        tool_name=tool_name,
        parameters_json=parameters_json,
        failure_count=failure_count,
        force_dom_summary_threshold=force_dom_summary_threshold,
    )


def build_action_blocked_message(tool_name: str, parameters_json: str, failure_count: int) -> str:
    return ACTION_BLOCKED_TEMPLATE.format(
        tool_name=tool_name,
        parameters_json=parameters_json,
        failure_count=failure_count,
    )


def build_download_pdf_already_done_message(available_files: List[str]) -> str:
    return DOWNLOAD_PDF_ALREADY_DONE_TEMPLATE.format(available_files=", ".join(available_files))


def build_context_switch_message(file_name: str) -> str:
    return CONTEXT_SWITCH_MESSAGE_TEMPLATE.format(file_name=file_name)


def build_download_context_switch_note(file_name: str) -> str:
    return DOWNLOAD_CONTEXT_SWITCH_NOTE_TEMPLATE.format(file_name=file_name)


def build_history_summary_request(summary: str) -> str:
    return HISTORY_SUMMARY_REQUEST_TEMPLATE.format(summary=summary)


def build_history_summarize_instruction() -> str:
    return HISTORY_SUMMARIZE_INSTRUCTION_TEMPLATE.format()
