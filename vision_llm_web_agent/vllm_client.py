"""
Vision Language Model Client
Interfaces with vision LLMs via OpenAI-compatible API
"""

import base64
import json
import os
from io import BytesIO
from pathlib import Path
from typing import Optional, Dict, List, Any

from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from openai import OpenAI
from PIL import Image
from .config.settings import (
    ARTIFACTS_DIR,
    KEEP_LAST_MESSAGES,
    MAX_CONTEXT_SUMMARY_CHARS,
    MAX_HISTORY_CHARS,
)
from .context_manager import ContextWindowManager
from .prompts import (
    build_agent_system_prompt,
    build_context_compression_prompt,
    build_current_state_message,
    build_json_retry_feedback,
    build_local_file_processing_note,
    build_text_summary_prompt,
    build_web_browsing_instruction,
)

# Load environment variables
load_dotenv()


class VLLMClient:
    """Client for Vision Language Models via LangChain and OpenAI-compatible APIs."""
    
    def __init__(
        self, 
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        language_model: Optional[str] = None,
        max_tokens: int = 8192,
        temperature: float = 0.7
    ):
        """
        Initialize VLLM client.
        
        Args:
            base_url: API base URL. Defaults to env OPENAI_BASE_URL or OpenAI's API
            api_key: API key. Defaults to env OPENAI_API_KEY
            model: Model name. Defaults to env OPENAI_MODEL or "gpt-4o"
            language_model: Language model name for DOM analysis. Defaults to env OPENAI_LANGUAGE_MODEL or same as model
            max_tokens: Maximum tokens in response
            temperature: Temperature for generation (0-2)
        
        Supports:
            - Local vLLM server (base_url="http://localhost:8000/v1")
            - OpenAI API (base_url="https://api.openai.com/v1", model="gpt-4o")
            - Other OpenAI-compatible APIs (Claude, Gemini via proxy, etc.)
        """
        self.base_url = base_url or os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
        self.api_key = api_key or os.getenv("OPENAI_API_KEY", "EMPTY")
        self.model = model or os.getenv("OPENAI_MODEL", "gpt-4o")
        self.language_model = language_model or os.getenv("OPENAI_LANGUAGE_MODEL", self.model)
        self.max_tokens = max_tokens
        self.temperature = temperature
        
        self.client = OpenAI(
            base_url=self.base_url,
            api_key=self.api_key
        )

        # LangChain model for agent planning while preserving the existing OpenAI
        # client interface used by DOM analyzer utilities.
        self.langchain_llm = ChatOpenAI(
            model=self.model,
            api_key=self.api_key,
            base_url=self.base_url,
            max_tokens=self.max_tokens,
            temperature=self.temperature
        )
        
        print(f"✅ VLLM Client initialized")
        print(f"   Base URL: {self.base_url}")
        print(f"   Model: {self.model}")

        self.context_manager = ContextWindowManager(
            max_history_chars=MAX_HISTORY_CHARS,
            keep_last_messages=KEEP_LAST_MESSAGES,
            max_summary_chars=MAX_CONTEXT_SUMMARY_CHARS,
        )

    def _to_langchain_messages(self, messages: List[Dict[str, Any]]) -> List[Any]:
        """Convert OpenAI-style message dicts to LangChain message objects."""
        lc_messages: List[Any] = []
        for msg in messages:
            role = msg.get("role")
            content = msg.get("content", "")
            if role == "system":
                lc_messages.append(SystemMessage(content=content))
            elif role == "assistant":
                lc_messages.append(AIMessage(content=content))
            else:
                lc_messages.append(HumanMessage(content=content))
        return lc_messages
    
    def encode_image(self, image_path: str, max_size: tuple = (1280, 720)) -> str:
        """
        Encode image to base64 data URL.
        
        Args:
            image_path: Path to the image file
            max_size: Maximum size (width, height) to resize image to
        
        Returns:
            Base64 encoded data URL
        """
        try:
            with Image.open(image_path) as img:
                # Resize if too large (to save tokens)
                if img.size[0] > max_size[0] or img.size[1] > max_size[1]:
                    img.thumbnail(max_size, Image.Resampling.LANCZOS)
                
                # Convert to RGB if needed
                if img.mode not in ('RGB', 'RGBA'):
                    img = img.convert('RGB')
                
                # Encode to base64
                buffered = BytesIO()
                img.save(buffered, format="PNG")
                img_str = base64.b64encode(buffered.getvalue()).decode()
                
                return f"data:image/png;base64,{img_str}"
        except Exception as e:
            raise ValueError(f"Failed to encode image {image_path}: {e}")
    
    def clean_messages_for_logging(self, messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Clean messages by removing image URLs for logging purposes.
        
        Args:
            messages: List of message dictionaries
        
        Returns:
            Cleaned messages with image URLs removed
        """
        cleaned_messages = []
        
        for message in messages:
            cleaned_message = message.copy()
            
            # Handle content that might contain image URLs
            if isinstance(cleaned_message.get('content'), list):
                # Multi-modal content (text + image)
                cleaned_content = []
                for item in cleaned_message['content']:
                    if item.get('type') == 'image_url':
                        # Replace image URL with placeholder
                        cleaned_content.append({
                            "type": "image_url",
                            "image_url": {
                                "url": "[IMAGE_DATA_REMOVED_FOR_LOGGING]",
                                "detail": item.get('image_url', {}).get('detail', 'high')
                            }
                        })
                    else:
                        # Keep text content as is
                        cleaned_content.append(item)
                cleaned_message['content'] = cleaned_content
            elif isinstance(cleaned_message.get('content'), str):
                # Text content - check if it contains base64 image data
                content = cleaned_message['content']
                if 'data:image/' in content and 'base64,' in content:
                    # Replace base64 image data with placeholder
                    import re
                    cleaned_content = re.sub(
                        r'data:image/[^;]+;base64,[A-Za-z0-9+/=]+',
                        '[IMAGE_DATA_REMOVED_FOR_LOGGING]',
                        content
                    )
                    cleaned_message['content'] = cleaned_content
            
            cleaned_messages.append(cleaned_message)
        
        return cleaned_messages

    def build_system_prompt(self, available_tools: List[Dict[str, Any]]) -> str:
        """Build system prompt with tool descriptions."""
        return build_agent_system_prompt(available_tools)

    def _summarize_for_context_compression(self, history_text: str, max_length: int) -> str:
        """Summarize history chunks to fit context window constraints."""
        try:
            prompt = build_context_compression_prompt(history_text, max_length)
            response = self.langchain_llm.invoke([HumanMessage(content=prompt)])
            content = response.content if isinstance(response.content, str) else str(response.content)
            return content[:max_length]
        except Exception as e:
            return f"Summary generation failed: {e}."
    
    def plan_next_action(
        self, 
        history: List[Dict[str, Any]], 
        state_info: Dict[str, Any],
        available_tools: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Analyze current state and plan next action.
        
        Args:
            history: Conversation history
            state_info: Current state (screenshot path, DOM, etc.)
            available_tools: List of available tool definitions
        
        Returns:
            Parsed response with action to take
        """
        # Build prompt with tool descriptions
        system_prompt = self.build_system_prompt(available_tools)
        
        # Prepare messages
        messages = [{"role": "system", "content": system_prompt}]
        
        # Add history
        # History now contains alternating assistant (tool call) and user (tool result) messages
        managed_history = self.context_manager.compress_history(
            history,
            self._summarize_for_context_compression,
        )
        if len(managed_history) < len(history):
            print(
                f"   🗜️ Context compressed: {len(history)} -> {len(managed_history)} messages "
                f"(history chars <= {self.context_manager.max_history_chars})"
            )

        for msg in managed_history:
            if msg['role'] in ['user', 'assistant']:
                messages.append(msg)
        
        # Add current state with vision input (if screenshot available)
        current_state_content = []
        
        # Check if screenshot is available and relevant
        screenshot_available = state_info.get('screenshot_available', False)
        screenshot_path = state_info.get('screenshot')
        context_mode = state_info.get('context_mode', 'web_browsing')
        
        # Only include screenshot if in web browsing mode
        if screenshot_available and screenshot_path and context_mode == 'web_browsing':
            try:
                # Add screenshot
                current_state_content.append({
                    "type": "image_url",
                    "image_url": {
                        "url": self.encode_image(screenshot_path),
                        "detail": "high"
                    }
                })
            except Exception as e:
                print(f"   ⚠️  Failed to encode screenshot: {e}")
                # Continue without screenshot
        
        # Add text description in JSON format
        dom_text = state_info.get('dom', 'N/A')
        round_num = state_info.get('round', 0)
        context_mode = state_info.get('context_mode', 'web_browsing')
        
        current_state_json = {
            "round": round_num,
            "context_mode": context_mode,
            "screenshot_available": screenshot_available,
            "dom_summary": dom_text,
            "instruction": state_info.get('instruction', build_web_browsing_instruction())
        }
        
        # Add PDF detection warning if PDF page detected
        if state_info.get('pdf_detected'):
            current_state_json["pdf_detected"] = True
            current_state_json["pdf_url"] = state_info.get('pdf_url', '')
            # Make instruction more prominent
            current_state_json["instruction"] = state_info.get('instruction', "🚨 PDF PAGE DETECTED! Download it using download_pdf(url=\"current\", file_name=\"report.pdf\")")
        
        # Add context-specific information
        if context_mode == "local_file_processing":
            current_state_json["available_local_files"] = state_info.get('available_local_files', [])
            extracted_images = state_info.get('extracted_images', [])
            current_state_json["extracted_images"] = extracted_images
            current_state_json["note"] = build_local_file_processing_note()
            
            # Add extracted images to VLLM input so it can "see" them
            # Add images BEFORE text content so VLLM can see them first
            # Limit to first 20 images to avoid overwhelming VLLM with too many images at once
            MAX_IMAGES_TO_SHOW = 20
            if extracted_images:
                from .config.settings import get_session_artifacts_dir
                session_artifacts_dir = get_session_artifacts_dir()
                images_to_show = extracted_images[:MAX_IMAGES_TO_SHOW]
                if len(extracted_images) > MAX_IMAGES_TO_SHOW:
                    print(f"   ⚠️  Limiting to first {MAX_IMAGES_TO_SHOW} images (total: {len(extracted_images)})")
                    current_state_json["note"] += f"\n⚠️ NOTE: Only showing first {MAX_IMAGES_TO_SHOW} of {len(extracted_images)} extracted images. If you need a specific image (e.g., Figure 1), extract images from the specific page instead of all pages."
                
                image_count = 0
                for img_rel_path in images_to_show:
                    try:
                        # Handle both relative paths and full paths
                        if Path(img_rel_path).is_absolute():
                            img_full_path = Path(img_rel_path)
                        else:
                            img_full_path = session_artifacts_dir / img_rel_path
                        
                        if img_full_path.exists() and img_full_path.is_file():
                            # Add image to current state content (before text)
                            current_state_content.insert(image_count, {
                                "type": "image_url",
                                "image_url": {
                                    "url": self.encode_image(str(img_full_path)),
                                    "detail": "high"
                                }
                            })
                            image_count += 1
                            print(f"   🖼️  Added extracted image to VLLM input: {img_rel_path}")
                        else:
                            print(f"   ⚠️  Image file not found: {img_full_path}")
                    except Exception as e:
                        print(f"   ⚠️  Failed to encode extracted image {img_rel_path}: {e}")
                        import traceback
                        traceback.print_exc()
        
        current_state_content.append({
            "type": "text",
            "text": build_current_state_message(
                json.dumps(current_state_json, ensure_ascii=False, indent=2)
            )
        })
        
        messages.append({
            "role": "user",
            "content": current_state_content
        })
        
        # Debug: Print messages being sent to VLLM
        print(f"\n📤 Sending to VLLM:")
        print(f"   Model: {self.model}")
        print(f"   Messages count: {len(messages)}")
        for i, msg in enumerate(messages):
            if msg['role'] == 'system':
                print(f"   [{i}] System: {msg['content'][:200]}...")
            elif msg['role'] == 'user':
                if isinstance(msg['content'], list):
                    print(f"   [{i}] User: {len(msg['content'])} content items (image + text)")
                else:
                    print(f"   [{i}] User: {msg['content'][:200]}...")
            else:
                print(f"   [{i}] {msg['role']}: {str(msg['content'])[:200]}...")
        
        # Call the model
        try:
            lc_messages = self._to_langchain_messages(messages)
            response = self.langchain_llm.invoke(lc_messages)

            # Parse response content
            content = response.content if isinstance(response.content, str) else str(response.content)
            
            # Debug: Print raw VLLM output
            print(f"\n🔍 VLLM Raw Output:")
            print("=" * 80)
            print(content)
            print("=" * 80)
            
            # Parse and debug the result
            parsed_result = self.parse_response(content)
            
            # Add raw input and output to parsed result
            parsed_result["vllm_raw_input"] = {
                "model": self.model,
                "messages": self.clean_messages_for_logging(messages),
                "history_messages_original": len(history),
                "history_messages_used": len(managed_history),
                "max_tokens": self.max_tokens,
                "temperature": self.temperature
            }
            parsed_result["vllm_raw_output"] = {
                "content": content,
                "response_object": {
                    "type": "langchain_ai_message",
                    "id": response.id,
                    "response_metadata": response.response_metadata,
                    "usage_metadata": response.usage_metadata,
                    "additional_kwargs": response.additional_kwargs
                }
            }
            
            # Debug: Print parsed result
            print(f"\n📋 Parsed Result:")
            print(f"   Is Complete: {parsed_result.get('is_complete', 'N/A')}")
            print(f"   Final Answer: {parsed_result.get('final_answer', 'N/A')}")
            print(f"   Tool Calls: {len(parsed_result.get('tool_calls', []))}")
            if parsed_result.get('tool_calls'):
                for i, tool_call in enumerate(parsed_result['tool_calls']):
                    print(f"     [{i}] {tool_call.get('name', 'unknown')}: {tool_call.get('params', {})}")
            if parsed_result.get('error'):
                print(f"   Error: {parsed_result['error']}")
                print(f"\n🔄 JSON Parse Error - Retrying with error feedback...")
                
                # Add assistant's invalid response to messages
                assistant_invalid_response = {
                    "role": "assistant",
                    "content": content
                }
                
                # Add error feedback as user message
                error_feedback = {
                    "role": "user",
                    "content": build_json_retry_feedback(f"{content[:500]}...")
                }
                
                # Retry with error feedback
                print(f"\n🔄 Retrying with error feedback...")
                retry_messages = messages + [assistant_invalid_response, error_feedback]
                retry_lc_messages = self._to_langchain_messages(retry_messages)
                retry_response = self.langchain_llm.invoke(retry_lc_messages)

                retry_content = (
                    retry_response.content
                    if isinstance(retry_response.content, str)
                    else str(retry_response.content)
                )
                print(f"\n🔍 VLLM Retry Output:")
                print("=" * 80)
                print(retry_content)
                print("=" * 80)
                
                # Parse retry result
                parsed_result = self.parse_response(retry_content)
                
                # Add raw input and output for retry
                parsed_result["vllm_raw_input"] = {
                    "model": self.model,
                    "messages": self.clean_messages_for_logging(retry_messages),
                    "history_messages_original": len(history),
                    "history_messages_used": len(managed_history),
                    "max_tokens": self.max_tokens,
                    "temperature": self.temperature
                }
                parsed_result["vllm_raw_output"] = {
                    "content": retry_content,
                    "response_object": {
                        "type": "langchain_ai_message",
                        "id": retry_response.id,
                        "response_metadata": retry_response.response_metadata,
                        "usage_metadata": retry_response.usage_metadata,
                        "additional_kwargs": retry_response.additional_kwargs
                    }
                }
                print(f"\n📋 Retry Parsed Result:")
                print(f"   Is Complete: {parsed_result.get('is_complete', 'N/A')}")
                print(f"   Final Answer: {parsed_result.get('final_answer', 'N/A')}")
                print(f"   Tool Calls: {len(parsed_result.get('tool_calls', []))}")
                if parsed_result.get('tool_calls'):
                    for i, tool_call in enumerate(parsed_result['tool_calls']):
                        print(f"     [{i}] {tool_call.get('name', 'unknown')}: {tool_call.get('params', {})}")
                if parsed_result.get('error'):
                    print(f"   Error: {parsed_result['error']}")
                print()
            else:
                print()
            
            return parsed_result
        
        except Exception as e:
            return {
                "error": str(e),
                "is_complete": False
            }
    
    def parse_response(self, content: str) -> Dict[str, Any]:
        """
        Parse model response to extract tool calls or completion status.
        
        Args:
            content: Raw response content
        
        Returns:
            Parsed response dict
        """
        # Clean the content first
        content = content.strip()
        
        # Try to find JSON in the response
        try:
            # Look for JSON blocks
            start_idx = content.find("{")
            end_idx = content.rfind("}") + 1
            if start_idx != -1 and end_idx > start_idx:
                json_str = content[start_idx:end_idx]
            else:
                json_str = content
            
            # Clean up the JSON string
            json_str = json_str.strip()
            
            # Parse JSON
            parsed = json.loads(json_str)
            
            # Check if task is complete
            if parsed.get("status") == "complete":
                return {
                    "is_complete": True,
                    "final_answer": parsed.get("result", "Task completed"),
                    "thought": parsed.get("thought", ""),
                    "raw_response": content
                }
            
            # Extract tool call
            if "tool" in parsed:
                return {
                    "is_complete": False,
                    "tool_calls": [{
                        "name": parsed["tool"],
                        "params": parsed.get("parameters", {})
                    }],
                    "thought": parsed.get("thought", ""),
                    "next": parsed.get("next", ""),
                    "raw_response": content
                }
            
            # If no clear action, return the content
            return {
                "is_complete": False,
                "error": "No clear tool call or completion in response",
                "raw_response": content
            }
        
        except json.JSONDecodeError as e:
            # Try to extract information from text format like "Tool: scroll\nResult: ..."
            if "Tool:" in content and "Result:" in content:
                lines = content.strip().split('\n')
                tool_name = None
                for line in lines:
                    if line.startswith("Tool:"):
                        tool_name = line.replace("Tool:", "").strip()
                        break
                
                if tool_name:
                    return {
                        "is_complete": False,
                        "tool_calls": [{
                            "name": tool_name,
                            "params": {}
                        }],
                        "thought": f"Extracted tool from text format: {tool_name}",
                        "raw_response": content
                    }
            
            # If all else fails, return error
            return {
                "is_complete": False,
                "error": f"Failed to parse JSON: {e}",
                "raw_response": content
            }
    
    def summarize_text(self, text: str, max_length: int = 500) -> str:
        """
        Generate a summary of the given text using the LLM.
        
        Args:
            text: Text content to summarize
            max_length: Maximum length of the summary in characters
        
        Returns:
            Summary text
        """
        try:
            # Truncate text if too long (to avoid token limits)
            # Keep first 8000 characters for summarization
            text_to_summarize = text[:8000] if len(text) > 8000 else text
            
            prompt = build_text_summary_prompt(text_to_summarize, max_length)
            
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=min(self.max_tokens, max_length // 2),  # Rough token estimate
                temperature=0.3  # Lower temperature for more focused summaries
            )
            
            summary = response.choices[0].message.content.strip()
            return summary
        except Exception as e:
            return f"❌ Failed to generate summary: {str(e)}"
    
    def test_connection(self) -> bool:
        """
        Test if the API connection is working.
        
        Returns:
            True if connection successful, False otherwise
        """
        try:
            response = self.langchain_llm.invoke([HumanMessage(content="Hello")])
            content = response.content if isinstance(response.content, str) else str(response.content)
            print(f"✅ API connection successful: {content[:50]}")
            return True
        except Exception as e:
            print(f"❌ API connection failed: {e}")
            return False


# Utility function to create client from config
def create_vllm_client_from_env() -> VLLMClient:
    """
    Create VLLM client from environment variables.
    
    Environment variables:
        OPENAI_BASE_URL: API base URL
        OPENAI_API_KEY: API key
        OPENAI_MODEL: Model name
    
    Returns:
        Configured VLLMClient instance
    """
    return VLLMClient(
        base_url=os.getenv("OPENAI_BASE_URL"),
        api_key=os.getenv("OPENAI_API_KEY"),
        model=os.getenv("OPENAI_MODEL")
    )


if __name__ == "__main__":
    # Test the client
    print("Testing VLLM Client...")
    
    client = VLLMClient(
        base_url=os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1"),
        api_key=os.getenv("OPENAI_API_KEY", "EMPTY"),
        model=os.getenv("OPENAI_MODEL", "gpt-4o")
    )
    
    # Test connection
    client.test_connection()
    
    # Test image encoding (if test screenshot exists)
    test_screenshot = ARTIFACTS_DIR / "test_screenshot.png"
    if test_screenshot.exists():
        print("\nTesting image encoding...")
        encoded = client.encode_image(str(test_screenshot))
        print(f"✅ Image encoded: {len(encoded)} characters")
    
    print("\n✅ VLLM Client ready!")

