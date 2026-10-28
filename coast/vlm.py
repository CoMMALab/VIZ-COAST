import os
import base64
from abc import ABC, abstractmethod
from typing import List, Optional, Dict, Any
from openai import OpenAI
from google import genai
from google.genai import types

OPENAI_API_KEY = None
GEMINI_API_KEY = None
QWEN_API_KEY = None

def _normalize_reasoning_effort(value: Optional[str]) -> Optional[str]:
    if value is None:
        return None
    effort = str(value).strip().lower()
    if effort in {"", "none", "off", "false", "0"}:
        return "none"
    if effort in {"low", "medium", "high"}:
        return effort
    return effort


def _extract_responses_api_reasoning(response: Any) -> str:
    """Extract reasoning-summary text from an OpenAI (or OpenAI-compatible)
    Responses API response. Empty if the model didn't emit a reasoning item
    (e.g. reasoning is disabled, or the model doesn't support summaries)."""
    chunks: List[str] = []
    for item in getattr(response, "output", None) or []:
        if getattr(item, "type", None) != "reasoning":
            continue
        for summary_part in getattr(item, "summary", None) or []:
            part_text = getattr(summary_part, "text", None)
            if part_text:
                chunks.append(part_text)
    return "\n".join(chunks).strip()

class VLMClient(ABC):
    @abstractmethod
    def generate(self, text: str, images: Optional[List[str]] = None, **kwargs) -> str:
        """
        Generate a response from the VLM.
        
        Args:
            text: The text prompt.
            images: A list of base64 encoded images (raw strings, without data URI prefix).
            **kwargs: Additional arguments passed to the underlying client (e.g. model).
            
        Returns:
            The generated text response.
        """
        pass

    @abstractmethod
    def generate_multiturn(
        self, messages: List[Dict[str, str]], images: Optional[List[str]] = None, **kwargs
    ) -> str:
        """
        Generate a response from a multi-turn conversation.

        Args:
            messages: Ordered conversation messages. Each item should include
                {"role": "user" | "assistant", "text": "..."}.
            images: Optional base64 images to attach to the first user turn.
            **kwargs: Additional provider-specific generation args.

        Returns:
            The generated text response.
        """
        pass

class OpenAIClient(VLMClient):
    def __init__(
        self,
        api_key: str = None,
        model: str = "gpt-5",
        reasoning_effort: Optional[str] = None,
    ):
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY") or OPENAI_API_KEY
        self.model = model
        self.reasoning_effort = _normalize_reasoning_effort(reasoning_effort)
        self.last_reasoning: Optional[str] = None

    def generate(self, text: str, images: Optional[List[str]] = None, **kwargs) -> str:
        model = kwargs.pop('model', self.model)
        reasoning_effort = _normalize_reasoning_effort(
            kwargs.pop("reasoning_effort", self.reasoning_effort)
        )

        content = [{"type": "input_text", "text": text}]

        if images:
            for img_base64 in images:
                # Check if it already has the prefix
                if img_base64.startswith("data:image"):
                    url = img_base64
                else:
                    url = f"data:image/png;base64,{img_base64}"
                content.append({"type": "input_image", "image_url": url})

        # Create a fresh client per call so no idle SSL connection persists
        # between requests.  PyBullet's native OpenGL threads (Intel Mesa)
        # corrupt the httpx connection-pool's SSL state while the connection
        # sits idle, leading to a segfault on the next ssl.read().
        with OpenAI(api_key=self.api_key) as client:
            request_kwargs = dict(kwargs)
            if reasoning_effort and reasoning_effort != "none":
                request_kwargs["reasoning"] = {"effort": reasoning_effort, "summary": "auto"}
            response = client.responses.create(
                model=model,
                input=[
                    {
                        "role": "user",
                        "content": content,
                    }
                ],
                **request_kwargs
            )
            self.last_reasoning = _extract_responses_api_reasoning(response)
            return response.output_text

    def generate_multiturn(
        self, messages: List[Dict[str, str]], images: Optional[List[str]] = None, **kwargs
    ) -> str:
        model = kwargs.pop("model", self.model)
        reasoning_effort = _normalize_reasoning_effort(
            kwargs.pop("reasoning_effort", self.reasoning_effort)
        )

        input_messages: List[Dict[str, Any]] = []
        attached_images = False
        for message in messages:
            role = message.get("role", "user")
            text = message.get("text", "")
            # OpenAI Responses API expects user content as input_text and
            # assistant history content as output_text.
            text_type = "output_text" if role == "assistant" else "input_text"
            content: List[Dict[str, Any]] = [{"type": text_type, "text": text}]

            if images and role == "user" and not attached_images:
                for img_base64 in images:
                    if img_base64.startswith("data:image"):
                        url = img_base64
                    else:
                        url = f"data:image/png;base64,{img_base64}"
                    content.append({"type": "input_image", "image_url": url})
                attached_images = True

            mapped_role = "assistant" if role == "assistant" else "user"
            input_messages.append({"role": mapped_role, "content": content})

        with OpenAI(api_key=self.api_key) as client:
            request_kwargs = dict(kwargs)
            if reasoning_effort and reasoning_effort != "none":
                request_kwargs["reasoning"] = {"effort": reasoning_effort, "summary": "auto"}
            response = client.responses.create(
                model=model,
                input=input_messages,
                **request_kwargs,
            )
            self.last_reasoning = _extract_responses_api_reasoning(response)
            return response.output_text

class GeminiClient(VLMClient):
    def __init__(
        self,
        api_key: str = None,
        model: str = "gemini-2.5-flash",
        reasoning_effort: Optional[str] = None,
    ):
        self.api_key = api_key or os.environ.get("GEMINI_API_KEY") or GEMINI_API_KEY
        self.client = genai.Client(api_key=self.api_key)
        self.model = model
        self.reasoning_effort = _normalize_reasoning_effort(reasoning_effort)
        self.last_reasoning: Optional[str] = None

    @staticmethod
    def _is_gemini_3_model(model: str) -> bool:
        model_l = (model or "").strip().lower()
        return ("gemini-3" in model_l) or ("gemini 3" in model_l)

    @staticmethod
    def _gemini_reasoning_config(model: str, reasoning_effort: Optional[str]):
        if not reasoning_effort:
            return None

        if GeminiClient._is_gemini_3_model(model):
            # Gemini 3 uses thinkingLevel.
            level_map = {"none": "minimal", "low": "low", "medium": "medium", "high": "high"}
            level = level_map.get(reasoning_effort)
            if level is None:
                return None
            return {"thinkingConfig": {"thinkingLevel": level, "includeThoughts": True}}

        # Older Gemini models keep using numeric thinking_budget.
        budget_map = {"none": 0, "low": 256, "medium": 1024, "high": 2048}
        budget = budget_map.get(reasoning_effort)
        if budget is None:
            return None
        return types.GenerateContentConfig(
            thinking_config=types.ThinkingConfig(thinking_budget=budget, include_thoughts=True)
        )

    @staticmethod
    def _extract_reasoning(response: Any) -> str:
        """Extract thought-summary text from a Gemini response (only present
        when the thinking config sets includeThoughts/include_thoughts)."""
        chunks: List[str] = []
        candidates = getattr(response, "candidates", None) or []
        content = getattr(candidates[0], "content", None) if candidates else None
        for part in getattr(content, "parts", None) or []:
            if getattr(part, "thought", False):
                text = getattr(part, "text", None)
                if text:
                    chunks.append(text)
        return "\n".join(chunks).strip()

    def generate(self, text: str, images: Optional[List[str]] = None, **kwargs) -> str:
        model = kwargs.pop('model', self.model)
        reasoning_effort = _normalize_reasoning_effort(
            kwargs.pop("reasoning_effort", self.reasoning_effort)
        )
        
        contents = [text]
        
        if images:
            for img_base64 in images:
                mime_type = "image/png" # Default assumption
                encoded = img_base64
                
                if img_base64.startswith("data:image"):
                    try:
                        header, encoded = img_base64.split(",", 1)
                        # Extract mime type from header like "data:image/png;base64"
                        mime_type = header.split(":")[1].split(";")[0]
                    except Exception:
                        pass
                
                image_bytes = base64.b64decode(encoded)
                contents.append(types.Part.from_bytes(data=image_bytes, mime_type=mime_type))
        
        request_kwargs = dict(kwargs)
        gemini_config = self._gemini_reasoning_config(model, reasoning_effort)
        if gemini_config is not None:
            request_kwargs.setdefault("config", gemini_config)

        response = self.client.models.generate_content(
            model=model,
            contents=contents,
            **request_kwargs
        )
        self.last_reasoning = self._extract_reasoning(response)
        return response.text

    def generate_multiturn(
        self, messages: List[Dict[str, str]], images: Optional[List[str]] = None, **kwargs
    ) -> str:
        model = kwargs.pop("model", self.model)
        reasoning_effort = _normalize_reasoning_effort(
            kwargs.pop("reasoning_effort", self.reasoning_effort)
        )
        contents: List[Dict[str, Any]] = []
        attached_images = False

        for message in messages:
            role = message.get("role", "user")
            text = message.get("text", "")
            mapped_role = "model" if role == "assistant" else "user"
            parts: List[Any] = [{"text": text}]

            if images and mapped_role == "user" and not attached_images:
                for img_base64 in images:
                    mime_type = "image/png"
                    encoded = img_base64
                    if img_base64.startswith("data:image"):
                        try:
                            header, encoded = img_base64.split(",", 1)
                            mime_type = header.split(":")[1].split(";")[0]
                        except Exception:
                            pass
                    image_bytes = base64.b64decode(encoded)
                    parts.append(types.Part.from_bytes(data=image_bytes, mime_type=mime_type))
                attached_images = True

            contents.append({"role": mapped_role, "parts": parts})

        request_kwargs = dict(kwargs)
        gemini_config = self._gemini_reasoning_config(model, reasoning_effort)
        if gemini_config is not None:
            request_kwargs.setdefault("config", gemini_config)

        response = self.client.models.generate_content(
            model=model,
            contents=contents,
            **request_kwargs,
        )
        self.last_reasoning = self._extract_reasoning(response)
        return response.text

class QwenClient(VLMClient):
    def __init__(
        self,
        api_key: str = None,
        model: str = "qwen3.5-flash",
        reasoning_effort: Optional[str] = None,
    ):
        self.api_key = api_key or os.environ.get("QWEN_API_KEY") or QWEN_API_KEY
        # self.base_url = "https://dashscope-intl.aliyuncs.com/compatible-mode/v1"
        self.base_url="https://dashscope-intl.aliyuncs.com/api/v2/apps/protocols/compatible-mode/v1"
        self.model = model
        self.reasoning_effort = _normalize_reasoning_effort(reasoning_effort)
        self.last_reasoning: Optional[str] = None

    @staticmethod
    def _thinking_enabled(reasoning_effort: Optional[str]) -> bool:
        effort = _normalize_reasoning_effort(reasoning_effort)
        return bool(effort and effort != "none")

    def _stream_text(self, client: OpenAI, **request_kwargs) -> str:
        """Run a request in streaming mode, accumulating the answer and the thinking
        summary. Dashscope frequently drops the message item from a non-streaming
        thinking response -- only the reasoning item comes back, so the caller sees
        an empty string (qwen3.7-plus: ~3/5 of calls, qwen3.5-flash: ~1/5). Streaming
        returns both parts reliably."""
        text_chunks: List[str] = []
        reasoning_chunks: List[str] = []
        with client.responses.stream(**request_kwargs) as stream:
            for event in stream:
                event_type = getattr(event, "type", "")
                if event_type == "response.output_text.delta":
                    text_chunks.append(getattr(event, "delta", "") or "")
                elif event_type == "response.reasoning_summary_text.delta":
                    reasoning_chunks.append(getattr(event, "delta", "") or "")
        self.last_reasoning = "".join(reasoning_chunks).strip()
        return "".join(text_chunks)

    @staticmethod
    def _extract_output_text(response: Any) -> str:
        text = getattr(response, "output_text", None)
        if text:
            return text
        # Fallback for providers that don't populate output_text.
        outputs = getattr(response, "output", None) or []
        chunks: List[str] = []
        for item in outputs:
            for content in getattr(item, "content", []) or []:
                ctext = getattr(content, "text", None)
                if ctext:
                    chunks.append(ctext)
        return "\n".join(chunks).strip()

    @staticmethod
    def _extract_reasoning(response: Any) -> str:
        """Best-effort extraction of thinking/reasoning content from Qwen's
        OpenAI-compatible Responses API. Covers a separate "reasoning" output
        item (OpenAI-style) as well as reasoning-flagged content parts inline
        in a message item, since Dashscope's exact schema isn't documented
        here."""
        chunks: List[str] = []
        for item in getattr(response, "output", None) or []:
            if getattr(item, "type", None) == "reasoning":
                for summary_part in getattr(item, "summary", None) or []:
                    part_text = getattr(summary_part, "text", None)
                    if part_text:
                        chunks.append(part_text)
                continue
            for content in getattr(item, "content", None) or []:
                content_type = str(getattr(content, "type", "") or "")
                if "reasoning" in content_type.lower() or "thinking" in content_type.lower():
                    ctext = getattr(content, "text", None)
                    if ctext:
                        chunks.append(ctext)
        return "\n".join(chunks).strip()

    def generate(self, text: str, images: Optional[List[str]] = None, **kwargs) -> str:
        model = kwargs.pop('model', self.model)
        reasoning_effort = _normalize_reasoning_effort(
            kwargs.pop("reasoning_effort", self.reasoning_effort)
        )

        content = [{"type": "input_text", "text": text}]
        
        if images:
            for img_base64 in images:
                if img_base64.startswith("data:image"):
                    url = img_base64
                else:
                    url = f"data:image/png;base64,{img_base64}"

                content.append({"type": "input_image", "image_url": url})

        # Fresh client per call — same SSL corruption avoidance as OpenAIClient
        with OpenAI(api_key=self.api_key, base_url=self.base_url) as client:
            request_kwargs = dict(kwargs)
            thinking_enabled = self._thinking_enabled(reasoning_effort)
            extra_body = dict(request_kwargs.pop("extra_body", {}) or {})
            # Qwen 3.5 thinking models use enable_thinking; keep it explicit.
            extra_body["enable_thinking"] = thinking_enabled
            request_kwargs["extra_body"] = extra_body
            input_messages = [
                {
                    "role": "user",
                    "content": content,
                }
            ]
            if thinking_enabled:
                return self._stream_text(
                    client, model=model, input=input_messages, **request_kwargs
                )
            response = client.responses.create(
                model=model,
                input=input_messages,
                **request_kwargs
            )
            self.last_reasoning = self._extract_reasoning(response)
            return self._extract_output_text(response)

    def generate_multiturn(
        self, messages: List[Dict[str, str]], images: Optional[List[str]] = None, **kwargs
    ) -> str:
        model = kwargs.pop("model", self.model)
        reasoning_effort = _normalize_reasoning_effort(
            kwargs.pop("reasoning_effort", self.reasoning_effort)
        )

        input_messages: List[Dict[str, Any]] = []
        attached_images = False
        for message in messages:
            role = message.get("role", "user")
            text = message.get("text", "")
            mapped_role = "assistant" if role == "assistant" else "user"
            text_type = "output_text" if mapped_role == "assistant" else "input_text"

            content: List[Dict[str, Any]] = [{"type": text_type, "text": text}]
            if images and mapped_role == "user" and not attached_images:
                for img_base64 in images:
                    if img_base64.startswith("data:image"):
                        url = img_base64
                    else:
                        url = f"data:image/png;base64,{img_base64}"
                    content.append({"type": "input_image", "image_url": url})
                attached_images = True

            input_messages.append({"role": mapped_role, "content": content})

        with OpenAI(api_key=self.api_key, base_url=self.base_url) as client:
            request_kwargs = dict(kwargs)
            thinking_enabled = self._thinking_enabled(reasoning_effort)
            extra_body = dict(request_kwargs.pop("extra_body", {}) or {})
            # Qwen 3.5 thinking models use enable_thinking; keep it explicit.
            extra_body["enable_thinking"] = thinking_enabled
            request_kwargs["extra_body"] = extra_body
            if thinking_enabled:
                return self._stream_text(
                    client, model=model, input=input_messages, **request_kwargs
                )
            response = client.responses.create(
                model=model,
                input=input_messages,
                **request_kwargs,
            )
            self.last_reasoning = self._extract_reasoning(response)
            return self._extract_output_text(response)

def get_vlm_client(provider: str = "openai", **kwargs) -> VLMClient:
    """
    Factory to get a VLM client.
    
    Args:
        provider: The provider name ('openai', 'gemini', 'qwen', etc.)
        **kwargs: Arguments to pass to the client constructor.
    """
    provider = provider.lower()
    if provider == "openai":
        return OpenAIClient(**kwargs)
    elif provider == "gemini":
        return GeminiClient(**kwargs)
    elif provider == "qwen":
        return QwenClient(**kwargs)
    else:
        raise ValueError(f"Unknown VLM provider: {provider}")
