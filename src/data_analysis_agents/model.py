"""A reusable local Qwen3-VL adapter, imported without loading inference libraries."""

from __future__ import annotations

from copy import deepcopy
from threading import RLock
from time import monotonic

from .types import GenerationResult, ModelConfig, TokenUsage
from .schema import generation_schema


class GenerationTimeout(TimeoutError):
    """Partial streams are diagnostics, never accepted as complete model output."""
    def __init__(self, message, *, partial_text='', partial_reasoning='', request_id=None):
        super().__init__(message)
        self.partial_text, self.partial_reasoning = partial_text, partial_reasoning
        self.request_id = request_id
        self.retry_safe = False


class GenerationConnectionError(ConnectionError):
    def __init__(self, message, *, partial_text='', partial_reasoning='', request_id=None):
        super().__init__(message)
        self.partial_text, self.partial_reasoning = partial_text, partial_reasoning
        self.request_id, self.retry_safe = request_id, False


class LocalModel:
    """Share one GPU model across Planner and task workers; generation is serialized."""

    def __new__(cls, config: ModelConfig | None = None, seed: int = 42):
        """Select the local INT4 service adapter without loading another GPU model."""
        if config is not None and config.backend == "vllm":
            return VLLMModel(config, seed)
        return super().__new__(cls)

    def __init__(self, config: ModelConfig, seed: int = 42) -> None:
        """Load already downloaded weights on a CUDA compute node, never on the login CPU."""
        import torch
        from transformers import AutoProcessor, Qwen3VLForConditionalGeneration, set_seed

        self.config = ModelConfig.model_validate(config)
        if not torch.cuda.is_available():
            raise RuntimeError("LocalModel requires an allocated CUDA compute node.")
        if config.quantization is not None:
            raise ValueError("quantization requires a separately validated adapter")
        if config.device_map == "cpu":
            raise ValueError("CPU inference is not configured for this cluster")
        set_seed(seed)
        self._torch, self._lock = torch, RLock()
        self._processor_lock = RLock()
        source = config.model_path or config.model_id
        self.processor = AutoProcessor.from_pretrained(source, revision=config.revision,
                                                       local_files_only=True, max_pixels=config.image_max_pixels)
        self.tokenizer = self.processor.tokenizer
        self.model = Qwen3VLForConditionalGeneration.from_pretrained(
            source, revision=config.revision, local_files_only=True,
            dtype=config.dtype, device_map=config.device_map, attn_implementation="sdpa",
        ).eval()
        self._tokenizer_data = None
        if self.config.structured_output:
            self._constraint_tokenizer()

    def _constraint_tokenizer(self):
        """Build reusable tokenizer data once; parser states remain private to each generation."""
        from lmformatenforcer.integrations.transformers import build_token_enforcer_tokenizer_data

        if self._tokenizer_data is None:
            self._tokenizer_data = build_token_enforcer_tokenizer_data(self.tokenizer)
            self._tokenizer_data.eos_token_id = self.model.generation_config.eos_token_id
        return self._tokenizer_data

    def _prefix_function(self, output_schema: dict):
        """Construct schema-guided token filtering without weakening unsupported schemas."""
        from lmformatenforcer import CharacterLevelParserConfig, JsonSchemaParser
        from lmformatenforcer.integrations.transformers import build_transformers_prefix_allowed_tokens_fn

        if not isinstance(output_schema, dict):
            raise ValueError("structured_output requires an explicit output_schema")
        parser = JsonSchemaParser(generation_schema(output_schema), config=CharacterLevelParserConfig(
            force_json_field_order=False, max_json_array_length=128))
        return build_transformers_prefix_allowed_tokens_fn(self._constraint_tokenizer(), parser)

    def _prepare(self, messages: list[dict]):
        """Use the model's actual chat template and image processor for token accounting."""
        normalized = [
            {**message, "content": [{"type": "text", "text": message["content"]}]
             if isinstance(message["content"], str) else message["content"]}
            for message in messages
        ]
        for message in normalized:
            content = message["content"]
            if isinstance(content, list):
                for item in content:
                    if item.get("type") == "image":
                        image = item.get("image", item.get("url", ""))
                        if not isinstance(image, str) or image.startswith(("http:", "https:", "data:")):
                            raise ValueError("only host-authorized local image files are supported")
        return self.processor.apply_chat_template(
            normalized, add_generation_prompt=True, tokenize=True,
            return_dict=True, return_tensors="pt",
            processor_kwargs={"images_kwargs": {"max_pixels": self.config.image_max_pixels}},
            enable_thinking=self.config.enable_thinking,
        )

    def count_tokens(self, messages: list[dict]) -> int:
        """Count processed input tokens, including expanded visual tokens, without inference."""
        with self._processor_lock:
            inputs = self._prepare(messages)
            return int(inputs["attention_mask"].sum().item())

    def _generation_config(self):
        """Build runtime decoding settings without modifying downloaded defaults or shared state."""
        config = deepcopy(self.model.generation_config)
        config.do_sample = self.config.do_sample
        if config.do_sample:
            config.temperature = self.config.temperature
        else:
            config.temperature = config.top_p = config.top_k = None
        return config

    def generate(self, messages: list[dict], *, max_new_tokens: int,
                 remaining_seconds: float, output_schema: dict | None = None) -> GenerationResult:
        """Generate within the remaining wall time and report actual input/output tokens."""
        from transformers import StoppingCriteria, StoppingCriteriaList

        started = monotonic()
        if remaining_seconds <= 0:
            raise TimeoutError("model deadline already elapsed")
        if not self._lock.acquire(timeout=remaining_seconds):
            raise TimeoutError("model deadline elapsed waiting for the shared GPU")
        try:
            with self._processor_lock:
                inputs = self._prepare(messages)
            prefix = self._prefix_function(output_schema) if (
                self.config.structured_output and output_schema is not None) else None
            available = remaining_seconds - (monotonic() - started)
            if available <= 0:
                raise TimeoutError("model deadline elapsed preparing inputs")
            input_count = int(inputs["attention_mask"].sum().item())
            if input_count + max_new_tokens > self.config.context_limit:
                raise ValueError("processed input plus reserved output exceeds context_limit")
            deadline = started + remaining_seconds

            class Deadline(StoppingCriteria):
                """Stop at a decoding boundary when the shared phase deadline expires."""

                def __call__(self, input_ids, scores, **kwargs):
                    """Return whether the wall-clock deadline has elapsed."""
                    return monotonic() >= deadline

            inputs = inputs.to(self.model.device)
            options = {"max_new_tokens": max_new_tokens, "generation_config": self._generation_config(),
                       "use_model_defaults": False,
                       "stopping_criteria": StoppingCriteriaList([Deadline()])}
            if prefix is not None:
                options["prefix_allowed_tokens_fn"] = prefix
            with self._torch.inference_mode():
                output = self.model.generate(**inputs, **options)
            generated = output[0, inputs["input_ids"].shape[-1]:].tolist()
            eos = self.model.generation_config.eos_token_id
            eos_ids = set(eos if isinstance(eos, list) else [eos])
            finish = "stop" if generated and generated[-1] in eos_ids else "length"
            if generated and generated[-1] in eos_ids:
                count = len(generated)
            else:
                padding = self.tokenizer.pad_token_id
                count = len(generated)
                while count and generated[count - 1] == padding:
                    count -= 1
            return GenerationResult(
                text=self.processor.decode(generated, skip_special_tokens=True),
                tokens=TokenUsage(input_tokens=input_count, output_tokens=count),
                finish_reason=finish,
            )
        finally:
            self._lock.release()


class VLLMModel:
    """Call a loopback vLLM service that owns the quantized GPU model."""

    def __init__(self, config: ModelConfig, seed: int = 42) -> None:
        """Load only the processor and verify the expected locally served model."""
        from transformers import AutoProcessor
        from urllib.request import ProxyHandler, build_opener

        import os
        self.config = ModelConfig.model_validate({**config.model_dump(),
            "api_base": os.environ.get("AGENT_MODEL_API_BASE", config.api_base)})
        self.seed = seed
        self._lock, self._processor_lock = RLock(), RLock()
        self._http = build_opener(ProxyHandler({}))
        self.processor = AutoProcessor.from_pretrained(config.model_path or config.model_id,
                                                       local_files_only=True,
                                                       max_pixels=config.image_max_pixels)
        self.tokenizer = self.processor.tokenizer
        models = self._request("models", timeout=10)
        if config.served_model_name not in {entry["id"] for entry in models.get("data", [])}:
            raise ValueError("The local vLLM service is not serving the configured model")

    def _request(self, path: str, payload: dict | None = None, *, timeout: float) -> dict:
        """Use deadline-bounded loopback HTTP and surface complete server error messages."""
        import json
        from urllib.error import HTTPError, URLError
        from urllib.request import Request

        data = json.dumps(payload, ensure_ascii=False).encode() if payload is not None else None
        request = Request(self.config.api_base.rstrip("/") + "/" + path, data=data,
                          headers={"Content-Type": "application/json"})
        try:
            with self._http.open(request, timeout=timeout) as response:
                return json.load(response)
        except HTTPError as error:
            raise ValueError("vLLM HTTP error: " + error.read().decode(errors="replace")[:2000]) from error
        except URLError as error:
            if isinstance(error.reason, TimeoutError):
                raise TimeoutError("vLLM request exceeded its deadline") from error
            raise ConnectionError(str(error.reason)) from error

    def _stream_request(self, payload, *, timeout):
        """Read OpenAI-compatible SSE with a total deadline and retain partial content."""
        import json
        import uuid
        from urllib.request import Request
        from urllib.error import HTTPError, URLError
        started=monotonic(); text=[]; reasoning=[]; usage={}; request_id=uuid.uuid4().hex
        finish=None; metadata={}; response=None; last_progress=started
        request=Request(self.config.api_base.rstrip('/')+'/chat/completions',
            data=json.dumps(payload,ensure_ascii=False).encode(),headers={'Content-Type':'application/json',
                'Accept':'text/event-stream','X-Request-Id':request_id})
        try:
            response=self._http.open(request,timeout=timeout)
            while True:
                remaining=timeout-(monotonic()-started)
                if remaining<=0: raise TimeoutError('stream generation deadline elapsed')
                # urllib's socket timeout alone is per read, so reset it to the total remainder.
                sock=getattr(getattr(getattr(response,'fp',None),'raw',None),'_sock',None)
                if sock is not None: sock.settimeout(remaining)
                line=response.readline()
                if not line: break
                if not line.startswith(b'data:'): continue
                data=line[5:].strip()
                if data==b'[DONE]': break
                packet=json.loads(data)
                if 'error' in packet: raise ValueError('vLLM stream error: '+str(packet['error']))
                metadata.update({k:packet[k] for k in ('id','model','created','object') if k in packet})
                request_id=packet.get('id',request_id)
                if packet.get('usage'): usage=packet['usage']
                for choice in packet.get('choices',[]):
                    delta=choice.get('delta',{})
                    if delta.get('content'): text.append(delta['content'])
                    part=delta.get('reasoning') or delta.get('reasoning_content')
                    if part: reasoning.append(part)
                    if choice.get('finish_reason'): finish=choice['finish_reason']
                if getattr(self, 'stream_progress', True) and monotonic()-last_progress>=10:
                    print('V2_MODEL_PROGRESS '+json.dumps({'request_id':request_id,
                        'seconds':round(monotonic()-started,1),'received_characters':sum(map(len,text))}),flush=True)
                    last_progress=monotonic()
            if finish is None: raise ConnectionError('vLLM stream ended without a finish_reason')
            return {**metadata,'choices':[{'message':{'role':'assistant','content':''.join(text),
                'reasoning':''.join(reasoning) or None},'finish_reason':finish}], 'usage':usage}
        except HTTPError as exc:
            raise ValueError('vLLM HTTP error: '+exc.read().decode(errors='replace')[:2000]) from exc
        except (TimeoutError,URLError,ConnectionError) as exc:
            if isinstance(exc,URLError) and not isinstance(exc.reason,TimeoutError):
                raise GenerationConnectionError(str(exc.reason),partial_text=''.join(text),
                    partial_reasoning=''.join(reasoning),request_id=request_id) from exc
            if isinstance(exc,ConnectionError):
                raise GenerationConnectionError(str(exc),partial_text=''.join(text),
                    partial_reasoning=''.join(reasoning),request_id=request_id) from exc
            raise GenerationTimeout('vLLM stream exceeded its deadline',partial_text=''.join(text),
                partial_reasoning=''.join(reasoning),request_id=request_id) from exc
        finally:
            if response is not None: response.close()

    def wait_until_idle(self, timeout):
        """Confirm zero active AND queued requests before retry; unknown metrics mean unsafe."""
        import re
        from time import sleep
        from urllib.request import Request
        deadline=monotonic()+timeout
        url=self.config.api_base.rstrip('/').removesuffix('/v1')+'/metrics'
        while monotonic()<deadline:
            try:
                with self._http.open(Request(url),timeout=max(.01,min(2,deadline-monotonic()))) as response:
                    body=response.read().decode()
                metrics={}
                for name in ('num_requests_running','num_requests_waiting'):
                    values=re.findall(r'^vllm:'+name+r'(?:\{[^\n]*\})?\s+([0-9.eE+-]+)',body,re.M)
                    if values: metrics[name]=sum(map(float,values))
                if len(metrics)==2 and all(v==0 for v in metrics.values()): return True
            except Exception:
                return False
            sleep(min(.2,max(0,deadline-monotonic())))
        return False

    def count_tokens(self, messages: list[dict]) -> int:
        """Count actual visual/text input tokens with the downloaded processor on CPU."""
        return LocalModel.count_tokens(self, messages)

    def _prepare(self, messages: list[dict]):
        """Reuse local image validation and multimodal template preparation."""
        return LocalModel._prepare(self, messages)

    def _wire_messages(self, messages: list[dict]) -> list[dict]:
        """Encode host-authorized image files as inline images instead of binary text."""
        import base64
        import mimetypes
        from pathlib import Path

        normalized = deepcopy(messages)
        for message in normalized:
            if not isinstance(message.get("content"), list):
                continue
            for index, item in enumerate(message["content"]):
                if item.get("type") != "image":
                    continue
                source = item.get("image", item.get("url", ""))
                if not isinstance(source, str) or source.startswith(("http:", "https:", "data:")):
                    raise ValueError("only host-authorized local image files are supported")
                path = Path(source)
                if path.suffix.lower() not in (".png", ".jpg", ".jpeg", ".webp", ".bmp"):
                    raise ValueError("unsupported image extension")
                mime = mimetypes.guess_type(path.name)[0] or "image/png"
                content = base64.b64encode(path.read_bytes()).decode("ascii")
                message["content"][index] = {"type": "image_url", "image_url": {
                    "url": f"data:{mime};base64,{content}"}}
        return normalized

    def generate(self, messages: list[dict], *, max_new_tokens: int, remaining_seconds: float,
                 output_schema: dict | None = None, thinking: bool | None = None) -> GenerationResult:
        """Generate normally and store reasoning separately from the parseable final content."""
        import re

        started = monotonic()
        if remaining_seconds <= 0 or not self._lock.acquire(timeout=max(0, remaining_seconds)):
            raise TimeoutError("vLLM generation deadline elapsed")
        try:
            input_count = self.count_tokens(messages)
            if input_count + max_new_tokens > self.config.context_limit:
                raise ValueError("processed input plus output exceeds context_limit")
            payload = {"model": self.config.served_model_name, "messages": self._wire_messages(messages),
                       "max_tokens": max_new_tokens, "temperature": self.config.temperature if self.config.do_sample else 0,
                       "seed": self.seed, "stream": self.config.stream_responses,
                       "chat_template_kwargs": {"enable_thinking": self.config.enable_thinking if thinking is None else thinking,
                                                "preserve_thinking": False},
                       "mm_processor_kwargs": {"max_pixels": self.config.image_max_pixels}}
            if output_schema is not None:
                payload["response_format"] = {"type": "json_schema", "json_schema": {
                    "name": "action_repair", "schema": output_schema}}
            if not self.config.do_sample:
                payload.update(top_p=1.0, top_k=-1)
            remaining = remaining_seconds - (monotonic() - started)
            if remaining <= 0:
                raise TimeoutError("deadline elapsed preparing the multimodal request")
            if self.config.stream_responses:
                payload['stream_options']={'include_usage':True}
                raw=self._stream_request(payload,timeout=remaining)
            else:
                raw = self._request("chat/completions", payload, timeout=remaining)
            if monotonic() - started >= remaining_seconds:
                raise TimeoutError("vLLM response arrived after the deadline")
            choice = raw["choices"][0]
            message = choice["message"]
            text = message.get("content") or ""
            reasoning = message.get("reasoning") or message.get("reasoning_content")
            # Handle only an explicit complete thinking block, never arbitrary surrounding prose.
            block = re.match(r"^\s*<think>(.*?)</think>\s*", text, flags=re.S)
            if block:
                reasoning = reasoning or block.group(1)
                text = text[block.end():]
            usage = raw.get("usage", {})
            return GenerationResult(text=text, reasoning=reasoning, raw_response=raw,
                                    request_parameters={key: value for key, value in payload.items() if key != "messages"},
                                    tokens=TokenUsage(input_tokens=usage.get("prompt_tokens"),
                                                      output_tokens=usage.get("completion_tokens")),
                                    finish_reason="length" if choice.get("finish_reason") == "length" else "stop")
        finally:
            self._lock.release()


def warmup_model(store, model, max_seconds: float = 90) -> list[dict]:
    """Warm text/image kernels within startup time, recording calls outside scored phase counters."""
    if not store.config.model_warmup or store.config.model.backend != "vllm":
        return []
    deadline, report = monotonic() + max_seconds, []
    samples = [("text", [{"role": "user", "content": "Reply OK only."}], store.dataset.challenge_ids[0])]
    for task_id in store.dataset.task_ids:
        images = [asset for asset in store.dataset.list_assets(task_id) if asset.reference.role == "image"]
        if images:
            path = store.dataset.resolve_asset(task_id, images[0].file_id)
            samples.append(("image", [{"role": "user", "content": [
                {"type": "text", "text": "Reply OK only. Do not analyze or solve anything."},
                {"type": "image", "image": str(path)}]}], store.dataset.get_task(task_id).challenge_id))
            break
    for kind, messages, challenge_id in samples:
        remaining = deadline - monotonic()
        if remaining <= 0:
            break
        event = store.append_event(challenge_id, "pending", "started",
                                   arguments={"call_kind": "warmup", "input_kind": kind},
                                   result_summary="Startup warmup; no benchmark question is supplied.")
        store.begin_model_call(event.event_id, {"call_kind": "warmup", "messages": messages,
            "generation": {**store.config.model.model_dump(), "max_new_tokens": 4},
            "remaining_seconds": remaining, "counted_in_phase_budget": False})
        started = monotonic()
        try:
            result = model.generate(messages, max_new_tokens=4, remaining_seconds=remaining, thinking=False)
            store.finish_model_call(event.event_id, result=result, duration_seconds=monotonic() - started)
            report.append({"kind": kind, "status": "ok", "duration_seconds": monotonic() - started})
        except (ValueError, TimeoutError, ConnectionError) as error:
            store.finish_model_call(event.event_id, error=str(error), duration_seconds=monotonic() - started)
            report.append({"kind": kind, "status": "warning", "error": str(error)})
    return report
