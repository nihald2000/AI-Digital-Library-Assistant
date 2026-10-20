from mistralai import Mistral
import logging
import asyncio
from typing import List, Dict, Any, Optional

import openai
import config
from services import llm_providers

logger = logging.getLogger(__name__)

class LLMService:
    def __init__(self):
        self.config = config.config

        self.nebius_client = None
        self.mistral_client = None
        self.openai_client = None

        # legs whose key/config failed PERMANENTLY this session (quota/auth)
        self._disabled: Dict[str, str] = {}

        self._initialize_clients()

    def _initialize_clients(self):
        """Initialize legacy SDK clients (chain legs build their own lazily)."""
        try:
            if self.config.OPENAI_API_KEY:
                self.openai_client = openai.OpenAI(
                    api_key=self.config.OPENAI_API_KEY
                )
                logger.info("OpenAI client initialized")

            if self.config.NEBIUS_API_KEY:
                self.nebius_client = openai.OpenAI(
                    api_key=self.config.NEBIUS_API_KEY,
                    base_url=self.config.NEBIUS_BASE_URL
                )
                logger.info("NEBIUS client initialized")

            if self.config.MISTRAL_API_KEY:
                self.mistral_client = Mistral( # Standard sync client
                    api_key=self.config.MISTRAL_API_KEY
                )
                logger.info("Mistral client initialized")

            chain = llm_providers.resolve_llm_providers(self.config)
            if chain:
                logger.info("LLM provider chain (%d legs): %s",
                            len(chain), " -> ".join(l["name"] for l in chain))
            else:
                logger.warning("No LLM provider chain legs resolved. Check API keys (GEMINI_API_KEY / GROQ_API_KEY / OPENROUTER_API_KEY / NEBIUS_API_KEY).")

        except Exception as e:
            logger.error(f"Error initializing LLM clients: {str(e)}")
            raise

    # ------------------------------------------------------------ chain core
    def _chain_legs(self) -> List[Dict[str, str]]:
        """Usable chain legs (permanently-failed legs filtered out)."""
        return [
            leg for leg in llm_providers.resolve_llm_providers(self.config)
            if leg["name"] not in self._disabled
        ]

    def _client_for(self, leg: Dict[str, str]):
        """Lazily built OpenAI-compat client per leg (max_retries=1: failover beats long retry loops)."""
        cache = getattr(self, "_compat_clients", None)
        if cache is None:
            cache = self._compat_clients = {}
        key = (leg["name"], leg["base_url"], leg["api_key"])
        if key not in cache:
            cache[key] = openai.OpenAI(
                api_key=leg["api_key"],
                base_url=leg["base_url"],
                max_retries=1,
            )
        return cache[key]

    async def _generate_compat(self, leg: Dict[str, str], prompt: str,
                               model_name: str, max_tokens: int, temperature: float) -> str:
        """One chat-completions call against an OpenAI-compatible leg."""
        client = self._client_for(leg)
        logger.debug("Generating via %s model=%s max_tokens=%s temp=%s",
                     leg["name"], model_name, max_tokens, temperature)
        loop = asyncio.get_event_loop()
        response = await loop.run_in_executor(
            None,
            lambda: client.chat.completions.create(
                model=model_name,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=max_tokens,
                temperature=temperature,
            )
        )
        if response.choices and response.choices[0].message and response.choices[0].message.content:
            return response.choices[0].message.content.strip()
        return ""

    def _disable_leg(self, name: str, exc: Exception) -> None:
        if name not in self._disabled:
            self._disabled[name] = str(exc)[:200]
            logger.warning("LLM leg '%s' disabled for this session: %s", name, self._disabled[name])

    async def _generate_on_leg(self, leg: Dict[str, str], prompt: str,
                               model_name: str, max_tokens: int, temperature: float) -> str:
        if leg["kind"] == "sdk-mistral":
            return await self._generate_with_mistral(prompt, model_name, max_tokens, temperature)
        return await self._generate_compat(leg, prompt, model_name, max_tokens, temperature)

    async def generate_text(self, prompt: str, model: str = "auto", max_tokens: int = 1000, temperature: float = 0.7) -> str:
        """Generate text through the provider chain.

        model='auto'   -> first healthy leg's primary model
        model='fast'   -> first healthy leg's fast model
        explicit model -> routed to the owning leg ('groq/...', 'gemini/...',
                          'openrouter/...', 'nebius/...', 'gpt-*', 'mistral*')

        A leg that fails with a PERMANENT error (401/402/403/429, quota text)
        is dropped for the session and the next leg serves immediately;
        transient errors also fall through to the next leg, and the last
        exception re-raises if every leg fails.
        """
        errors: List[str] = []

        # ---- explicit model routing
        if model not in ("auto", "fast"):
            leg_name = llm_providers.leg_for_model(model)
            if leg_name:
                for leg in self._chain_legs():
                    if leg["name"] != leg_name:
                        continue
                    stripped = model
                    for prefix in llm_providers.PROVIDER_PREFIXES.get(leg_name, ()):
                        if stripped.lower().startswith(prefix):
                            stripped = stripped[len(prefix):]
                            break
                    stripped = stripped or leg["model"]
                    try:
                        return await self._generate_on_leg(leg, prompt, stripped, max_tokens, temperature)
                    except Exception as exc:
                        errors.append(f"{leg['name']}: {exc}")
                        if llm_providers.permanent_llm_error(exc):
                            self._disable_leg(leg["name"], exc)
                        break  # explicit leg owns this model; then fail over below
                # fall through to chain for resilience if the owned leg failed
            else:
                return await self._generate_with_fallback_chain(
                    prompt, self.config.OPENAI_MODEL, max_tokens, temperature, errors
                )

        # ---- chain walk
        legs = self._chain_legs()
        if not legs:
            raise ValueError(
                "No LLM provider available. Add a free key (GEMINI_API_KEY, "
                f"GROQ_API_KEY or OPENROUTER_API_KEY). Attempts: {errors}"
            )
        key = "fast_model" if model == "fast" else "model"
        for leg in legs:
            model_name = leg.get(key) or leg["model"]
            try:
                return await self._generate_on_leg(leg, prompt, model_name, max_tokens, temperature)
            except Exception as exc:
                errors.append(f"{leg['name']}: {exc}")
                logger.warning("LLM leg '%s' failed (%s) - trying next provider",
                               leg["name"], str(exc)[:120])
                if llm_providers.permanent_llm_error(exc):
                    self._disable_leg(leg["name"], exc)
        raise ValueError(f"All LLM providers failed. {'; '.join(e[:150] for e in errors)}")

    async def _generate_with_fallback_chain(self, prompt: str, fallback_model: str,
                                            max_tokens: int, temperature: float,
                                            errors: List[str]) -> str:
        """Legacy single-model choice routed through the chain (explicit unknown model)."""
        for leg in self._chain_legs():
            try:
                return await self._generate_on_leg(leg, prompt, fallback_model, max_tokens, temperature)
            except Exception as exc:
                errors.append(f"{leg['name']}: {exc}")
                if llm_providers.permanent_llm_error(exc):
                    self._disable_leg(leg["name"], exc)
        raise ValueError(f"All LLM providers failed. {'; '.join(e[:150] for e in errors)}")

    async def _generate_with_openai(self, prompt: str, model_name: str, max_tokens: int, temperature: float) -> str:
        """Generate text using OpenAI"""
        if not self.openai_client:
            raise RuntimeError("OpenAI client not initialized.")
        try:
            logger.debug(f"Generating with OpenAI model: {model_name}, max_tokens: {max_tokens}, temp: {temperature}")
            loop = asyncio.get_event_loop()
            
            # Determine correct token parameter based on model family
            # GPT-5, o1, o3 series use max_completion_tokens
            use_completion_tokens = any(x in model_name for x in ["gpt-5", "o1-", "o3-"])
            
            kwargs = {
                "model": model_name,
                "messages": [{"role": "user", "content": prompt}],
            }
            
            if use_completion_tokens:
                kwargs["max_completion_tokens"] = max_tokens
                # Reasoning models enforce temperature=1
                kwargs["temperature"] = 1
                if temperature != 1:
                    logger.warning(f"Temperature {temperature} ignored for model {model_name} (requires 1).")
            else:
                kwargs["max_tokens"] = max_tokens
                kwargs["temperature"] = temperature

            response = await loop.run_in_executor(
                None,
                lambda: self.openai_client.chat.completions.create(**kwargs)
            )
            if response.choices and response.choices[0].message:
                 content = response.choices[0].message.content
                 if content is not None:
                     return content.strip()
            return ""
        except Exception as e:
            logger.error(f"Error with OpenAI generation (model: {model_name}): {str(e)}")
            raise

    async def _generate_with_nebius(self, prompt: str, model_name: str, max_tokens: int, temperature: float) -> str:
        """Generate text using NEBIUS (OpenAI OSS models via sync client)"""
        if not self.nebius_client:
            raise RuntimeError("NEBIUS client not initialized.")
        try:
            logger.debug(f"Generating with NEBIUS model: {model_name}, max_tokens: {max_tokens}, temp: {temperature}, prompt: '{prompt[:50]}...'")
            loop = asyncio.get_event_loop()
            
            response = await loop.run_in_executor(
                None,
                lambda: self.nebius_client.chat.completions.create(
                    model=model_name,
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=max_tokens,
                    temperature=temperature
                )
            )
            if response.choices and response.choices[0].message:
                 content = response.choices[0].message.content
                 if content is not None:
                     return content.strip()
                 else:
                     logger.warning(f"NEBIUS response message content is None for model {model_name}.")
                     return ""
            else:
                logger.warning(f"NEBIUS response did not contain expected choices or message for model {model_name}.")
                return ""
        except Exception as e:
            logger.error(f"Error with NEBIUS generation (model: {model_name}): {str(e)}")
            raise

    async def _generate_with_mistral(self, prompt: str, model_name: str, max_tokens: int, temperature: float) -> str:
        """Generate text using Mistral (Sync via run_in_executor)"""
        if not self.mistral_client:
            raise RuntimeError("Mistral client not initialized.")
        try:
            logger.debug(f"Generating with Mistral model: {model_name}, temp: {temperature}, prompt: '{prompt[:50]}...' (max_tokens: {max_tokens} - note: not directly used by MistralClient.chat)")
            loop = asyncio.get_event_loop()
            
            response = await loop.run_in_executor(
                None,
                lambda: self.mistral_client.chat.complete(
                    model=model_name, 
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=max_tokens, 
                    temperature=temperature
                )
            )
            if response.choices and response.choices[0].message:
                content = response.choices[0].message.content
                if content is not None:
                    return content.strip()
                else:
                    logger.warning(f"Mistral response message content is None for model {model_name}.")
                    return ""
            else:
                logger.warning(f"Mistral response did not contain expected choices or message for model {model_name}.")
                return ""
        except Exception as e:
            logger.error(f"Error with Mistral generation (model: {model_name}): {str(e)}")
            raise
    

    async def summarize(self, text: str, style: str = "concise", max_length: Optional[int] = None) -> str:
        if not text.strip():
            return ""
        
        style_prompts = {
            "concise": "Provide a concise summary of the following text, focusing on the main points:",
            "detailed": "Provide a detailed summary of the following text, including key details and supporting information:",
            "bullet_points": "Summarize the following text as a list of bullet points highlighting the main ideas:",
            "executive": "Provide an executive summary of the following text, focusing on key findings and actionable insights:"
        }
        prompt_template = style_prompts.get(style, style_prompts["concise"])
        if max_length:
            prompt_template += f" Keep the summary under approximately {max_length} words."
        
        prompt = f"{prompt_template}\n\nText to summarize:\n{text}\n\nSummary:"
        
        try:
            summary_max_tokens = (max_length * 2) if max_length else 500 
            summary = await self.generate_text(prompt, model="auto", max_tokens=summary_max_tokens, temperature=0.3)
            return summary.strip()
        except Exception as e:
            logger.error(f"Error generating summary: {str(e)}")
            return "Error generating summary"
    
    async def generate_tags(self, text: str, max_tags: int = 5) -> List[str]:
        if not text.strip():
            return []
        
        prompt = f"""Generate up to {max_tags} relevant tags for the following text.
        Tags should be concise, descriptive keywords or phrases (1-3 words typically) that capture the main topics or themes.
        Return only the tags, separated by commas. Do not include any preamble or explanation.

        Text:
        {text}

        Tags:"""
        
        try:
            # Use FAST_MODEL for tags
            response = await self.generate_text(prompt, model="fast", max_tokens=100, temperature=0.5)
            tags = [tag.strip().lower() for tag in response.split(',') if tag.strip()]
            tags = [tag for tag in tags if tag and len(tag) > 1 and len(tag) < 50]
            return list(dict.fromkeys(tags))[:max_tags]
        except Exception as e:
            logger.error(f"Error generating tags: {str(e)}")
            return []
    
    async def categorize(self, text: str, categories: List[str]) -> str:
        if not text.strip() or not categories:
            return "Uncategorized"
        
        categories_str = ", ".join([f"'{cat}'" for cat in categories])
        prompt = f"""Classify the following text into ONE of these categories: {categories_str}.
        Choose the single most appropriate category based on the content and main theme of the text.
        Return only the category name as a string, exactly as it appears in the list provided. Do not add any other text or explanation.

        Text to classify:
        {text}

        Category:"""
        
        try:
            # Use FAST_MODEL for categorization
            response = await self.generate_text(prompt, model="fast", max_tokens=50, temperature=0.1) 
            category_candidate = response.strip().strip("'\"")
            
            for cat in categories:
                if cat.lower() == category_candidate.lower():
                    return cat
            
            logger.warning(f"LLM returned category '{category_candidate}' which is not in the provided list: {categories}. Falling back.")
            return categories[0] if categories else "Uncategorized"
        except Exception as e:
            logger.error(f"Error categorizing text: {str(e)}")
            return "Uncategorized"
    
    async def answer_question(self, question: str, context: str, max_context_length: int = 3000) -> str:
        if not question.strip():
            return "No question provided."
        if not context.strip():
            return "I don't have enough context to answer this question. Please provide relevant information."
        
        if len(context) > max_context_length:
            context = context[:max_context_length] + "..."
            logger.warning(f"Context truncated to {max_context_length} characters for question answering.")
        
        prompt = f"""You are an expert Q&A assistant. Your task is to synthesize an answer to the user's question based *only* on the provided source documents.
Analyze all the source documents provided in the context below.
If the information is present, provide a comprehensive answer.

Here are the source documents:
--- START OF CONTEXT ---
{context}
--- END OF CONTEXT ---

Based on the context above, please provide a clear and concise answer to the following question.

Question: {question}

Answer:"""
        
        try:
            answer = await self.generate_text(prompt, model="auto", max_tokens=800, temperature=0.5)
            return answer.strip()
        except Exception as e:
            logger.error(f"Error answering question: {str(e)}")
            return "I encountered an error while trying to answer your question."
    
    async def extract_key_information(self, text: str) -> Dict[str, Any]:
        if not text.strip():
            return {}
        
        prompt = f"""Analyze the following text and extract key information.
        Provide the response as a JSON object with the following keys:
        - "main_topic": (string) The main topic or subject of the text.
        - "key_points": (array of strings) A list of 3-5 key points or takeaways.
        - "entities": (array of strings) Important people, places, organizations, or products mentioned.
        - "sentiment": (string) Overall sentiment of the text (e.g., "positive", "neutral", "negative", "mixed").
        - "content_type": (string) The perceived type of content (e.g., "article", "email", "report", "conversation", "advertisement", "other").

        If a piece of information is not found or not applicable, use null or an empty array/string as appropriate for the JSON structure.

        Text to analyze:
        ---
        {text}
        ---

        JSON Analysis:"""
        
        try:
            response_str = await self.generate_text(prompt, model="auto", max_tokens=500, temperature=0.4)
            
            import json
            try:
                if response_str.startswith("```json"):
                    response_str = response_str.lstrip("```json").rstrip("```").strip()
                
                info = json.loads(response_str)
                expected_keys = {"main_topic", "key_points", "entities", "sentiment", "content_type"}
                if not expected_keys.issubset(info.keys()):
                    logger.warning(f"Extracted information missing some expected keys. Got: {info.keys()}")
                return info
            except json.JSONDecodeError as je:
                logger.error(f"Failed to parse JSON from LLM response for key_information: {je}")
                logger.debug(f"LLM Response string was: {response_str}")
                info_fallback = {}
                lines = response_str.split('\n')
                for line in lines:
                    if ':' in line:
                        key, value = line.split(':', 1)
                        key_clean = key.strip().lower().replace(' ', '_')
                        value_clean = value.strip()
                        if value_clean:
                            if key_clean in ["key_points", "entities"] and '[' in value_clean and ']' in value_clean:
                                try:
                                    info_fallback[key_clean] = [item.strip().strip("'\"") for item in value_clean.strip('[]').split(',') if item.strip()]
                                except: info_fallback[key_clean] = value_clean
                            else: info_fallback[key_clean] = value_clean
                if info_fallback:
                    logger.info("Successfully parsed key information using fallback line-based method.")
                    return info_fallback
                return {"error": "Failed to parse LLM output", "raw_response": response_str}
        except Exception as e:
            logger.error(f"Error extracting key information: {str(e)}")
            return {"error": f"General error extracting key information: {str(e)}"}

    async def check_availability(self) -> Dict[str, bool]:
        """Check each chain leg with a tiny test call."""
        availability: Dict[str, bool] = {}
        test_prompt = "Hello"
        test_max_tokens = 5
        test_temp = 0.1

        logger.info("Checking LLM chain availability...")
        for leg in llm_providers.resolve_llm_providers(self.config):
            name = leg["name"]
            if name in self._disabled:
                availability[name] = False
                continue
            try:
                logger.debug(f"Testing '{name}' with model {leg['model']}...")
                test_response = await self._generate_on_leg(
                    leg, test_prompt, leg["model"], test_max_tokens, test_temp
                )
                availability[name] = bool(test_response.strip())
            except Exception as e:
                availability[name] = False
                logger.warning(f"LLM leg '{name}' availability check failed: {e}")
            logger.info(f"LLM leg '{name}' available: {availability[name]}")

        logger.info(f"Final LLM Availability: {availability}")
        return availability
