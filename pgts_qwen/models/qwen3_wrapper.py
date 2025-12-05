"""
Qwen3 wrapper for reasoning step generation.
"""
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import Optional, List, Dict, Tuple
import logging

logger = logging.getLogger(__name__)


class Qwen3ReasoningGenerator:
    """
    Wrapper for Qwen3 to generate reasoning steps.

    Configuration:
        - Model: Qwen/Qwen3-8B or Qwen3-14B
        - Temperature: 0.7 (for diverse generation)
        - Max tokens per step: 128
        - Stop tokens: ['\\n\\n', 'Therefore', '####']

    Methods:
        - generate_step(): Generate single reasoning step
        - generate_branch(): Generate alternative step
        - extract_hidden_states(): Get last layer hidden state
        - format_prompt(): Create step generation prompt
    """

    def __init__(
        self,
        model_name: str = "Qwen/Qwen3-8B",
        device: str = "cuda",
        torch_dtype: torch.dtype = torch.bfloat16,
        max_new_tokens: int = 128,
        temperature: float = 0.7,
        top_p: float = 0.95,
        use_vllm: bool = False
    ):
        """
        Initialize Qwen3 model.

        Args:
            model_name: HuggingFace model identifier
            device: Device to load model on
            torch_dtype: Data type for model weights
            max_new_tokens: Maximum tokens per generation
            temperature: Sampling temperature
            top_p: Nucleus sampling parameter
            use_vllm: Whether to use vLLM for inference
        """
        self.model_name = model_name
        self.device = device
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.use_vllm = use_vllm

        logger.info(f"Loading Qwen3 model: {model_name}")

        if use_vllm:
            # Use vLLM for faster inference
            try:
                from vllm import LLM, SamplingParams
                self.llm = LLM(
                    model=model_name,
                    dtype=str(torch_dtype).split('.')[-1],
                    tensor_parallel_size=torch.cuda.device_count()
                )
                self.sampling_params = SamplingParams(
                    temperature=temperature,
                    top_p=top_p,
                    max_tokens=max_new_tokens,
                    stop=['\n\n', 'Therefore', '####']
                )
                self.tokenizer = AutoTokenizer.from_pretrained(model_name)
                logger.info("Using vLLM for inference")
            except ImportError:
                logger.warning("vLLM not available, falling back to HuggingFace")
                self.use_vllm = False

        if not use_vllm:
            # Standard HuggingFace loading
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch_dtype,
                device_map="auto"
            )
            self.model.eval()
            logger.info(f"Model loaded on {device}")

        # Stop tokens
        self.stop_tokens = ['\n\n', 'Therefore', '####']

    def format_prompt(
        self,
        problem: str,
        reasoning_chain: List[str],
        action: str = "EXPAND"
    ) -> str:
        """
        Create prompt for step generation.

        Args:
            problem: Original math problem
            reasoning_chain: Previous reasoning steps
            action: EXPAND or BRANCH

        Returns:
            Formatted prompt string
        """
        if action == "EXPAND":
            if len(reasoning_chain) == 0:
                prompt = f"""Problem: {problem}

Let's solve this step by step:

Step 1:"""
            else:
                steps_text = "\n".join(
                    f"Step {i+1}: {step}"
                    for i, step in enumerate(reasoning_chain)
                )
                prompt = f"""Problem: {problem}

{steps_text}

Step {len(reasoning_chain) + 1}:"""

        elif action == "BRANCH":
            if len(reasoning_chain) == 0:
                prompt = f"""Problem: {problem}

Let's try a different approach:

Step 1:"""
            else:
                # For branching, we show previous steps but suggest alternative
                steps_text = "\n".join(
                    f"Step {i+1}: {step}"
                    for i, step in enumerate(reasoning_chain[:-1])
                )
                prompt = f"""Problem: {problem}

{steps_text}

Alternative approach for Step {len(reasoning_chain)}:"""
        else:
            raise ValueError(f"Unknown action: {action}")

        return prompt

    def generate_step(
        self,
        problem: str,
        reasoning_chain: List[str],
        return_hidden_states: bool = True
    ) -> Tuple[str, Optional[torch.Tensor]]:
        """
        Generate next reasoning step.

        Args:
            problem: Math problem text
            reasoning_chain: Previous reasoning steps
            return_hidden_states: Whether to return hidden states

        Returns:
            Tuple of (generated_text, hidden_state)
        """
        prompt = self.format_prompt(problem, reasoning_chain, action="EXPAND")
        return self._generate(prompt, return_hidden_states)

    def generate_branch(
        self,
        problem: str,
        reasoning_chain: List[str],
        return_hidden_states: bool = True
    ) -> Tuple[str, Optional[torch.Tensor]]:
        """
        Generate alternative reasoning step.

        Args:
            problem: Math problem text
            reasoning_chain: Previous reasoning steps
            return_hidden_states: Whether to return hidden states

        Returns:
            Tuple of (generated_text, hidden_state)
        """
        prompt = self.format_prompt(problem, reasoning_chain, action="BRANCH")
        return self._generate(prompt, return_hidden_states)

    def _generate(
        self,
        prompt: str,
        return_hidden_states: bool = True
    ) -> Tuple[str, Optional[torch.Tensor]]:
        """
        Internal generation method.

        Args:
            prompt: Formatted prompt
            return_hidden_states: Whether to return hidden states

        Returns:
            Tuple of (generated_text, hidden_state)
        """
        if self.use_vllm:
            # vLLM generation (no hidden states)
            outputs = self.llm.generate([prompt], self.sampling_params)
            generated_text = outputs[0].outputs[0].text.strip()
            return generated_text, None

        # HuggingFace generation
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                temperature=self.temperature,
                top_p=self.top_p,
                do_sample=True,
                output_hidden_states=return_hidden_states,
                return_dict_in_generate=True,
                pad_token_id=self.tokenizer.pad_token_id or self.tokenizer.eos_token_id
            )

        # Decode generated text
        generated_ids = outputs.sequences[0][inputs.input_ids.shape[1]:]
        generated_text = self.tokenizer.decode(generated_ids, skip_special_tokens=True)

        # Clean up text (stop at stop tokens)
        for stop_token in self.stop_tokens:
            if stop_token in generated_text:
                generated_text = generated_text.split(stop_token)[0]

        generated_text = generated_text.strip()

        # Extract hidden states with robust error handling
        hidden_state = None
        if return_hidden_states:
            hidden_state = self._extract_hidden_state_robust(outputs)

            # Fallback: if extraction failed, use forward pass method
            if hidden_state is None:
                logger.debug("Failed to extract from generation outputs, using fallback")
                try:
                    # Get full text for forward pass
                    full_text = prompt + " " + generated_text
                    hidden_state = self.extract_hidden_states(full_text)
                except Exception as e:
                    logger.warning(f"Fallback hidden state extraction also failed: {e}")

        return generated_text, hidden_state

    def _extract_hidden_state_robust(self, outputs) -> Optional[torch.Tensor]:
        """
        Robustly extract hidden state from generation outputs.

        Args:
            outputs: GenerateDecoderOnlyOutput from model.generate()

        Returns:
            Hidden state tensor or None if extraction fails
        """
        try:
            if not hasattr(outputs, 'hidden_states') or outputs.hidden_states is None:
                return None

            if len(outputs.hidden_states) == 0:
                return None

            # hidden_states structure: tuple of (tuple of layer tensors) for each generation step
            last_step_hidden = outputs.hidden_states[-1]  # Last generated token

            if not isinstance(last_step_hidden, tuple) or len(last_step_hidden) == 0:
                return None

            last_layer = last_step_hidden[-1]  # Last layer (e.g., layer 32 for Qwen3)

            # Validate tensor shape
            if not isinstance(last_layer, torch.Tensor):
                return None

            if last_layer.dim() != 3:  # Should be [batch, seq_len, hidden_dim]
                logger.warning(f"Unexpected hidden state shape: {last_layer.shape}")
                return None

            # Extract: first batch, last position
            hidden_state = last_layer[0, -1, :]

            # Validate dimension
            expected_dim = self.get_hidden_dim()
            if hidden_state.shape[0] != expected_dim:
                logger.warning(
                    f"Hidden state dim mismatch: got {hidden_state.shape[0]}, "
                    f"expected {expected_dim}"
                )
                return None

            return hidden_state

        except (IndexError, AttributeError, TypeError) as e:
            logger.debug(f"Hidden state extraction error: {e}")
            return None

    def extract_hidden_states(
        self,
        text: str
    ) -> torch.Tensor:
        """
        Extract hidden states for given text.

        Args:
            text: Input text

        Returns:
            Hidden state tensor from last layer, last token
        """
        if self.use_vllm:
            raise NotImplementedError("Hidden state extraction not supported with vLLM")

        inputs = self.tokenizer(text, return_tensors="pt").to(self.device)

        with torch.no_grad():
            outputs = self.model(
                **inputs,
                output_hidden_states=True
            )

        # Get last layer, last token
        hidden_state = outputs.hidden_states[-1][0, -1, :]
        return hidden_state

    def batch_generate(
        self,
        prompts: List[str],
        return_hidden_states: bool = False
    ) -> List[Tuple[str, Optional[torch.Tensor]]]:
        """
        Generate for multiple prompts in batch.

        Args:
            prompts: List of prompts
            return_hidden_states: Whether to return hidden states

        Returns:
            List of (generated_text, hidden_state) tuples
        """
        if self.use_vllm:
            outputs = self.llm.generate(prompts, self.sampling_params)
            return [(output.outputs[0].text.strip(), None) for output in outputs]

        # For HuggingFace, generate sequentially for now
        # Could be optimized with proper batching
        results = []
        for prompt in prompts:
            result = self._generate(prompt, return_hidden_states)
            results.append(result)

        return results

    def get_hidden_dim(self) -> int:
        """Get the hidden dimension of the model."""
        if self.use_vllm:
            # Load config to get hidden dim
            from transformers import AutoConfig
            config = AutoConfig.from_pretrained(self.model_name)
            return config.hidden_size
        return self.model.config.hidden_size
