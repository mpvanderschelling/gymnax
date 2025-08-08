"""Simplified PyTorch-compatible wrapper for batched gymnax environments."""

from typing import Any, Dict, Tuple
import jax
import jax.numpy as jnp
import torch
import numpy as np
import gymnax
import os


class BatchedTrainerCompatibleWrapper:
    """Minimal batched environment wrapper for PyTorch trainers."""
    
    def __init__(self, 
                 batch_size: int,
                 env_name: str = "PuzzlePacking",
                 device: str = "cpu",
                 env_kwargs: Dict[str, Any] | None = None,
                 env_params: Dict[str, Any] | None = None):
        """Initialize batched environment wrapper.
        
        Args:
            batch_size: Number of parallel environments
            env_name: Name of the gymnax environment (must be PuzzlePacking)
            device: PyTorch device
            env_kwargs: Environment creation kwargs:
                - grid_size: Size of puzzle grid (default: 4)
                - n_pieces: Number of puzzle pieces (default: 4)  
                - min_piece_size: Min piece size (default: 2)
                - max_piece_size: Max piece size (default: 4)
            env_params: Environment runtime parameters:
                - penalty_factor: Penalty calculation factor (default: 0.0)
        """
        self.batch_size = batch_size
        self.device = device
        
        # Configure JAX for GPU if requested and available
        if device != "cpu" and jax.default_backend() == "gpu":
            # JAX will automatically use GPU if available
            pass
        elif device != "cpu":
            # Warn if GPU requested but not available
            if not jax.default_backend() == "gpu":
                print(f"Warning: GPU requested but JAX backend is {jax.default_backend()}. Using CPU.")
        
        # Initialize JAX environment
        if env_kwargs is None:
            env_kwargs = {}
        if env_params is None:
            env_params = {}
            
        self.env, self.env_params = gymnax.make(env_name, **env_kwargs)
        self.env_params = self.env_params.replace(**env_params)
        
        # JIT compile batch functions for efficiency
        self._batch_reset = jax.jit(jax.vmap(self.env.reset, in_axes=(0, None)))
        self._batch_step = jax.jit(jax.vmap(self.env.step, in_axes=(0, 0, 0, None)))
        
        # Internal state tracking
        self._current_states = None
        self._keys = None
        
    def _jax_to_torch(self, jax_array: jax.Array) -> torch.Tensor:
        """Convert JAX array to PyTorch tensor."""
        # Use device_get to efficiently transfer from JAX device to host
        numpy_array = np.array(jax.device_get(jax_array))
        return torch.from_numpy(numpy_array).to(self.device)
        
    def _torch_to_jax(self, torch_tensor: torch.Tensor) -> jax.Array:
        """Convert PyTorch tensor to JAX array."""
        if torch_tensor.is_cuda:
            torch_tensor = torch_tensor.cpu()
        return jnp.array(torch_tensor.detach().numpy())
        
    def reset(self, seed = None) -> Tuple[torch.Tensor, Any]:
        """Reset all environments and return initial observations.
        
        Args:
            seed: JAX key or integer seed for reproducible resets
            
        Returns:
            observations: (batch_size, n_pieces + 1, board_size, board_size)
            states: Internal JAX states (kept for efficiency)
        """
        # Use JAX key directly or create one from integer
        if seed is None:
            key = jax.random.PRNGKey(0)
        else:
            # Assume seed is already a JAX key, or convert if it's an integer
            try:
                # Try to use as JAX key directly
                key = seed
                # Test if it's a valid key by trying to split it
                jax.random.split(key, 1)
            except:
                # If that fails, assume it's an integer and convert
                key = jax.random.PRNGKey(int(seed))
            
        # Split key for batch of environments
        self._keys = jax.random.split(key, self.batch_size)
        
        # Reset all environments in parallel
        observations, self._current_states = self._batch_reset(self._keys, self.env_params)
        
        # Convert to PyTorch
        torch_obs = self._jax_to_torch(observations)
        
        return torch_obs, self._current_states
        
    def step(self, states: Any, actions: torch.Tensor) -> Tuple[torch.Tensor, Any, torch.Tensor, torch.Tensor, Dict]:
        """Step all environments with batched actions.
        
        Args:
            states: Current environment states from previous step/reset
            actions: (batch_size,) tensor of actions
            
        Returns:
            observations: (batch_size, n_pieces + 1, board_size, board_size)
            next_states: Updated environment states  
            rewards: (batch_size,) tensor
            dones: (batch_size,) boolean tensor
            infos: Dictionary of additional information
        """
        # Convert actions to JAX and ensure proper shape
        if isinstance(actions, torch.Tensor):
            if actions.dim() == 2 and actions.shape[1] == 1:
                actions = actions.squeeze(1)
            jax_actions = self._torch_to_jax(actions)
        else:
            jax_actions = jnp.array(actions)
        
        # Split keys for this step (JAX environments need fresh randomness)
        self._keys = jax.vmap(lambda k: jax.random.split(k)[0])(self._keys)
        step_keys = jax.vmap(lambda k: jax.random.split(k)[1])(self._keys)
        
        # Step all environments in parallel
        observations, next_states, rewards, dones, infos = self._batch_step(
            step_keys, states, jax_actions, self.env_params
        )
        
        # Convert outputs to PyTorch
        torch_obs = self._jax_to_torch(observations)
        torch_rewards = self._jax_to_torch(rewards)
        torch_dones = self._jax_to_torch(dones).bool()
        
        return torch_obs, next_states, torch_rewards, torch_dones, infos
    
    @property 
    def observation_shape(self) -> Tuple[int, ...]:
        """Get observation shape for single environment."""
        obs, _ = self.reset()
        return obs.shape[1:]  # Remove batch dimension
    
    @property
    def num_actions(self) -> int:
        """Get number of actions."""
        return self.env.num_actions
        
    @property
    def max_steps(self) -> int:
        """Get max steps per episode."""
        if hasattr(self.env, 'n_pieces'):
            return self.env.n_pieces
        return self.env_params.max_steps_in_episode