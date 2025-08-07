from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp
import jax.random as jr
import matplotlib.pyplot as plt
from flax import struct
from jaxtyping import PRNGKeyArray

from gymnax.environments import environment, spaces


@struct.dataclass
class EnvParams(environment.EnvParams):
    min_piece_size: int = 2
    max_piece_size: int = 4


@struct.dataclass
class EnvState(environment.EnvState):
    grid: jax.Array  # shape (grid_size, grid_size)
    next_piece: jax.Array  # shape (grid_size, grid_size)
    other_pieces: jax.Array  # shape (n_pieces-1, grid_size, grid_size)
    time: int

    @classmethod
    def init(cls, key: PRNGKeyArray,
             grid_size: int, n_pieces: int,
             min_piece_size: int, max_piece_size: int,
             time: int,) -> EnvState:
        puzzle = create_puzzle(
            key,
            grid_size=grid_size,
            n_pieces=n_pieces,
            min_piece_size=min_piece_size,
            max_piece_size=max_piece_size,
        )
        return cls(grid=puzzle[0],
                   next_piece=puzzle[1],
                   other_pieces=puzzle[2:],
                   time=time,
                   )

    def roll_top_left(self) -> EnvState:
        next_piece_rolled = roll_top_left(self.next_piece)
        other_pieces_rolled = jax.vmap(roll_top_left)(self.other_pieces)
        return EnvState(grid=self.grid,
                        next_piece=next_piece_rolled,
                        other_pieces=other_pieces_rolled,
                        time=self.time)


class PuzzlePacking(environment.Environment[EnvState, EnvParams]):
    def __init__(self, grid_size: int = 4, n_pieces: int = 4,
                 min_piece_size: int = 2, max_piece_size: int = 4):
        super().__init__()
        self.grid_size = grid_size
        self.n_pieces = n_pieces
        self.min_piece_size = min_piece_size
        self.max_piece_size = max_piece_size

    @property
    def default_params(self) -> EnvParams:
        return EnvParams()

    def step_env(
        self,
        key: jax.Array,
        state: EnvState,
        action: int | float | jax.Array,
        params: EnvParams,
    ) -> tuple[jax.Array, EnvState, jax.Array, jax.Array, dict[Any, Any]]:
        ...

    def reset_env(self, key: PRNGKeyArray, params: EnvParams):
        state = EnvState.init(
            key,
            grid_size=self.grid_size,
            n_pieces=self.n_pieces,
            min_piece_size=self.min_piece_size,
            max_piece_size=self.max_piece_size,
            time=0,
        ).roll_top_left()
        return self.get_obs(state), state

    def get_obs(self, state: EnvState, params=None, key=None) -> jax.Array:
        obs = jnp.concatenate(
            [state.grid[None, :],
             state.next_piece[None, :],
             state.other_pieces
             ],
            dtype=jnp.float32
        )

        return obs.astype(jnp.float32)

    def is_terminal(self, state: EnvState, params: EnvParams) -> jax.Array:
        ...

    @property
    def name(self) -> str:
        return "PuzzlePacking"

    @property
    def num_actions(self) -> int:
        return self.grid_size * self.grid_size

    def action_space(self, params: EnvParams | None = None) -> spaces.Discrete:
        return spaces.Discrete(self.grid_size * self.grid_size)

    def observation_space(self, params: EnvParams) -> spaces.Box:
        return spaces.Box(
            low=0,
            high=1,
            shape=(self.n_pieces + 1, self.grid_size, self.grid_size),
            dtype=jnp.float32,
        )

    def state_space(self, params: EnvParams) -> spaces.Dict:
        return spaces.Dict(
            {
                "grid": spaces.Box(
                    low=0,
                    high=1,
                    shape=(self.n_pieces + 1,
                           self.grid_size, self.grid_size),
                    dtype=jnp.bool_,
                ),
            }
        )

    def render(self, state: EnvState, _: EnvParams):
        _, axes = plt.subplots(1, 5, figsize=(5, 1))
        for i, ax in enumerate(axes):
            if i == 0:
                ax.imshow(state.grid, cmap='Greys')
            else:
                ax.imshow(self.grid[i], cmap=COLORMAPS[i])
            ax.set_xticklabels([])
            ax.set_yticklabels([])
            ax.set_xticks([])
            ax.set_yticks([])

        return axes


COLORMAPS = [
    'Purples',
    'Blues',
    'Greens',
    'Oranges',
    'Reds',
]

deltas = jnp.array([
    [0, 1],   # up
    [0, -1],  # down
    [-1, 0],  # left
    [1, 0],   # right
])  # shape (4, 2)


def roll_top_left(arr):
    # Roll up until first row is not all zeros
    def cond_row(x):
        return jnp.all(x[0] == 0)

    def body_row(x):
        return jnp.roll(x, shift=-1, axis=0)

    arr = jax.lax.while_loop(cond_row, body_row, arr)

    # Roll left until first column is not all zeros
    def cond_col(x):
        return jnp.all(x[:, 0] == 0)

    def body_col(x):
        return jnp.roll(x, shift=-1, axis=1)

    arr = jax.lax.while_loop(cond_col, body_col, arr)

    return arr


@jax.jit
def sample_coord(key, prob_matrix):
    # Flatten the 4x4 matrix
    probs_flat = prob_matrix.reshape(-1)
    path_grid = jnp.full_like(prob_matrix, 0.0)

    # Sample an index according to the probability distribution
    idx = jr.choice(key, a=probs_flat.shape[0], p=probs_flat)

    # Convert flat index back to (row, col)
    row = idx // prob_matrix.shape[1]
    col = idx % prob_matrix.shape[1]

    prob_matrix = prob_matrix.at[row, col].set(
        False)  # Set the sampled position to 0
    # Mark the sampled position in the path grid
    path_grid = path_grid.at[row, col].set(1.0)

    return jnp.array([row, col]), prob_matrix, path_grid


# @eqx.filter_jit
def create_puzzle(
        key: PRNGKeyArray,
        grid_size: int = 4,
        n_pieces: int = 4,
        min_piece_size: int = 2,
        max_piece_size: int = 5):

    key_walk, key_sizes = jr.split(key)
    init_grid = jnp.ones((grid_size, grid_size), dtype=bool)
    piece_sizes = jr.randint(key_sizes, shape=(
        n_pieces,), minval=min_piece_size, maxval=max_piece_size+1)

    def place_piece(carry, piece_size):
        key, visited_grid = carry
        new_key, _ = jr.split(key)

        path_init = jnp.full((max_piece_size, 2), -1)
        start_pos, visited_grid, path_grid = sample_coord(key, visited_grid)

        path_init = path_init.at[0].set(start_pos)

        def step_fn(carry, _):
            pos, visited, path, path_grid, step, done, key, piece_size = carry
            key, subkey = jr.split(key)

            too_long = step >= piece_size - 1

            def early_exit():
                return (pos, visited, path, path_grid, step, True,
                        key, piece_size), None

            def do_step():
                candidates = pos + deltas  # (4, 2)
                in_bounds = jnp.all((candidates >= 0) & (
                    candidates < grid_size), axis=1)
                cx, cy = candidates.T
                not_visited = visited[cx, cy]
                valid_mask = in_bounds & not_visited

                # If no valid moves, done=True
                any_valid = jnp.any(valid_mask)

                def no_valid():
                    return (pos, visited, path, path_grid, step, True,
                            key, piece_size), None

                def has_valid():
                    # Assign large negative logits to invalid moves so
                    # they won't be sampled
                    logits = jnp.where(valid_mask, 0.0, -1e9)
                    move_idx = jr.categorical(key, logits)
                    new_pos = candidates[move_idx]
                    visited_updated = visited.at[new_pos[0], new_pos[1]].set(
                        False)
                    path_grid_updated = path_grid.at[new_pos[0],
                                                     new_pos[1]].set(
                        1.0)
                    path_updated = path.at[step + 1].set(new_pos)
                    return (new_pos, visited_updated, path_updated,
                            path_grid_updated, step + 1, False, key,
                            piece_size), None

                return jax.lax.cond(any_valid, has_valid, no_valid)

            return jax.lax.cond(done | too_long, early_exit, do_step)

        init_state = (start_pos, visited_grid, path_init,
                      path_grid, 0, False, key, piece_size)

        final_state, _ = jax.lax.scan(
            step_fn,
            init=init_state,
            xs=None,
            length=max_piece_size - 1)
        _, final_grid, _, new_path_grid, _, _, _, _ = final_state

        return (new_key, final_grid), new_path_grid
    init_carry = (key_walk, init_grid)
    (_, final_state), pieces = jax.lax.scan(
        place_piece,
        init=init_carry,
        xs=piece_sizes,
    )

    return jnp.concat((final_state[None, :], pieces), axis=0)
