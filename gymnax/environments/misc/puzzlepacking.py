from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp
import jax.random as jr
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
from flax import struct
from jaxtyping import PRNGKeyArray
from matplotlib import gridspec

from gymnax.environments import environment, spaces


@struct.dataclass
class EnvParams(environment.EnvParams):
    penalty_factor: float = 1.0


@struct.dataclass
class EnvState(environment.EnvState):
    grid: jax.Array  # shape (grid_size, grid_size)
    next_piece: jax.Array  # shape (grid_size, grid_size)
    other_pieces: jax.Array  # shape (n_pieces-1, grid_size, grid_size)
    initial_free_space: int
    time: int

    @classmethod
    def init(cls, key: PRNGKeyArray,
             grid_size: int, n_pieces: int,
             min_piece_size: int, max_piece_size: int) -> EnvState:
        puzzle = create_puzzle(
            key,
            grid_size=grid_size,
            n_pieces=n_pieces,
            min_piece_size=min_piece_size,
            max_piece_size=max_piece_size,
        ).astype(jnp.float32)

        initial_free_space = jnp.sum(jnp.abs(puzzle[0])).astype(jnp.int32)

        return cls(grid=puzzle[0],
                   next_piece=puzzle[1],
                   other_pieces=puzzle[2:],
                   initial_free_space=initial_free_space,
                   time=0,
                   )

    def roll_top_left(self) -> EnvState:
        next_piece_rolled = roll_top_left(self.next_piece)
        other_pieces_rolled = jax.vmap(roll_top_left)(self.other_pieces)
        return EnvState(grid=self.grid,
                        next_piece=next_piece_rolled,
                        other_pieces=other_pieces_rolled,
                        initial_free_space=self.initial_free_space,
                        time=self.time,
                        )


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
        moved_piece, penalty = padded_translate(
            piece=state.next_piece,
            shift=self.num_to_action(action),
            grid_size=self.grid_size,
        )

        # Add the moved piece to the grid
        new_grid = state.grid + moved_piece

        # The next piece becomes the first of the other pieces
        new_next_piece = state.other_pieces[0]

        # Shift other pieces
        new_other_pieces = jnp.roll(
            state.other_pieces, shift=-1, axis=0)
        new_other_pieces = new_other_pieces.at[-1].set(
            jnp.zeros_like(new_other_pieces[-1]))

        # Calculate reward
        reward = self.calculate_reward(
            grid=state.grid,
            piece=moved_piece,
            penalty=penalty,
            initial_free_space=state.initial_free_space,
            penalty_factor=params.penalty_factor,
        )

        # Update state
        state = state.replace(
            grid=new_grid,
            next_piece=new_next_piece,
            other_pieces=new_other_pieces,
            time=state.time + 1,
        )

        done = self.is_terminal(state, params)

        info = {}

        return (
            jax.lax.stop_gradient(self.get_obs(state)),
            jax.lax.stop_gradient(state),
            reward.astype(jnp.float32),
            done,
            info,
        )

    def num_to_action(self, action: int | float | jax.Array) -> jax.Array:
        """Convert action number to (row_shift, col_shift)."""
        action = jnp.asarray(action).astype(jnp.int32)
        row_shift = action // self.grid_size
        col_shift = action % self.grid_size
        return jnp.array([row_shift, col_shift])

    def calculate_reward(self, grid: jax.Array,
                         piece: jax.Array, penalty: int,
                         initial_free_space: int,
                         penalty_factor: float):

        old_penalty = jnp.abs(grid).sum()
        new_penalty = jnp.abs(grid + piece).sum() + (penalty * penalty_factor)

        # old_penalty_scaled = scale_to_unit_range(
        #     old_penalty,
        #     m=initial_free_space,
        #     penalty_factor=penalty_factor
        # )

        # new_penalty_scaled = scale_to_unit_range(
        #     new_penalty,
        #     m=initial_free_space,
        #     penalty_factor=penalty_factor
        # )

        # reward = new_penalty_scaled - old_penalty_scaled
        reward = (old_penalty - new_penalty) / initial_free_space

        return reward

    def reset_env(self, key: PRNGKeyArray, params: EnvParams):
        state = EnvState.init(
            key,
            grid_size=self.grid_size,
            n_pieces=self.n_pieces,
            min_piece_size=self.min_piece_size,
            max_piece_size=self.max_piece_size,
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
        return state.time >= self.n_pieces

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

    # def render(self, state: EnvState, _: EnvParams):
    #     # Create a custom colormap for the grid
    #     grid = state.grid
    #     vmax = int(state.initial_free_space)

    #     # Define colors: white for -1, copper for [0, vmax]
    #     cmap_copper = plt.get_cmap("copper")
    #     norm = mcolors.Normalize(vmin=0, vmax=vmax)

    #     # Create a new colormap with white for -1
    #     colors = [(1, 1, 1)] + [cmap_copper(norm(i)) for i in range(vmax + 1)]
    #     grid_cmap = mcolors.ListedColormap(colors, name="custom_grid")

    #     # Create a boundary norm to match -1 to 0, 0 to 1, ..., vmax to vmax+1
    #     boundaries = np.arange(-1.5, vmax + 1.5)
    #     grid_norm = mcolors.BoundaryNorm(boundaries, grid_cmap.N)

    #     # Plot
    #     _, axes = plt.subplots(1, self.n_pieces+1, figsize=(self.n_pieces, 1))
    #     for i, ax in enumerate(axes):
    #         if i == 0:
    #             ax.imshow(grid, cmap=grid_cmap, norm=grid_norm)
    #         elif i == 1:
    #             ax.imshow(state.next_piece, cmap=COLORMAPS[i])
    #         else:
    #             ax.imshow(state.other_pieces[i - 2], cmap=COLORMAPS[i])
    #         ax.set_xticks([])
    #         ax.set_yticks([])

    #     return axes

    def render(self, state: EnvState, _: EnvParams):
        grid = state.grid
        vmax = int(state.initial_free_space)
        reward_proxy = 1.0 - (np.sum(np.abs(grid)) / vmax)

        # Create custom colormap with white for -1 and copper for >= 0
        cmap_copper = plt.get_cmap("copper")
        norm = mcolors.Normalize(vmin=0, vmax=vmax)
        colors = [(1, 1, 1)] + [cmap_copper(norm(i)) for i in range(vmax + 1)]
        grid_cmap = mcolors.ListedColormap(colors, name="custom_grid")
        boundaries = np.arange(-1.5, vmax + 1.5)
        grid_norm = mcolors.BoundaryNorm(boundaries, grid_cmap.N)

        # Layout setup
        fig = plt.figure(figsize=(4, 4))
        gs = gridspec.GridSpec(3, 3, height_ratios=[1, 4, 0.3], figure=fig)

        # Top: Pieces
        ax_next = fig.add_subplot(gs[0, 0])
        ax_other1 = fig.add_subplot(gs[0, 1])
        ax_other2 = fig.add_subplot(gs[0, 2])

        ax_next.imshow(state.next_piece, cmap="Purples")
        # ax_next.set_title("Next Piece", fontsize=10)
        ax_other1.imshow(state.other_pieces[0], cmap="Blues")
        # ax_other1.set_title("Piece 2", fontsize=8)
        ax_other2.imshow(state.other_pieces[1], cmap="Reds")
        # ax_other2.set_title("Piece 3", fontsize=8)

        for ax in [ax_next, ax_other1, ax_other2]:
            ax.set_xticks([])
            ax.set_yticks([])

        # Middle: Grid
        ax_grid = fig.add_subplot(gs[1, :])
        ax_grid.imshow(grid, cmap=grid_cmap, norm=grid_norm)
        # ax_grid.set_title("Grid", fontsize=10)
        ax_grid.set_xticks([])
        ax_grid.set_yticks([])

        # Bottom: Progress bar
        ax_bar = fig.add_subplot(gs[2, :])
        # m = vmax  # assuming m = initial_free_space
        ax_bar.barh(0, reward_proxy, color='green', height=0.3)
        ax_bar.set_xlim(0, 1)
        ax_bar.set_yticks([])
        ax_bar.set_xticks([])
        ax_bar.set_xticklabels([])
        # ax_bar.set_title(
        #     f"Reward: {reward_proxy:.0f} / {m}", fontsize=9)

        fig.suptitle(
            f"Number of pieces: {state.time:.0f}/{self.n_pieces:.0f}",
            fontsize=12)

        return fig


COLORMAPS = [
    'Purples',
    'Blues',
    'Greens',
    'Oranges',
    'Reds',
    'Purples',
    'Blues',
    'Greens',
    'Oranges',
    'Reds',
    'Purples',
    'Blues',
    'Greens',
    'Oranges',
    'Reds',
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


def create_puzzle(
        key: PRNGKeyArray,
        grid_size: int = 4,
        n_pieces: int = 4,
        min_piece_size: int = 2,
        max_piece_size: int = 5,
        return_creation: bool = False,):

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
                        key, piece_size), visited

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
                            key, piece_size), visited

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
                            piece_size), visited

                return jax.lax.cond(any_valid, has_valid, no_valid)

            return jax.lax.cond(done | too_long, early_exit, do_step)

        init_state = (start_pos, visited_grid, path_init,
                      path_grid, 0, False, key, piece_size)

        final_state, visited_progression = jax.lax.scan(
            step_fn,
            init=init_state,
            xs=None,
            length=max_piece_size - 1)
        _, final_grid, _, new_path_grid, _, _, _, _ = final_state

        return (new_key, final_grid), (new_path_grid, visited_progression)
    init_carry = (key_walk, init_grid)
    (_, final_state), (pieces, visited_progression) = jax.lax.scan(
        place_piece,
        init=init_carry,
        xs=piece_sizes,
    )
    puzzle = jnp.concat((final_state[None, :] - 1.0, pieces), axis=0)

    if return_creation:
        return puzzle, visited_progression

    return puzzle


def padded_translate(piece: jax.Array, shift: jax.Array, grid_size: int):
    # Step 1: Pad array with zeros
    padded = jnp.pad(
        piece, ((0, grid_size-1), (0, grid_size-1)), mode='constant')

    # Step 2: Roll the padded array
    rolled = jnp.roll(padded, shift=shift, axis=(0, 1))
    rolled_cropped = rolled[:grid_size, :grid_size]
    off_grid_penalty = rolled.sum() - rolled_cropped.sum()
    # Step 3: Crop the central 4x4 region

    # Step 4: get off_grid_penalty as int
    off_grid_penalty = off_grid_penalty.astype(jnp.int32)

    return rolled[:grid_size, :grid_size], off_grid_penalty


def scale_to_unit_range(x: jax.Array, m: int, penalty_factor: int):
    worst = m + m * penalty_factor
    return 1 - 2 * (x / worst)
