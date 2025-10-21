git submodule update --init --recursive rust
git submodule update --remote
(cd rust && uv pip uninstall hiivelabs-zertz-mcts)
bash -c 'source "$HOME/.cargo/env" && unset CONDA_PREFIX && (cd rust && uv run python -m maturin build --release)'
