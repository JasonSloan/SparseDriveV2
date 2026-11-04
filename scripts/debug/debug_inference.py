"""Debug one cached scene through the SparseDriveV2 inference pipeline."""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader, Subset

from navsim.agents.sparsedrive.sparsedrive_agent import SparseDriveAgent
from navsim.agents.sparsedrive.sparsedrive_config import SparseDriveConfig
from navsim.planning.training.dataset import CacheOnlyDataset


def move_to_device(value, device):
    if isinstance(value, torch.Tensor):
        return value.to(device)
    if isinstance(value, dict):
        return {key: move_to_device(item, device) for key, item in value.items()}
    if isinstance(value, list):
        return [move_to_device(item, device) for item in value]
    if isinstance(value, tuple):
        return tuple(move_to_device(item, device) for item in value)
    return value


def save_trajectory_plot(prediction, targets, token, output_dir):
    """Save a bird's-eye plot of the prediction against cached targets."""
    prediction = prediction.detach().cpu()[0].numpy()
    gt_trajectory = targets["trajectory"][0].detach().cpu().numpy()
    gt_path = targets["path"][0].detach().cpu().numpy()
    path_mask = targets["path_mask"][0].detach().cpu().numpy().astype(bool)

    # Ego-frame x points forward and y points left; plot forward on the vertical axis.
    fig, ax = plt.subplots(figsize=(7, 8))
    ax.plot(gt_path[path_mask, 1], gt_path[path_mask, 0], color="0.55", linestyle="--",
            marker=".", label="GT path (50 points)")
    ax.plot(gt_trajectory[:, 1], gt_trajectory[:, 0], color="tab:green", marker="o",
            label="GT trajectory (4 s)")
    ax.plot(prediction[:, 1], prediction[:, 0], color="tab:red", marker="o",
            label="SparseDrive prediction (4 s)")
    ax.scatter([0], [0], color="black", marker="s", label="Ego at t=0", zorder=5)
    ax.set_xlabel("y: left / right (m)")
    ax.set_ylabel("x: forward (m)")
    ax.set_title(f"Trajectory prediction — {token}")
    ax.set_aspect("equal", adjustable="datalim")
    ax.grid(True, alpha=0.3)
    ax.legend()

    output_dir.mkdir(parents=True, exist_ok=True)
    safe_token = str(token).replace("/", "_")
    output_path = output_dir / f"{safe_token}_trajectory.png"
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return output_path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default="ckpt/sparsedrive_navsimv2.ckpt")
    parser.add_argument("--cache-path", type=Path, default=Path("exp/data_cache_navtest"))
    parser.add_argument("--log-name", help="Cache log folder; defaults to the first sorted log.")
    parser.add_argument("--sample-index", type=int, default=0, help="Sample index within the selected log.")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--plot-dir", type=Path, default=Path("exp/debug_inference/plots"),
        help="Directory for the predicted-versus-GT trajectory plot.",
    )
    args = parser.parse_args()

    torch.manual_seed(0)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable in the selected Python environment.")
    if not args.cache_path.is_dir():
        raise FileNotFoundError(f"Cache directory not found: {args.cache_path}")
    log_names = sorted(path.name for path in args.cache_path.iterdir() if path.is_dir())
    if not log_names:
        raise RuntimeError(f"No cached logs found in {args.cache_path}")
    log_name = args.log_name or log_names[0]
    if log_name not in log_names:
        raise FileNotFoundError(f"Cached log not found: {args.cache_path / log_name}")

    print("Loading SparseDriveV2 and checkpoint...", flush=True)
    agent = SparseDriveAgent(
        config=SparseDriveConfig(), lr=1e-4, checkpoint_path=args.checkpoint
    )
    agent.initialize()
    agent.to(device).eval()

    dataset = CacheOnlyDataset(
        cache_path=str(args.cache_path),
        feature_builders=agent.get_feature_builders(),
        target_builders=agent.get_target_builders(),
        test_mode=True,
        log_names=[log_name],
    )
    dataset.tokens.sort()
    if not 0 <= args.sample_index < len(dataset):
        raise IndexError(f"Sample index must be in [0, {len(dataset)}), got {args.sample_index}")
    loader = DataLoader(
        Subset(dataset, [args.sample_index]), batch_size=1, num_workers=0
    )
    features, targets, tokens = next(iter(loader))
    features = move_to_device(features, device)
    print(f"Log: {log_name}\nToken: {tokens[0]}", flush=True)
    print(f"Input images: {tuple(features['camera_feature']['imgs'].shape)}", flush=True)

    # Step into this call to follow model -> backbone -> trajectory head -> decoder.
    with torch.no_grad():
        predictions, loss_dict = agent(features, None)

    trajectory = predictions["trajectory"].detach().cpu()
    if not torch.isfinite(trajectory).all():
        raise RuntimeError("Model output contains non-finite trajectory values.")
    print(f"Output trajectory: {tuple(trajectory.shape)}", flush=True)
    print(trajectory[0], flush=True)
    print(f"Inference loss keys: {list(loss_dict)}", flush=True)
    plot_path = save_trajectory_plot(trajectory, targets, tokens[0], args.plot_dir)
    print(f"Trajectory plot: {plot_path}", flush=True)


if __name__ == "__main__":
    main()
