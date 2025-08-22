import torch
import yaml
from model_zoo.ddpm import DDPM


def thor_inference(batch: torch.Tensor, modality: str, noise: str, weights_dir: str = "./weights") -> torch.Tensor:
    """Run THOR anomaly detection on a batch tensor.

    Parameters
    ----------
    batch : torch.Tensor
        Input images in ``[0,1]`` range with shape ``(B, C, H, W)`` or ``(B, H, W)``.
    modality : str
        Either ``"brain"`` or ``"wxr"`` to select the proper config.
    noise : str
        Either ``"gaussian"`` or ``"simplex"``.
    weights_dir : str, optional
        Directory containing the pretrained weights named
        ``<modality>_<Noise>.pt`` (e.g. ``brain_Gaussian.pt``), by default
        ``"./weights"``.

    Returns
    -------
    torch.Tensor
        Anomaly map tensor on CPU with shape matching the input.
    """
    # Ensure input on CPU and correct dimensionality
    batch = batch.to("cpu")
    if batch.ndim == 3:
        batch = batch.unsqueeze(1)
    if batch.ndim != 4:
        raise ValueError("Input must have shape (B,C,H,W) or (B,H,W)")

    if batch.max() > 1 or batch.min() < 0:
        raise ValueError("Input tensor should be scaled to [0,1]")

    # Normalize to [-1,1]
    batch = batch * 2 - 1

    modality = modality.lower()
    noise = noise.lower()
    if modality not in {"brain", "wxr"}:
        raise ValueError("modality must be 'brain' or 'wxr'")
    if noise not in {"gaussian", "simplex"}:
        raise ValueError("noise must be 'gaussian' or 'simplex'")

    config_name = "thor.yaml" if noise == "gaussian" else "thor_simplex.yaml"
    config_path = f"projects/thor/configs/{modality}/{config_name}"

    with open(config_path, "r") as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)
    model_params = cfg["model"]["params"]

    model = DDPM(**model_params)

    weight_file = f"{weights_dir}/{modality}_{noise.capitalize()}.pt"
    checkpoint = torch.load(weight_file, map_location="cpu")
    model.load_state_dict(checkpoint["model_weights"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device).eval()
    batch = batch.to(device)

    with torch.no_grad():
        recon, _ = model.sample_from_image(batch, noise_level=model.noise_level_recon)

    recon = (recon + 1) / 2
    orig = (batch + 1) / 2
    anomaly_map = torch.abs(orig - recon)
    return anomaly_map.cpu()


if __name__ == "__main__":
    # Example usage with a dummy batch of zeros
    dummy = torch.zeros(1, 1, 128, 128)
    try:
        anomaly = thor_inference(dummy, "brain", "gaussian")
        print(anomaly.shape)
    except FileNotFoundError:
        print("Pretrained weights not found. Place weights in ./weights directory.")
