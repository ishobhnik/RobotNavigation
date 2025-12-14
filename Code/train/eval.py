import torch
from vint_train.models.nomad.nomad import NOMAD
from vint_train.training.train_utils import evaluate_nomad
from config import get_config  # your existing config loader

def main():
    config = get_config()

    # Load model
    model = NOMAD(config)
    checkpoint = torch.load("logs/nomad/nomad_2025_04_06_19_06_12/ema_latest.pth", map_location='cuda')
    model.load_state_dict(checkpoint)
    model.eval()
    model.cuda()

    # Run evaluation
    evaluate_nomad(
        config=config,
        model=model,
        eval_dataloader=...  # load your test dataloader here
    )

if __name__ == "__main__":
    main()
