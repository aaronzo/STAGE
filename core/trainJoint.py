from core.LMs.joint_trainer import JointTrainer
from core.config import cfg, update_cfg
import pandas as pd
import logging

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def run(cfg):
    seeds = [cfg.seed] if cfg.seed is not None else range(cfg.runs)
    all_acc = []
    for seed in seeds:
        logger.info(f"Running seed {seed}...")
        cfg.seed = seed
        trainer = JointTrainer(cfg)
        logger.info("Trainer loaded!")
        logger.info("Beginning Training...")
        trainer.train()
        logger.info("Training Fnished!")
        acc = trainer.eval_and_save()
        all_acc.append(acc)

    if len(all_acc) > 1:
        df = pd.DataFrame(all_acc)
        for k, v in df.items():
            print(f"{k}: {v.mean():.4f} ± {v.std():.4f}")


if __name__ == '__main__':
    cfg = update_cfg(cfg)
    run(cfg)
