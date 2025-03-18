import torch
import pandas as pd
from box import Box
import numpy as np
import os
import logging
from gpudrive.env.dataset import SceneDataLoader
from eval_utils import (
    load_config,
    make_env,
    load_policy,
    evaluate_policy,
)
from gpudrive.datatypes.trajectory import LogTrajectory
from gpudrive.datatypes.metadata import Metadata

import random
import torch
import numpy as np

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # If using CUDA
    torch.backends.cudnn.deterministic = True

logging.basicConfig(level=logging.INFO)
SEED = 42  # Set to any fixed value
set_seed(SEED)

if __name__ == "__main__":

    # Load configurations
    eval_config = load_config("examples/experimental/config/eval_config")
    model_config = load_config("examples/experimental/config/model_config")

    # Make environment

    for model in model_config.models:

        logging.info(f"Evaluating model {model.name}")

        # Load policy
        policy = load_policy(
            path_to_cpt=model_config.models_path,
            model_name=model.name,
            device=eval_config.device,
        )

        data_loader = SceneDataLoader(
            root=eval_config.test_dir,
            batch_size=eval_config.num_worlds,
            dataset_size=eval_config.test_dataset_size,
            sample_with_replacement=False,
            shuffle=True,
            file_prefix="nuplan"
        )
        env = make_env(eval_config, data_loader)

        # Rollouts
        logging.info(
            f"Rollouts on {len(set(data_loader.dataset))} scenes"
        )

        df_res = evaluate_policy(
            env=env,
            policy=policy,
            data_loader=data_loader,
            dataset_name="ego_only",
            deterministic=False,
            render_sim_state=False,
            rollout_only_ego=True
        )

        # Add metadata
        df_res["model_name"] = model.name
        df_res["train_dataset_size"] = model.train_dataset_size

        # Store
        if not os.path.exists(eval_config.res_path):
            os.makedirs(eval_config.res_path)

        log_trajectory = LogTrajectory.from_tensor(
            env.sim.expert_trajectory_tensor(),
            env.num_worlds,
            env.max_agent_count,
            backend=env.backend,
        )

        # Index log positions at current time steps
        log_traj_pos = []
        for i in range(env.num_worlds):
            log_traj_pos.append(
                log_trajectory.pos_xy[i, :, env.episode_len, :]
            )
        log_traj_pos_tensor = torch.stack(log_traj_pos)

        sdc_mask = Metadata.from_tensor(env.sim.metadata_tensor(), backend="torch").isSdc.argmax(dim=1)
        agent_pos = df_res["agent_positions"].gather(env.num_worlds, sdc_mask.unsqueeze(1))

        # compute euclidean distance between agent and logs
        dist_to_logs = torch.norm(log_traj_pos_tensor - agent_pos, dim=-1)

        print(dist_to_logs)
