"""Mode and gradient checks for head-only foundation-model training."""

import unittest

import numpy as np
import torch
from torch import nn

from utils.EEGDataLoader import EEGData
from utils.ModelLoader import ModelLoader
from utils.optimizer import create_optimizer


class ToyLoader(ModelLoader):
    def __init__(self, strategy="head_only", with_adapter=True):
        dataset = EEGData(
            dataset_name="toy",
            eeg_data=np.zeros((2, 4, 4)),
            subject_ids=[0, 1],
            channel_names=["a", "b", "c", "d"],
            sampling_rate=4,
            labels=[0, 1],
        )
        super().__init__(dataset, strategy)
        self.main_model = nn.Sequential(nn.Linear(4, 4), nn.BatchNorm1d(4), nn.Dropout(0.5))
        self.vqvae = nn.Sequential(nn.Linear(4, 4), nn.Dropout(0.5))
        if with_adapter:
            self.chan_conv = nn.Linear(4, 4)
        self.scale_param = nn.Parameter(torch.ones(1))
        self.task_head = nn.Sequential(nn.Dropout(0.2), nn.Linear(4, 2))
        self.apply_finetune_strategy()

    def forward(self, x):
        if hasattr(self, "chan_conv"):
            x = self.chan_conv(x)
        return self.task_head(self.main_model(x))


class HeadOnlyModeTests(unittest.TestCase):
    def test_frozen_modules_stay_in_eval_and_adapter_receives_gradients(self):
        model = ToyLoader()
        model.train()

        self.assertTrue(model.training)
        self.assertFalse(model.main_model.training)
        inputs = torch.randn(8, 4)
        self.assertTrue(torch.equal(model.main_model(inputs), model.main_model(inputs)))
        self.assertFalse(model.vqvae.training)
        self.assertTrue(model.chan_conv.training)
        self.assertTrue(model.task_head.training)
        trainable = {name for name, param in model.named_parameters() if param.requires_grad}
        self.assertEqual(
            trainable,
            {"chan_conv.weight", "chan_conv.bias", "task_head.1.weight", "task_head.1.bias"},
        )

        running_mean = model.main_model[1].running_mean.clone()
        model(torch.randn(8, 4)).square().sum().backward()
        self.assertTrue(torch.equal(model.main_model[1].running_mean, running_mean))
        self.assertIsNotNone(model.chan_conv.weight.grad)
        self.assertIsNotNone(model.task_head[1].weight.grad)
        self.assertIsNone(model.main_model[0].weight.grad)

        optimizer = create_optimizer("adamw", 0.001, 0.01, model=model, filter_bias_and_bn=False)
        optimizer_params = {id(param) for group in optimizer.param_groups for param in group["params"]}
        self.assertEqual(optimizer_params, {id(param) for param in model.parameters() if param.requires_grad})

        model.eval()
        self.assertFalse(model.task_head.training)
        self.assertFalse(model.chan_conv.training)
        model.train()
        self.assertFalse(model.main_model.training)
        self.assertTrue(model.task_head.training)

    def test_head_only_without_adapter_and_full_finetuning(self):
        model = ToyLoader(with_adapter=False)
        model.train()
        self.assertEqual(
            {name for name, param in model.named_parameters() if param.requires_grad},
            {"task_head.1.weight", "task_head.1.bias"},
        )
        self.assertFalse(model.main_model.training)

        full_model = ToyLoader(strategy="full")
        full_model.train()
        self.assertTrue(full_model.main_model.training)
        self.assertTrue(all(param.requires_grad for param in full_model.parameters()))

        model.apply_finetune_strategy("full")
        self.assertTrue(model.main_model.training)
        self.assertTrue(all(param.requires_grad for param in model.parameters()))


if __name__ == "__main__":
    unittest.main()
