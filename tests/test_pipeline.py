"""检查语言模型的数据边界、因果性与训练/推断流程。"""
from dataclasses import replace
from pathlib import Path
import json
import os
import tempfile
import unittest
from unittest import mock

import torch
from torch import nn

from config import ExperimentConfig, MODEL_TYPES, get_device, resolve_data_path, resolve_output_dir
from src.dataset import TimeMachineDataset, prepare_data
from src.engine import evaluate, fit, seed_everything
from src.generation import generate_text, load_model, predict_next
from src.model_factory import create_model
from src.plotting import comparison_table
from src.preprocessing import build_vocab, decode, extract_body, numericalize, split_tokens, tokenize


class DataTests(unittest.TestCase):
    def test_gutenberg_and_word_tokens(self):
        text = (
            'HEADER\n*** START OF THE PROJECT GUTENBERG EBOOK THE TIME MACHINE ***\n'
            "Title\nCONTENTS\n I Introduction\n\n I.\n Introduction\n"
            "The Traveller’s machine! Don't stop—now.\n"
            '*** END OF THE PROJECT GUTENBERG EBOOK THE TIME MACHINE ***\nLICENSE'
        )
        body = extract_body(text)
        self.assertEqual(tokenize(body), ['the', "traveller's", 'machine', "don't", 'stop', 'now'])
        self.assertNotIn('LICENSE', body)
        self.assertNotIn('CONTENTS', body)
        self.assertEqual(extract_body('Plain text without markers.'), 'Plain text without markers.')

    def test_training_only_vocabulary_and_contiguous_splits(self):
        tokens = ['trainword'] * 80 + ['validword'] * 10 + ['testword'] * 10
        splits = split_tokens(tokens)
        self.assertEqual(splits['train'], tokens[:80])
        self.assertEqual(splits['valid'], tokens[80:90])
        self.assertEqual(splits['test'], tokens[90:])
        vocab = build_vocab([splits['train']], min_freq=1)
        self.assertNotIn('validword', vocab)
        self.assertNotIn('testword', vocab)
        self.assertEqual(numericalize(splits['valid'], vocab), [vocab['<unk>']] * 10)

    def test_targets_shift_without_crossing_dataset_boundary(self):
        tokens = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'tail']
        vocab = build_vocab([tokens], min_freq=1)
        dataset = TimeMachineDataset(tokens, vocab, seq_len=4)
        self.assertEqual(len(dataset), 2)
        X0, y0 = dataset[0]
        X1, y1 = dataset[1]
        self.assertEqual(decode(X0, vocab), tokens[:4])
        self.assertEqual(decode(y0, vocab), tokens[1:5])
        self.assertEqual(decode(X1, vocab), tokens[4:8])
        self.assertEqual(decode(y1, vocab), tokens[5:9])
        self.assertEqual(y0[-1].item(), X1[0].item())
        with self.assertRaises(IndexError):
            dataset[2]
        with self.assertRaises(ValueError):
            TimeMachineDataset(tokens[:4], vocab, seq_len=4)

    def test_invalid_configuration_is_rejected(self):
        for kwargs in ({'seq_len': 0}, {'epochs': 0}, {'train_ratio': 0.95}, {'dropout': 1}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                ExperimentConfig(**kwargs)
        self.assertEqual(get_device('cpu'), torch.device('cpu'))


class ModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_all_models_are_causal_and_state_can_continue(self):
        config = ExperimentConfig(embed_size=8, hidden_size=12, num_layers=2)
        X = torch.tensor([[2, 3, 4, 5, 6, 7]])
        changed_future = torch.tensor([[2, 3, 4, 8, 8, 8]])
        for kind in MODEL_TYPES:
            with self.subTest(model=kind):
                seed_everything(config.seed)
                model = create_model(kind, 10, config).eval()
                with torch.no_grad():
                    logits, state = model(X)
                    changed_logits, _ = model(changed_future)
                    first, first_state = model(X[:, :3])
                    second, _ = model(X[:, 3:], first_state)
                self.assertEqual(logits.shape, (1, 6, 10))
                torch.testing.assert_close(logits[:, :3], changed_logits[:, :3])
                torch.testing.assert_close(logits, torch.cat((first, second), dim=1))
                if kind == 'lstm':
                    self.assertEqual(len(state), 2)
                    self.assertEqual(state[0].shape, (2, 1, 12))
                else:
                    self.assertEqual(state.shape, (2, 1, 12))

    def test_training_checkpoint_generation_and_comparison(self):
        config = ExperimentConfig(
            min_freq=1, seq_len=4, batch_size=8, embed_size=8,
            hidden_size=12, epochs=3, learning_rate=0.03,
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            corpus = root / 'tiny.txt'
            corpus.write_text('alpha beta gamma ' * 80, encoding='utf-8')
            output = root / 'outputs'
            for kind in MODEL_TYPES:
                with self.subTest(model=kind):
                    seed_everything(config.seed)
                    data = prepare_data(corpus, config)
                    model = create_model(kind, len(data.vocab), config)
                    result = fit(model, data, config, torch.device('cpu'), output)
                    self.assertLess(result['history'][-1]['train_loss'], result['history'][0]['train_loss'])
                    restored, vocab, checkpoint = load_model(result['checkpoint_path'])
                    self.assertEqual(checkpoint['best_epoch'], result['metrics']['best_epoch'])
                    self.assertEqual(result['metrics']['valid']['loss'], min(row['valid_loss'] for row in result['history']))
                    measured = evaluate(restored, data.loaders['test'], nn.CrossEntropyLoss(), torch.device('cpu'))
                    self.assertAlmostEqual(measured['loss'], result['metrics']['test']['loss'])
                    for original, loaded in zip(model.parameters(), restored.parameters()):
                        torch.testing.assert_close(original, loaded)
                    candidates = predict_next(restored, vocab, 'alpha beta', top_k=3)
                    self.assertAlmostEqual(sum(row['probability'] for row in candidates), 1.0, places=5)
                    self.assertNotIn('<unk>', [row['token'] for row in candidates])
                    restored.train()
                    sentence = generate_text(restored, vocab, 'alpha beta', num_words=10, seed=7)
                    self.assertTrue(restored.training)
                    self.assertEqual(len(sentence.split()), 12)
                    self.assertEqual(sentence, generate_text(restored, vocab, 'alpha beta', num_words=10, seed=7))
                    self.assertNotIn('<pad>', sentence)
                    self.assertNotIn('<unk>', sentence)
                    with self.assertRaises(ValueError):
                        generate_text(restored, vocab, '!!!')
                    greedy = generate_text(restored, vocab, 'alpha beta', num_words=1, temperature=0)
                    self.assertEqual(greedy.split()[-1], candidates[0]['token'])
            table = comparison_table(output)
            self.assertEqual(set(table['model']), {'RNN', 'LSTM', 'GRU'})
            path = output / 'gru' / 'metrics.json'
            metrics = json.loads(path.read_text())
            metrics['experiment_config'] = vars(replace(config, seed=43))
            path.write_text(json.dumps(metrics), encoding='utf-8')
            with self.assertRaises(ValueError):
                comparison_table(output)


class PlatformTests(unittest.TestCase):
    def test_explicit_and_environment_paths(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            corpus = root / 'pg35.txt'
            corpus.write_text('the time traveller', encoding='utf-8')
            with mock.patch.dict(os.environ, {
                'TIME_MACHINE_DATA_PATH': str(corpus),
                'TIME_MACHINE_OUTPUT_DIR': str(root / 'results'),
            }):
                self.assertEqual(resolve_data_path(), corpus.resolve())
                self.assertEqual(resolve_output_dir(), root / 'results')
                self.assertEqual(resolve_output_dir(root / 'explicit'), root / 'explicit')
                with self.assertRaises(FileNotFoundError):
                    resolve_data_path(root / 'missing.txt')

    def test_kaggle_mount_discovery_and_ambiguity(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            input_root = root / 'kaggle/input'
            working_root = root / 'kaggle/working'
            first = input_root / 'dataset-a/pg35.txt'
            first.parent.mkdir(parents=True)
            first.write_text('the time traveller', encoding='utf-8')
            working_root.mkdir(parents=True)

            def mapped_path(value):
                paths = {'/kaggle/input': input_root, '/kaggle/working': working_root,
                         '/kaggle/working/time_machine': working_root / 'time_machine'}
                return paths.get(str(value), Path(value))

            with mock.patch.dict(os.environ, {}, clear=True), \
                    mock.patch('config.PROJECT_ROOT', root / 'code'), \
                    mock.patch('config.Path', side_effect=mapped_path):
                self.assertEqual(resolve_data_path(), first)
                self.assertEqual(resolve_output_dir(), working_root / 'time_machine')
                second = input_root / 'dataset-b/pg35.txt'
                second.parent.mkdir()
                second.write_text('another corpus', encoding='utf-8')
                with self.assertRaises(ValueError):
                    resolve_data_path()

    def test_device_selection_and_explicit_unavailable_device(self):
        with mock.patch('torch.cuda.is_available', return_value=True):
            self.assertEqual(get_device('auto'), torch.device('cuda'))
        with mock.patch('torch.cuda.is_available', return_value=False), \
                mock.patch('torch.backends.mps.is_available', return_value=True):
            self.assertEqual(get_device('auto'), torch.device('mps'))
        with mock.patch('torch.cuda.is_available', return_value=False), \
                mock.patch('torch.backends.mps.is_available', return_value=False):
            self.assertEqual(get_device('auto'), torch.device('cpu'))
            with self.assertRaises(RuntimeError):
                get_device('cuda')


if __name__ == '__main__':
    unittest.main()
