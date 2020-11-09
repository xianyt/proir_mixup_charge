import logging
import os
import shutil
from argparse import ArgumentParser
from functools import reduce
from os import path

import torch.optim as optim
from ignite.contrib.handlers import ProgressBar
from ignite.engine import Engine, Events
from ignite.handlers import EarlyStopping
from ignite.metrics import RunningAverage, Precision, Recall, Average, Accuracy, MetricsLambda
from torch import nn
from torch.optim.lr_scheduler import StepLR

from sklearn.metrics import *

from model import StructAttMixup
from pred_saver import PredSaver
from rnn_encoder import RnnEncoder
from utils import *

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(name)s %(levelname)s %(message)s",
                    datefmt='%Y-%m-%d  %H:%M:%S %a')


def get_args():
    parser = ArgumentParser()

    parser.add_argument("--data_dir", type=str, default='data/attribute_charge/data')
    parser.add_argument('--train', type=str, default='train', help='train file')
    parser.add_argument('--dev', type=str, default='valid', help='dev file')
    parser.add_argument('--test', type=str, default='test', help='test file '),
    parser.add_argument('--tokenized', action="store_true"),
    parser.add_argument('--pad_left', action="store_true"),

    parser.add_argument('--multi_label', action='store_true'),

    parser.add_argument('--token_min_count', type=int, default=5),
    parser.add_argument('--label_min_count', type=int, default=0),

    parser.add_argument("--cache", type=str, default='cache/small')

    parser.add_argument('--tokenizer', type=str, default='')

    parser.add_argument("--encoder", type=str, default='rnn')

    # ------- arguments form RNN encoder --------
    parser.add_argument("--embed_dim", type=int, default=100)
    parser.add_argument("--embed_weights", type=str, default='data/attribute_charge/data/words.vec')

    parser.add_argument("--rnn_type", type=str, default="lstm")
    parser.add_argument("--rnn_dim", type=int, default=300)
    parser.add_argument("--rnn_dropout", type=float, default=0.1)
    parser.add_argument("--rnn_layer_num", type=int, default=1)
    parser.add_argument("--rnn_bidir", action="store_true")

    # ------- arguments form Struct --------
    parser.add_argument("--summary_type", type=str, nargs='+', default=['struct_att'],
                        help='["max", "mean", "struct_att"]')
    parser.add_argument("--attention_head", type=int, default=24)
    parser.add_argument("--attention_dim", type=int, default=64)

    parser.add_argument("--disable_mixup", action="store_true")
    parser.add_argument("--mixup_beta_concentration", type=float, default=150.0)
    parser.add_argument("--mixup_type", type=str, default='prior_mix', help='[mixup, prior_mix]')

    parser.add_argument("--dropout", type=float, default=0.1, help="dropout rate for features")
    parser.add_argument("--max_seq_len", type=int, default=500, help="max length of sequence")

    parser.add_argument("--train_batch_size", type=int, default=128, help="Batch size for training")
    parser.add_argument("--valid_batch_size", type=int, default=128, help="Batch size for validation")

    parser.add_argument("--alpha", type=float, default=1.0)

    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=0.0, help="weight decay")

    parser.add_argument("--max_epochs", type=int, default=50, help="Number of training epochs")

    parser.add_argument("--output_path", type=str, default='./out_small', help="checkpoint directory")

    parser.add_argument("--early_stop_patience", type=int, default=10)
    parser.add_argument("--early_stop_metric", type=str, default='loss', help='loss, acc, MP, MR, F1')

    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu",
                        help="Device (cuda or cpu)")

    cfg = parser.parse_args()

    return cfg


def get_model(cfg):
    if cfg.encoder == 'rnn':
        encoder = RnnEncoder(cfg.rnn_type, cfg.vocab_size, cfg.embed_dim, cfg.rnn_dim,
                             cfg.rnn_dropout, cfg.rnn_layer_num, cfg.rnn_bidir)
        if cfg.embed_weights:
            logger.info('Loading embedding weights from %s' % cfg.embed_weights)
            weights = load_embeddings(cfg.vocab, cfg.embed_weights, cfg.embed_dim)
            weights = torch.tensor(weights, dtype=torch.float).to(cfg.device)
            if encoder.load_embed_weights:
                encoder.load_embed_weights(weights)
    else:
        raise Exception('unsupported encoder %s' % cfg.encoder)

    return StructAttMixup(cfg, encoder)


def get_data_loaders(cfg, data):
    token2id, label2id, data_train, data_dev, data_test = data

    def _count(c, it):
        y, x = it
        for yy in y:
            c[yy] += 1
        return c

    label_count = [0] * len(label2id)
    label_count = reduce(_count, data_train, label_count)

    # data_train = data_train[:100]
    # data_dev = data_dev[:100]
    # data_test = data_test[:100]

    num_train_batch = math.ceil(len(data_train) / cfg.train_batch_size)
    train_gen = data_generator(data_train, cfg.train_batch_size)

    num_dev_batch = math.ceil(len(data_dev) / cfg.valid_batch_size)
    dev_gen = data_generator(data_dev, cfg.valid_batch_size)

    num_test_batch = math.ceil(len(data_test) / cfg.valid_batch_size)
    test_gen = data_generator(data_test, cfg.valid_batch_size)

    cfg.vocab = token2id
    cfg.vocab_size = len(token2id)
    cfg.num_class = len(label2id)
    cfg.label_count = label_count
    cfg.num_train_batch = num_train_batch
    cfg.num_train_samples = len(data_train)

    cfg.num_dev_batch = num_dev_batch
    cfg.num_dev_samples = len(data_dev)

    cfg.num_test_batch = num_test_batch
    cfg.num_test_samples = len(data_test)

    cfg.id2label = {v: k for k, v in label2id.items()}

    return (token2id, label2id, train_gen, num_train_batch,
            dev_gen, num_dev_batch, test_gen, num_test_batch)


def initialize(cfg):
    model = get_model(cfg).to(cfg.device)
    optimizer = optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    # optimizer = optim.Adam(model.parameters(), lr=cfg.lr)

    if cfg.multi_label:
        criterion = nn.BCEWithLogitsLoss()
    else:
        criterion = nn.CrossEntropyLoss()

    le = cfg.num_train_batch
    lr_scheduler = StepLR(optimizer, step_size=le, gamma=0.9)

    return model, optimizer, criterion, lr_scheduler


def batch_to_tensor(batch, cfg):
    y, x = batch

    if cfg.multi_label:
        y_tensor = torch.zeros((len(y), cfg.num_class), dtype=torch.float32)
        idx_i, idx_j = [], []
        for i, l in enumerate(y):
            for j in l:
                idx_i.append(i)
                idx_j.append(j)
        indices = (torch.tensor(idx_i, dtype=torch.long), torch.tensor(idx_j, dtype=torch.long))
        y_tensor.index_put_(indices, torch.ones(len(idx_i), dtype=torch.float32))
    else:
        y = [v[0] for v in y]
        y_tensor = torch.tensor(y, dtype=torch.long)

    max_len = max([len(s) for s in x])
    if cfg.max_seq_len > 0:
        max_len = min(max_len, cfg.max_seq_len)

    x = [pad_seq(s, max_len, pad_left=cfg.pad_left) for s in x]
    x_tensor = torch.tensor(x, dtype=torch.long)

    return x_tensor, y_tensor


def create_trainer(model, optimizer, criterion, lr_scheduler, cfg):
    # Define any training logic for iteration update
    log_softmax = nn.LogSoftmax(dim=-1)

    def _train_step(engine, batch):
        x, y = batch_to_tensor(batch, cfg)
        x, y = x.to(cfg.device), y.to(cfg.device)

        model.train()

        logits, mix_logits, mix_labels = model(x, y)
        loss = criterion(logits, y)

        if not cfg.disable_mixup:
            mixup_loss = -(log_softmax(mix_logits) * mix_labels).sum(dim=-1).mean()
            loss += cfg.alpha * mixup_loss
        else:
            mixup_loss = 0.

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        lr_scheduler.step()

        if cfg.multi_label:
            logits = (logits > 0).float()

        return logits, y, loss.item(), mixup_loss

    # Define trainer engine
    trainer = Engine(_train_step)

    # Define metrics for trainer
    acc = Accuracy(lambda x: x[0:2], is_multilabel=cfg.multi_label)
    RunningAverage(acc).attach(trainer, 'acc')
    RunningAverage(output_transform=lambda x: x[2]).attach(trainer, 'loss')

    metrics = ['loss', 'acc']
    if not cfg.disable_mixup:
        RunningAverage(output_transform=lambda x: x[3]).attach(trainer, 'mixup_loss')
        metrics.append('mixup_loss')

    # Add progress bar showing trainer metrics
    ProgressBar(persist=True).attach(trainer, metrics)

    return trainer


def create_evaluator(model, criterion, cfg):
    def _validation_step(_, batch):
        model.eval()
        with torch.no_grad():
            x, y = batch_to_tensor(batch, cfg)
            x, y = x.to(cfg.device), y.to(cfg.device)

            y_pred, hidden = model(x)
            loss = criterion(y_pred, y)

            if cfg.multi_label:
                y_pred = (y_pred > 0).float()

            return y_pred, y, loss, hidden

    evaluator = Engine(_validation_step)

    accuracy = Accuracy(lambda x: x[0:2], is_multilabel=cfg.multi_label)
    accuracy.attach(evaluator, "acc")

    precision = Precision(lambda x: x[0:2], average=False, is_multilabel=cfg.multi_label)
    precision.attach(evaluator, 'precision')
    MetricsLambda(lambda t: torch.mean(t).item(), precision).attach(evaluator, "MP")

    recall = Recall(lambda x: x[0:2], average=False, is_multilabel=cfg.multi_label)
    recall.attach(evaluator, 'recall')
    MetricsLambda(lambda t: torch.mean(t).item(), recall).attach(evaluator, "MR")

    F1 = 2. * precision * recall / (precision + recall + 1e-20)
    f1 = MetricsLambda(lambda t: torch.mean(t).item(), F1)
    f1.attach(evaluator, "F1")

    Average(lambda x: x[2]).attach(evaluator, 'loss')

    return evaluator


def training(cfg, data):
    # create data generators
    (vocab, classes, train_gen, num_train_batch,
     val_gen, num_val_batch, test_gen, num_test_batch) = get_data_loaders(cfg, data)

    # create model, optimizer, criterion and learning rate scheduler
    model, optimizer, criterion, lr_scheduler = initialize(cfg)

    # print settings
    print_table([(k, str(v)[0:60]) for k, v in vars(cfg).items()])

    # print model parameters
    print(parameters_string(model))

    # Setup model trainer and evaluator
    trainer = create_trainer(model, optimizer, criterion, lr_scheduler, cfg)
    evaluator = create_evaluator(model, criterion, cfg)
    tester = create_evaluator(model, criterion, cfg)

    # Run model evaluation every epochs and show results
    @trainer.on(Events.EPOCH_COMPLETED(every=1))
    def evaluate_model():
        val_state = evaluator.run(val_gen, epoch_length=num_val_batch)
        eval_metrics = [(m, val_state.metrics[m]) for m in ['acc', 'MP', 'MR', 'F1', 'loss']]
        eval_metrics = ", ".join([("%s: %.4f" % (m, v)) for m, v in eval_metrics])
        logger.info(eval_metrics)

    def score_function(_):
        if cfg.early_stop_metric == 'loss':
            return - evaluator.state.metrics['loss']
        elif cfg.early_stop_metric in evaluator.state.metrics:
            return evaluator.state.metrics[cfg.early_stop_metric]
        else:
            raise Exception('unsupported metric %s' % cfg.early_stop_metric)

    @evaluator.on(Events.EPOCH_COMPLETED)
    def save_checkpoint():
        best_score = getattr(evaluator.state, 'best_score', None)
        epoch_score = score_function(evaluator)
        epoch = trainer.state.epoch

        os.makedirs(cfg.output_path, exist_ok=True)
        if best_score is None or epoch_score > best_score:
            checkpoint = "checkpoint_%03d_%.4f.pt" % (epoch, epoch_score)
            checkpoint = path.join(cfg.output_path, checkpoint)
            torch.save(model.state_dict(), checkpoint)

            evaluator.state.best_score = epoch_score
            evaluator.state.best_epoch = epoch

            best_checkpoint = path.join(cfg.output_path, "best.pt")
            shutil.copy(checkpoint, best_checkpoint)

    hdl_early_stop = EarlyStopping(patience=cfg.early_stop_patience, score_function=score_function, trainer=trainer)
    # Note: the handler is attached to an *Evaluator* (runs one epoch on validation dataset).
    evaluator.add_event_handler(Events.COMPLETED, hdl_early_stop)

    hdl_hidden_saver = None
    hdl_pred_saver = PredSaver(lambda x: (x[1], x[0] if cfg.multi_label else torch.max(x[0], dim=-1)[1]))
    tester.add_event_handler(Events.ITERATION_COMPLETED, hdl_pred_saver)

    trainer.run(train_gen, max_epochs=cfg.max_epochs, epoch_length=num_train_batch)

    best_ckpt = path.join(cfg.output_path, "best.pt")
    best_epoch = evaluator.state.best_epoch
    logger.info("Load best model checkpoint from Epoch[%d], '%s'" % (best_epoch, best_ckpt))

    state_dict = torch.load(best_ckpt)
    model.load_state_dict(state_dict)

    logger.info("Evaluate model on test data, %s" % cfg.test)
    test_state = tester.run(test_gen, epoch_length=num_test_batch)
    test_metrics = [(m, test_state.metrics[m]) for m in ['acc', 'MP', 'MR', 'F1', 'loss']]
    test_metrics = ", ".join([("%s: %.4f" % (m, v)) for m, v in test_metrics])
    logger.info(test_metrics)

    y_true = hdl_pred_saver.y_true.long().numpy().tolist()
    y_pred = hdl_pred_saver.y_pred.long().numpy().tolist()
    labels = [cfg.id2label[i] for i in range(len(cfg.id2label))]
    rep = classification_report(y_true, y_pred,
                                labels=[la for la in range(len(labels))],
                                target_names=labels, zero_division=0, digits=4)
    print(rep)


def preprocess(cfg):
    f_train = path.join(cfg.data_dir, cfg.train)
    f_dev = path.join(cfg.data_dir, cfg.dev)
    f_test = path.join(cfg.data_dir, cfg.test)

    token2id = None
    encoder = None
    if not cfg.tokenized:
        tokenizer, token2id, encoder = create_tokenizer(cfg)

        f_tok_train = path.join(cfg.data_dir, 'tok_%s' % cfg.train)
        f_tok_dev = path.join(cfg.data_dir, 'tok_%s' % cfg.dev)
        f_tok_test = path.join(cfg.data_dir, 'tok_%s' % cfg.test)

        logger.info("tokenize {}".format(f_train))
        tokenize_file(tokenizer, f_train, f_tok_train)

        logger.info("tokenize {}".format(f_dev))
        tokenize_file(tokenizer, f_dev, f_tok_dev)

        logger.info("tokenize {}".format(f_test))
        tokenize_file(tokenizer, f_test, f_tok_test)
    else:
        f_tok_train, f_tok_dev, f_tok_test = f_train, f_dev, f_test

    if token2id is None and encoder is None:
        logger.info("Build vocabulary from '%s'" % f_train)
        vocab = build_vocab(f_tok_train, cfg.token_min_count, cfg.multi_label)
        token2id = {t: i for i, t in enumerate(vocab)}

    label_count = count_label(f_train)
    label2id = {la: i for i, la in enumerate(label_count.keys())}

    logger.info("Convert token and label to id.")
    data_train = token_to_id(f_tok_train, token2id, label2id, cfg.multi_label, encoder, cfg.max_seq_len)
    data_dev = token_to_id(f_tok_dev, token2id, label2id, cfg.multi_label, encoder, cfg.max_seq_len)
    data_test = token_to_id(f_tok_test, token2id, label2id, cfg.multi_label, encoder, cfg.max_seq_len)

    logger.info("save preprocessed data to cache %s" % cfg.cache)

    os.makedirs(re.sub("/[^/]+$", "", cfg.cache), exist_ok=True)
    torch.save([token2id, label2id, data_train, data_dev, data_test], cfg.cache)

    return token2id, label2id, data_train, data_dev, data_test


# --- Single computation device ---
if __name__ == "__main__":
    logger = logging.getLogger('main')

    config = get_args()

    if not path.isfile(config.cache):
        dataset = preprocess(config)
    else:
        logger.info("Load data from cache: '%s'" % config.cache)
        dataset = torch.load(config.cache)

    torch.manual_seed(2357)
    np.random.seed(2357)

    if 'cuda' in config.device:
        torch.cuda.manual_seed(2357)

    training(config, dataset)
    exit(0)
