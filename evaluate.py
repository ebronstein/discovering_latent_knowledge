import argparse
import json
import os

from sklearn.linear_model import LogisticRegression

from utils import (
    CCS,
    EXCLUDE_GENERATION_ARGS,
    concat_args_str,
    get_parser,
    load_all_generations,
)


def main(args, generation_args):
    # load hidden states and labels
    neg_hs, pos_hs, y = load_all_generations(generation_args)

    # Make sure the shape is correct
    assert neg_hs.shape == pos_hs.shape
    neg_hs, pos_hs = neg_hs[..., -1], pos_hs[..., -1]  # take the last layer
    if neg_hs.shape[1] == 1:  # T5 may have an extra dimension; if so, get rid of it
        neg_hs = neg_hs.squeeze(1)
        pos_hs = pos_hs.squeeze(1)

    # Very simple train/test split (using the fact that the data is already shuffled)
    train_frac = args.train_frac
    num_neg_train = int(train_frac * len(neg_hs))
    num_pos_train = int(train_frac * len(pos_hs))
    num_y_train = int(train_frac * len(y))
    num_train = max(num_neg_train, num_pos_train, num_y_train)
    neg_hs_train, neg_hs_test = neg_hs[:num_train], neg_hs[num_train:]
    pos_hs_train, pos_hs_test = pos_hs[:num_train], pos_hs[num_train:]
    y_train, y_test = y[:num_train], y[num_train:]

    # Make sure logistic regression accuracy is reasonable; otherwise our method won't have much of a chance of working
    # you can also concatenate, but this works fine and is more comparable to CCS inputs
    x_train = neg_hs_train - pos_hs_train
    x_test = neg_hs_test - pos_hs_test
    lr = LogisticRegression(class_weight="balanced")
    lr.fit(x_train, y_train)
    lr_acc = lr.score(x_test, y_test)
    print("Logistic regression accuracy: {}".format(lr_acc))

    # Set up CCS. Note that you can usually just use the default args by simply doing ccs = CCS(neg_hs, pos_hs, y)
    ccs = CCS(neg_hs_train, pos_hs_train, nepochs=args.nepochs, ntries=args.ntries, lr=args.lr, batch_size=args.ccs_batch_size,
                    verbose=args.verbose, device=args.ccs_device, linear=args.linear, weight_decay=args.weight_decay,
                    var_normalize=args.var_normalize)

    # train and evaluate CCS
    ccs.repeated_train()
    ccs_acc = ccs.get_acc(neg_hs_test, pos_hs_test, y_test)
    print("CCS accuracy: {}".format(ccs_acc))

    # Save results.
    ccs_results = {"ccs_acc": ccs_acc, "lr_acc": lr_acc}
    generation_args_str = concat_args_str(generation_args, EXCLUDE_GENERATION_ARGS)
    ccs_args_str = concat_args_str(args, ["verbose"])
    results_dir = os.path.join("ccs_results", generation_args_str)
    if not os.path.exists(results_dir):
        os.mkdir(results_dir)
    results_path = os.path.join(results_dir, f"{ccs_args_str}.json")
    with open(results_path, "w") as f:
        json.dump(ccs_results, f)


if __name__ == "__main__":
    parser, generation_arg_names = get_parser()

    # We'll also add some additional args for evaluation
    ccs_arg_names = ["nepochs", "ntries", "lr", "ccs_batch_size", "verbose", "ccs_device", "linear", "weight_decay", "var_normalize", "train_frac", "seed"]
    ccs_group = parser.add_argument_group("CCS args", description="Arguments for CCS.")
    ccs_group.add_argument("--nepochs", type=int, default=1000)
    ccs_group.add_argument("--ntries", type=int, default=10)
    ccs_group.add_argument("--lr", type=float, default=1e-3)
    ccs_group.add_argument("--ccs_batch_size", type=int, default=-1)
    ccs_group.add_argument("--verbose", action="store_true")
    ccs_group.add_argument("--ccs_device", type=str, default="cuda")
    ccs_group.add_argument("--linear", action="store_true")
    ccs_group.add_argument("--weight_decay", type=float, default=0.01)
    ccs_group.add_argument("--var_normalize", action="store_true")
    ccs_group.add_argument("--train_frac", type=float, default=0.6, help="Fraction of data to use for training")
    ccs_group.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    generation_args = argparse.Namespace(**{arg: getattr(args, arg) for arg in generation_arg_names})
    ccs_args = argparse.Namespace(**{arg: getattr(args, arg) for arg in ccs_arg_names})

    main(ccs_args, generation_args)
