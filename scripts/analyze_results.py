import argparse
import os
import itertools
import pandas as pd
import numpy as np
import re


def truncate_at_key(df, key, limit):
    return df[df[key] <= limit]


def exclude_worst_performing_by_metric(
    dfs, metric, n_exclude, rolling=50, descending=False
):
    best_rolling_max = [
        (
            k,
            getattr(
                dfs[k][metric].dropna().rolling(rolling).mean().fillna(0),
                "max" if not descending else "min",
            )(),
        )
        for k in range(len(dfs))
    ]
    sort_keys = sorted(
        best_rolling_max, key=lambda k: k[1] if not descending else -1 * k[1]
    )
    print(sort_keys)

    return [dfs[k[0]] for k in sort_keys[n_exclude:]]


def get_top_values_for_corresponding_value(
    name, dfs, corresponding, values, rolling=1, descending=False
):
    select_cols = values + ([corresponding] if corresponding not in values else [])

    nonrolling_dfs = [
        df[select_cols].dropna() for df in dfs
        if all([s in df.columns for s in select_cols])
    ]

    rolling_dfs = [
        df[select_cols].rolling(rolling).mean().fillna(0) for df in nonrolling_dfs
    ]

    argwheres = [
        np.argwhere(
            rolling_df[corresponding].values
            == getattr(rolling_df[corresponding], "max" if not descending else "min")()
        )[-1][0]
        for rolling_df in rolling_dfs
    ]
    print(name, list(zip(argwheres, map(lambda df: df.shape[0], rolling_dfs))))

    values_df = pd.DataFrame(
        np.stack(
            [
                nonrolling_df[values].iloc[argwhere].values
                for argwhere, nonrolling_df in zip(argwheres, nonrolling_dfs)
            ]
        ),
        columns=values,
    )
    print(name)
    print(values_df)
    return values_df


def format_experiment_name(experiment_config, params):
    name_dict = {**experiment_config, **params}
    return "_".join(
        map(
            str,
            [
                name_dict["headline"],
                "s",
                name_dict["seed"],
                "m",
                name_dict["model"],
                "l",
                name_dict["layers"],
                "h",
                name_dict["heads"],
                "d",
                name_dict["hidden"],
                "it",
                name_dict["iterations"],
                "b",
                name_dict["batch_size"],
                "d",
                name_dict["dataset"],
                "t",
                name_dict["tag"],
                "drop",
                name_dict["dropout"],
            ]
            + (
                ["ml_d_limit", name_dict["ml_d_limit"]]
                if "ml_d_limit" in name_dict and name_dict["ml_d_limit"] is not None
                else []
            ),
        )
    )


def format_model_name(experiment_config, include_hparams=True):
    return "_".join(
        map(
            str,
            [experiment_config["model"]]
            + (
                [
                    "l",
                    experiment_config["layers"],
                    "h",
                    experiment_config["heads"],
                    "d",
                    experiment_config["hidden"],
                ]
                if include_hparams
                else []
            ),
        )
    )


def format_log_path(logs_dir, experiment_config, params, model_include_hparams=True):
    # We have to do some testing of paths here, which doesn't scale all
    # that well, but its fine for a small number of paths
    base_path = os.path.join(
        format_experiment_name(experiment_config, params),
        format_model_name(experiment_config, include_hparams=model_include_hparams),
        experiment_config["dataset"],
        str(params["seed"]),
        "lightning_logs",
    )

    if os.path.exists(os.path.join(logs_dir, base_path, "100", "metrics.csv")):
        return os.path.join(base_path, "100", "metrics.csv")

    return os.path.join(base_path, "version_100", "metrics.csv")


def read_csv_and_truncate(path, truncate_key, truncate_value):
    try:
        df = pd.read_csv(path)
        print(f"Read csv {path} - {df.shape} values")
    except Exception as e:
        print(f"Could not read csv {path} - {e}")
        return None
    truncated = truncate_at_key(df, truncate_key, truncate_value)

    return truncated


TEST_SPLIT_DATALOADER_MAPPINGS = {
    "gscan": {
        "vexact/dataloader_idx_0": "A",
        "vexact/dataloader_idx_1": "B",
        "vexact/dataloader_idx_2": "C",
        "vexact/dataloader_idx_3": "D",
        "vexact/dataloader_idx_5": "E",
        "vexact/dataloader_idx_6": "F",
        "vexact/dataloader_idx_7": "G",
        "vexact/dataloader_idx_8": "H",
    },
    "sr": {
        "vexact/dataloader_idx_0": "Test",
        "vexact/dataloader_idx_2": "II",
        "vexact/dataloader_idx_9": "III",
        "vexact/dataloader_idx_10": "IV",
        "vexact/dataloader_idx_11": "V",
        "vexact/dataloader_idx_12": "VI",
    },
    "reascan": {
        "vexact/dataloader_idx_0": "Test",
        "vexact/dataloader_idx_1": "A1",
        "vexact/dataloader_idx_2": "A2",
        "vexact/dataloader_idx_3": "B1",
        "vexact/dataloader_idx_4": "B2",
        "vexact/dataloader_idx_5": "C1",
        "vexact/dataloader_idx_6": "C2",
    },
}

_RE_DIRECTORY_NAME = r"(?P<headline>[a-z_]+)_s_(?P<seed>[0-9])_m_(?P<model>[0-9a-z_]+)_l_(?P<layers>[0-9]+)_h_(?P<heads>[0-9]+)_d_(?P<hidden>[0-9]+)_it_(?P<iterations>[0-9]+)_b_(?P<batch_size>[0-9]+)_d_(?P<dataset>[0-9a-z_]+)_t_(?P<tag>[a-z_0-9]+)_drop_(?P<dropout>[0-9\.]+)(?:_ml_d_limit_(?P<ml_d_limit>[0-9]+))?"


def check_matches_or_warn(regex, string):
    if not re.match(regex, string):
        print(f"String {string} doesn't match regex {regex}")
        return False

    return True


def collate_func_key(x, keys):
    return tuple([x[k] for k in keys if k != "seed"])


def read_and_collate_from_directory(
    logs_dir, limit, filter_expression=None, exclude_seeds=[]
):
    listing = os.listdir(logs_dir)
    parsed_listing = list(
        map(
            lambda x: re.match(_RE_DIRECTORY_NAME, x).groupdict(),
            filter(
                lambda x: (
                    check_matches_or_warn(_RE_DIRECTORY_NAME, x) and (
                        filter_expression is None or
                        re.match(filter_expression, x) is not None
                    )
                ),
                listing,
            )
        )
    )
    keys = sorted(parsed_listing[0].keys())
    keys_not_seed = [k for k in keys if k != "seed"]
    grouped_listing_indices = [
        (
            {k: v for k, v in zip(keys_not_seed, key_tuple)},
            list(
                map(
                    lambda index: (index, parsed_listing[index]["seed"]),
                    list(zip(*group))[0],
                )
            ),
        )
        for key_tuple, group in itertools.groupby(
            sorted(
                list(enumerate(parsed_listing)),
                key=lambda x: collate_func_key(x[1], keys_not_seed),
            ),
            lambda x: collate_func_key(x[1], keys_not_seed),
        )
    ]

    detected_config_and_csv_tuples = [
        (
            config,
            list(
                filter(
                    lambda x: x[1] is not None,
                    [
                        (
                            seed,
                            format_log_path(logs_dir, config, {"seed": seed}),
                        )
                        for index, seed in values
                        if seed not in exclude_seeds
                        and os.path.exists(
                            os.path.join(
                                logs_dir,
                                format_log_path(logs_dir, config, {"seed": seed}),
                            )
                        )
                    ],
                )
            ),
        )
        for config, values in grouped_listing_indices
    ]

    return list(filter(lambda x: x[1], detected_config_and_csv_tuples))


def list_and_collate_from_directory(
    logs_dir, filter_expression=None, exclude_seeds=[]
):
    listing = os.listdir(logs_dir)
    parsed_listing = list(
        map(
            lambda x: re.match(_RE_DIRECTORY_NAME, x).groupdict(),
            filter(
                lambda x: (
                    check_matches_or_warn(_RE_DIRECTORY_NAME, x) and (
                        filter_expression is None or
                        re.match(filter_expression, x) is not None
                    )
                ),
                listing,
            )
        )
    )
    keys = sorted(parsed_listing[0].keys())
    keys_not_seed = [k for k in keys if k != "seed"]
    grouped_listing_indices = [
        (
            {k: v for k, v in zip(keys_not_seed, key_tuple)},
            list(
                map(
                    lambda index: (index, parsed_listing[index]["seed"]),
                    list(zip(*group))[0],
                )
            ),
        )
        for key_tuple, group in itertools.groupby(
            sorted(
                list(enumerate(parsed_listing)),
                key=lambda x: collate_func_key(x[1], keys_not_seed),
            ),
            lambda x: collate_func_key(x[1], keys_not_seed),
        )
    ]

    detected_config_and_csv_tuples = [
        (
            config,
            list(
                filter(
                    lambda x: x[1] is not None,
                    [
                        (
                            seed,
                            format_log_path(logs_dir, config, {"seed": seed}),
                        )
                        for index, seed in values
                        if seed not in exclude_seeds
                        and os.path.exists(
                            os.path.join(
                                logs_dir,
                                format_log_path(logs_dir, config, {"seed": seed}),
                            )
                        )
                    ],
                )
            ),
        )
        for config, values in grouped_listing_indices
    ]

    return list(filter(lambda x: x[1], detected_config_and_csv_tuples))


def read_matched_configs_at_limit(logs_dir, matched_configs, limit):
    return list(
        map(
            lambda x: (
                x[0],
                x[1],
                list(
                    filter(
                        lambda y: y[-1] is not None,
                        map(
                            lambda y: (y[0], read_csv_and_truncate(
                                os.path.join(logs_dir, y[-1]),
                                "step",
                                limit,
                            )),
                            x[-1]
                        )
                    )
                )
            ),
            matched_configs
        )
    )



MATCH_CONFIGS = {
    "baseline_vilbert": {
        "model": "vilbert_cross_encoder_decode_actions",
        "dataset": "baseline",
        "headline": "gscan",
        "tag": "only_ca",
    },
    "baseline_transformer": {
        "model": "transformer_encoder_only_decode_actions",
        "dataset": "baseline",
        "headline": "gscan",
    },
    "baseline_transformer_i2g_seq2seq": {
        "model": "transformer_encoder_only_decode_actions",
        "dataset": "i2g_seq2seq_model_score",
        "headline": "gscan",
        "tag": "include_demos"
    },
    "baseline_roformer": {
        "model": "roformer_encoder_only_decode_actions",
        "dataset": "baseline",
        "headline": "gscan",
    },
    "baseline_universal_transformer": {
        "model": "universal_transformer_encoder_only_decode_actions",
        "dataset": "baseline",
        "headline": "gscan",
    },
    "baseline_transformer_img": {
        "model": "transformer_img_encoder_only_decode_actions",
        "dataset": "baseline",
        "headline": "gscan",
    },
    "baseline_pp_vilbert": {
        "model": "vilbert_cross_encoder_decode_actions",
        "dataset": "baseline_paraphrased",
        "headline": "gscan",
        "tag": "only_ca",
    },
    "baseline_pp_transformer": {
        "model": "transformer_encoder_only_decode_actions",
        "dataset": "baseline_paraphrased",
        "headline": "gscan",
    },
    "baseline_pp_transformer_img": {
        "model": "transformer_img_encoder_only_decode_actions",
        "dataset": "baseline_paraphrased",
        "headline": "gscan",
    },
    "baseline_sr_vilbert": {
        "model": "vilbert_cross_encoder_decode_actions",
        "dataset": "baseline_sr",
        "headline": "gscan",
        "tag": "only_ca",
    },
    "baseline_sr_transformer": {
        "model": "transformer_encoder_only_decode_actions",
        "dataset": "baseline_sr",
        "headline": "gscan",
    },
    "baseline_sr_pp_vilbert": {
        "model": "vilbert_cross_encoder_decode_actions",
        "dataset": "baseline_sr_paraphrased",
        "headline": "gscan",
        "tag": "only_ca",
    },
    "baseline_sr_pp_transformer": {
        "model": "transformer_encoder_only_decode_actions",
        "dataset": "baseline_sr_paraphrased",
        "headline": "gscan",
    },
    "baseline_sr_i2g_seq2seq_ft": {
        "model": "transformer_encoder_only_decode_actions",
        "dataset": "i2g_seq2seq_model_score_baseline_sr",
        "headline": "gscan",
        "tag": "include_demos"
    },
    "baseline_reascan_vilbert": {
        "model": "vilbert_cross_encoder_decode_actions",
        "dataset": "baseline_reascan",
        "headline": "gscan",
        "tag": "only_ca",
    },
    "baseline_reascan_transformer": {
        "model": "transformer_encoder_only_decode_actions",
        "dataset": "baseline_reascan",
        "headline": "gscan",
    },
    "baseline_reascan_pp_vilbert": {
        "model": "vilbert_cross_encoder_decode_actions",
        "dataset": "baseline_reascan_paraphrased",
        "headline": "gscan",
        "tag": "only_ca",
    },
    "baseline_reascan_pp_transformer": {
        "model": "transformer_encoder_only_decode_actions",
        "dataset": "baseline_reascan_paraphrased",
        "headline": "gscan",
    },
    "baseline_reascan_i2g_seq2seq_ft": {
        "model": "transformer_encoder_only_decode_actions",
        "dataset": "i2g_seq2seq_model_score_baseline_reascan",
        "headline": "gscan",
        "tag": "include_demos"
    },
    "babyai_gscan_comb_gotolocal_transformer": {
        "model": "transformer_encoder_only_decode_actions",
        "dataset": "babyai_gscan_comb_gotolocal",
        "headline": "gscan",
    },
    "transformer_full": {
        "model": "vilbert_cross_encoder_decode_actions",
        "headline": "gscan",
    },
    "i2g": {"dataset": "i2g", "headline": "meta_gscan"},
    "i2g_seq2seq_big_transformer": {
        "dataset": "i2g_seq2seq_model_score",
        "model": "meta_symbol_encdec_big_transformer",
        "headline": "meta_gscan",
        "tag": "none",
        "dropout": "0.1",
        "ml_d_limit": "16",
    },
    "i2g_seq2seq_big_transformer_noshuffle": {
        "dataset": "i2g_seq2seq_model_score",
        "model": "meta_symbol_encdec_big_transformer",
        "headline": "meta_gscan",
        "tag": "noshuffle",
        "dropout": "0.1",
        "ml_d_limit": "16",
    },
    "i2g_seq2seq_big_transformer_unscored": {
        "dataset": "i2g_seq2seq_unscored",
        "model": "meta_symbol_encdec_big_transformer",
        "headline": "meta_gscan",
        "dropout": "0.1",
        "ml_d_limit": "16",
        "tag": "no_reorder_flag"
    },
    "i2g_seq2seq_big_transformer_img": {
        "dataset": "i2g_seq2seq_model_score",
        "model": "meta_img_encdec_big_transformer",
        "headline": "meta_gscan",
        "tag": "none",
        "dropout": "0.1",
        "ml_d_limit": "16",
    },
    "i2g_seq2seq_big_transformer_pp_8": {
        "dataset": "i2g_seq2seq_paraphrased",
        "model": "meta_encdec_big_transformer",
        "headline": "meta_gscan",
        "tag": "none",
        "dropout": "0.1",
        "ml_d_limit": "8",
    },
    "i2g_seq2seq_big_transformer_pp_16": {
        "dataset": "i2g_seq2seq_paraphrased",
        "model": "meta_symbol_encdec_big_transformer",
        "headline": "meta_gscan",
        "tag": "none",
        "dropout": "0.1",
        "ml_d_limit": "16",
    },
    "i2g_seq2seq_big_transformer_pp_24": {
        "dataset": "i2g_seq2seq_paraphrased",
        "model": "meta_encdec_big_transformer",
        "headline": "meta_gscan",
        "tag": "none",
        "dropout": "0.1",
        "ml_d_limit": "24",
    },
    "i2g_seq2seq_big_transformer_pp_16_img": {
        "dataset": "i2g_seq2seq_paraphrased",
        "model": "meta_img_encdec_big_transformer",
        "headline": "meta_gscan",
        "tag": "none",
        "dropout": "0.1",
        "ml_d_limit": "16",
    },
    "baseline_reascan_i2g_seq2seq": {
        "dataset": "i2g_seq2seq_model_score_baseline_reascan",
        "model": "meta_symbol_encdec_big_transformer",
        "headline": "meta_gscan",
        "tag": "none",
        "dropout": "0.1",
        "ml_d_limit": "16"
    },
    "baseline_reascan_i2g_seq2seq_noshuffle": {
        "dataset": "i2g_seq2seq_model_score_baseline_reascan",
        "model": "meta_symbol_encdec_big_transformer",
        "headline": "meta_gscan",
        "tag": "noshuffle",
        "dropout": "0.1",
        "ml_d_limit": "16",
    },
    "baseline_reascan_i2g_seq2seq_unscored": {
        "dataset": "i2g_seq2seq_unscored_baseline_reascan",
        "model": "meta_symbol_encdec_big_transformer",
        "headline": "meta_gscan",
        "dropout": "0.1",
        "ml_d_limit": "16"
    },
    "baseline_reascan_gandr": {
        "dataset": "gandr_baseline_reascan",
        "model": "meta_symbol_encdec_big_transformer",
        "headline": "meta_gscan",
        "tag": "none",
        "dropout": "0.1",
        "ml_d_limit": "16"
    },
    "baseline_sr_i2g_seq2seq": {
        "dataset": "i2g_seq2seq_model_score_baseline_sr",
        "model": "meta_symbol_encdec_big_transformer",
        "headline": "meta_gscan",
        "tag": "none",
        "dropout": "0.1",
        "ml_d_limit": "16"
    },
    "baseline_sr_i2g_seq2seq_noshuffle": {
        "dataset": "i2g_seq2seq_model_score_baseline_sr",
        "model": "meta_symbol_encdec_big_transformer",
        "headline": "meta_gscan",
        "tag": "noshuffle",
        "dropout": "0.1",
        "ml_d_limit": "16",
    },
    "baseline_sr_i2g_seq2seq_unscored": {
        "dataset": "i2g_seq2seq_unscored_baseline_sr",
        "model": "meta_symbol_encdec_big_transformer",
        "headline": "meta_gscan",
        "dropout": "0.1",
        "ml_d_limit": "16",
    },
    "baseline_sr_gandr": {
        "dataset": "gandr_baseline_sr",
        "model": "meta_symbol_encdec_big_transformer",
        "headline": "meta_gscan",
        "tag": "none",
        "dropout": "0.1",
        "ml_d_limit": "16"
    },
    "gandr": {"dataset": "gandr", "headline": "meta_gscan"},
    "gandr_coverage": {"dataset": "gandr_coverage", "headline": "meta_gscan"},
    "gandr_coverage_img": {
        "dataset": "gandr_coverage_fixed",
        "model": "meta_img_encdec_big_transformer",
        "headline": "meta_gscan",
        "tag": "none",
        "dropout": "0.1",
        "ml_d_limit": "16",
    },
    "gandr_coverage_pp_img": {
        "dataset": "gandr_coverage_paraphrased_fixed",
        "model": "meta_img_encdec_big_transformer",
        "headline": "meta_gscan",
        "tag": "none",
        "dropout": "0.1",
        "ml_d_limit": "16",
    },
    "gandr_coverage_pp": {"dataset": "gandr_coverage_paraphrased", "headline": "meta_gscan"},
    "gscan_oracle_full": {"dataset": "metalearn_allow_any", "headline": "meta_gscan"},
    "coverage_retrieval": {
        "dataset": "coverage_retrieval",
        "model": "meta_symbol_encdec_big_transformer",
        "headline": "meta_gscan",
        "tag": "none",
        "dropout": "0.1",
        "ml_d_limit": "16",
    },
    "coverage_retrieval_img": {
        "dataset": "coverage_retrieval",
        "model": "meta_img_encdec_big_transformer",
        "headline": "meta_gscan",
        "tag": "none",
        "dropout": "0.1",
        "ml_d_limit": "16",
    },
    "coverage_retrieval_pp_img": {
        "dataset": "coverage_retrieval_paraphrased",
        "model": "meta_img_encdec_big_transformer",
        "headline": "meta_gscan",
        "tag": "none",
        "dropout": "0.1",
        "ml_d_limit": "16",
    },
    "coverage_retrieval_pp": {
        "dataset": "coverage_retrieval_paraphrased",
        "model": "meta_symbol_encdec_big_transformer",
        "headline": "meta_gscan",
        "tag": "none",
        "dropout": "0.1",
        "ml_d_limit": "16",
    },
    "retrieval_gscan_sr": {
        "dataset": "baseline_sr_retrieval",
        "model": "meta_symbol_encdec_big_transformer",
        "headline": "meta_gscan",
        "tag": "none",
        "dropout": "0.1",
        "ml_d_limit": "16",
    },
    "retrieval_reascan": {
        "dataset": "baseline_reascan_retrieval",
        "model": "meta_symbol_encdec_big_transformer",
        "headline": "meta_gscan",
        "tag": "none",
        "dropout": "0.1",
        "ml_d_limit": "16",
    },
    "gscan_metalearn_only_random": {
        "dataset": "metalearn_random_instructions_same_layout_allow_any",
        "headline": "meta_gscan",
    },
    "gscan_metalearn_sample_environments": {
        "dataset": "metalearn_find_matching_instruction_demos_allow_any",
        "headline": "meta_gscan",
    },
    "retrieval_babyai_comb_gotolocal": {
        "dataset": "babyai_gscan_comb_gotolocal_retrieval",
        "model": "meta_symbol_encdec_big_transformer",
        "headline": "meta_gscan",
        "tag": "none",
        "dropout": "0.1",
        "ml_d_limit": "16",
    },
}


def zeroth_or_none(list_object):
    if len(list_object):
        return list_object[0]

    return None


def match_to_configs(configs, configs_and_results_tuples):
    return list(
        filter(
            lambda x: x is not None,
            [
                zeroth_or_none(
                    [
                        (name, config, results)
                        for config, results in configs_and_results_tuples
                        if all(
                            [
                                config[k] == requested_config[k]
                                for k in requested_config.keys()
                            ]
                        )
                    ]
                )
                for name, requested_config in configs.items()
            ],
        )
    )


def format_results_table(metrics, configs, index, column_names=None):
    results_table = (
        pd.concat(
            [metrics[k].T["mean"] for k in configs],
            axis=1,
        )
        .T.round(2)
        .astype(str)
        .reset_index()
        + " ± "
        + (
            pd.concat(
                [metrics[k].T["std"] for k in configs],
                axis=1,
            )
            .T.round(2)
            .astype(str)
            .reset_index()
        )
    ).T
    results_table.index = ["index"] + index

    if column_names is not None:
        results_table.columns = column_names
    else:
        results_table.columns = [k for k in configs]

    return results_table.to_latex(float_format="%.2f", escape=False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--logs-dir", required=True)
    parser.add_argument("--limit", type=int, default=300000)
    parser.add_argument("--drop-bad-seeds", type=int, default=0)
    parser.add_argument("--exclude-by-a-smoothing", type=int, default=1)
    parser.add_argument("--result-smoothing", type=int, default=1)
    parser.add_argument("--exclude-seeds", nargs="*")
    parser.add_argument("--dataset", choices=("gscan", "reascan", "sr"), required=True)
    parser.add_argument(
        "--filter-expression",
        type=str,
        help="Regular expression used to filter matches",
    )
    parser.add_argument("--config-columns", nargs="+", choices=MATCH_CONFIGS.keys())
    parser.add_argument("--column-labels", nargs="*")
    args = parser.parse_args()

    metrics_dfs_filenames_by_config = list_and_collate_from_directory(
        args.logs_dir,
        filter_expression=args.filter_expression,
        exclude_seeds=args.exclude_seeds or []
    )

    metrics_dfs_filenames_by_config = list(
        filter(
            lambda x: x[0] in args.config_columns,
            match_to_configs(MATCH_CONFIGS, metrics_dfs_filenames_by_config)
        )
    )

    read_metrics_dfs_at_limit_by_config = read_matched_configs_at_limit(
        args.logs_dir,
        metrics_dfs_filenames_by_config,
        args.limit
    )

    read_metrics_dfs_excluded = [
        (
            name,
            config,
            exclude_worst_performing_by_metric(
                # Take the df from the seed/df pair
                list(zip(*read_metrics_df_at_limit_and_seeds))[1],
                "vexact/dataloader_idx_0",
                args.drop_bad_seeds,
                args.exclude_by_a_smoothing,
                descending=False,
            ),
        )
        for name, config, read_metrics_df_at_limit_and_seeds in read_metrics_dfs_at_limit_by_config
    ]

    read_metrics_dfs_best_at_0 = [
        (
            name,
            config,
            get_top_values_for_corresponding_value(
                name,
                read_metrics_df_excluded,
                "vexact/dataloader_idx_0",
                list(TEST_SPLIT_DATALOADER_MAPPINGS[args.dataset].keys()),
                args.result_smoothing,
                descending=False,
            ).describe(),
        )
        for name, config, read_metrics_df_excluded in read_metrics_dfs_excluded
    ]

    read_metrics_dict = {
        name: metrics for name, config, metrics in read_metrics_dfs_best_at_0
    }

    print(
        format_results_table(
            read_metrics_dict,
            args.config_columns,
            column_names=args.column_labels,
            index=list(TEST_SPLIT_DATALOADER_MAPPINGS[args.dataset].values()),
        )
    )


if __name__ == "__main__":
    main()
