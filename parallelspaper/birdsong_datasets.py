from scipy.io import loadmat
import numpy as np
import textgrid
from glob import glob
from tqdm import tqdm_notebook as tqdm
import pandas as pd
from datetime import datetime
from parallelspaper import information_theory as it 
from parallelspaper import model_fitting as mf
from parallelspaper.quickplots import plot_model_fits

BCOL_DICT = {'Starling':[0.298, 0.447, 0.69 ], 
            'BF': [0.506, 0.447, 0.702],
            'CAVI': [0.769, 0.306, 0.322],
           'CATH': [3/255, 165/255, 150/255]}

def get_tiers(tg_loc):
    grid = textgrid.TextGrid.fromFile(tg_loc)
    tiers_dict = {j.name: i for i, j in enumerate(grid.tiers)}
    return {
        tier_name: [
            tiers_dict[tier_name],
            [
                i.mark
                for i in grid.tiers[tiers_dict[tier_name]].intervals
                if i.mark != ""
            ],
        ]
        for tier_name in tiers_dict.keys()
    }


def prep_CAVI_CATH(all_indvs, isi_thresh=10):
    # grab sequence data from textgrids
    CAVI_isi = []
    CATH_isi = []
    for indv in tqdm(all_indvs):
        species = indv.split("/")[-2]
        if species not in ["CATH", "CAVI"]:
            continue
        indv_textgrids = glob(indv + "/TextGrids/*.TextGrid")
        for textgrid_loc in tqdm(indv_textgrids, leave=False):
            # load textgrid and tiers
            grid = textgrid.TextGrid.fromFile(textgrid_loc)
            isi = np.array(
                [float(i.duration()) for i in grid.tiers[0][1:-1] if i.mark == ""]
            )
            if species == "CAVI":
                CAVI_isi.append(isi)
            if species == "CATH":
                CATH_isi.append(isi)
    CATH_isi = np.concatenate(CATH_isi)
    CAVI_isi = np.concatenate(CAVI_isi)

    # create a formatted song dataframe from seqences
    CATH_syll_lens = []
    CAVI_syll_lens = []
    CATH_grid_lens = []
    CAVI_grid_lens = []
    CAVI_isi = []
    CATH_isi = []

    # rec_num = 0
    song_df = pd.DataFrame(
        columns=[
            "indv",
            "species",
            "bird",
            "syllables",
            "time",
            "tier_num",
            "within_recording_bout",
            "day",
        ]
    )
    for indv in tqdm(all_indvs):
        species = indv.split("/")[-2]
        indv_textgrids = glob(indv + "/TextGrids/*.TextGrid")
        for textgrid_loc in tqdm(indv_textgrids, leave=False):
            # get time
            wav_time = datetime.strptime(
                textgrid_loc.split("/")[-1][:-9], "%Y-%m-%d_%H-%M-%S-%f"
            )
            day = wav_time.strftime("%Y/%m/%d")
            # load textgrid and tiers
            grid = textgrid.TextGrid.fromFile(textgrid_loc)
            tiers_dict = {j.name: i for i, j in enumerate(grid.tiers)}
            tiers = {
                tier_name: [
                    tiers_dict[tier_name],
                    [
                        i.mark
                        for i in grid.tiers[tiers_dict[tier_name]].intervals
                        if i.mark != ""
                    ],
                ]
                for tier_name in tiers_dict.keys()
                if type(grid.tiers[tiers_dict[tier_name]])
                != textgrid.textgrid.PointTier
            }
            # get statistics on textgrid
            # grid_len = float(grid.tiers[0][-1].maxTime - grid.tiers[0][0].minTime)
            # syll_lens = [float(i.duration()) for i in grid.tiers[0] if i.mark!='']

            for tier in tiers.keys():

                tier_num = tiers[tier][0]
                # skip if too short
                if len(grid.tiers[tier_num]) < 3:
                    continue

                isi = np.array(
                    [
                        float(i.duration())
                        for i in grid.tiers[tier_num][1:-1]
                        if i.mark == ""
                    ]
                )

                intervals = grid.tiers[tier_num][1:-1]
                isi_over_thresh = np.where(isi > isi_thresh)

                grid_marks = np.array([i.mark for i in grid.tiers[tier_num][1:-1]])
                break_locs = np.where([grid_marks == ""])[1]
                split_points = np.concatenate(
                    [break_locs[isi_over_thresh], [len(intervals)]]
                )

                sp_last = 0
                for spn, sp in enumerate(split_points):
                    marks = grid_marks[sp_last:sp]
                    mask = marks != ""
                    if np.sum(mask) == 0:
                        continue
                    voc_interval = np.array(intervals[sp_last:sp])[mask]
                    song_df.loc[len(song_df)] = [
                        indv.split("/")[-1],
                        species,
                        tier,
                        marks[mask],
                        wav_time,
                        tiers[tier][0],
                        spn,
                        day,
                    ]
                    voc_dur = [float(i.duration()) for i in voc_interval]

                    if tier_num == 0:
                        if species == "CAVI":
                            CAVI_syll_lens.append(voc_dur)
                            if sp == len(intervals):
                                CAVI_grid_lens.append(
                                    float(
                                        intervals[sp - 1].maxTime
                                        - intervals[sp_last].minTime
                                    )
                                )
                            else:
                                CAVI_grid_lens.append(
                                    float(
                                        intervals[sp].minTime
                                        - intervals[sp_last].minTime
                                    )
                                )
                            CAVI_isi.append(isi)
                        elif species == "CATH":
                            CATH_syll_lens.append(voc_dur)
                            if sp == len(intervals):
                                CATH_grid_lens.append(
                                    float(
                                        intervals[sp - 1].maxTime
                                        - intervals[sp_last].minTime
                                    )
                                )
                            else:
                                CATH_grid_lens.append(
                                    float(
                                        intervals[sp].minTime
                                        - intervals[sp_last].minTime
                                    )
                                )
                            CATH_isi.append(isi)

                    sp_last = sp

    # subset only the birds we're interested in
    song_df = song_df[
        (song_df.species.values == "CATH") | (song_df.species.values == "CAVI")
    ]
    song_df = song_df[(song_df.tier_num == 0)]

    # index recording numbers
    song_df["rec_num"] = None
    for bird in np.unique(song_df.indv):
        song_idxs = (
            song_df[song_df.indv.values == bird]
            .sort_values(by=["time", "within_recording_bout"])
            .index
        )
        for idxi, idx in enumerate(song_idxs):
            song_df.set_value(idx, "rec_num", idxi)
    # rename column
    song_df["bird"] = song_df["indv"]

    return (
        song_df,
        CATH_isi,
        CAVI_isi,
        CATH_syll_lens,
        CAVI_syll_lens,
        CATH_grid_lens,
        CAVI_grid_lens,
        CAVI_isi,
        CATH_isi,
    )


def prep_STARLING(sequence_dfs, isi_thresh=10):
    # get the most recent df for each bird
    sdf_table = pd.DataFrame(
        [
            [
                sdf.split("/")[-3],
                datetime.strptime(sdf.split("/")[-2], "%Y-%m-%d_%H-%M-%S"),
                sdf,
            ]
            for sdf in sequence_dfs
        ],
        columns=["bird", "dt", "sdf"],
    ).sort_values(by="dt")
    seq_dfs = []
    for bird in np.unique(sdf_table.bird):
        birdname = bird.split("_")[-2]
        if birdname[-5:] == ".hdf5":
            continue
        sdf_early = np.argsort(sdf_table[sdf_table.bird == bird].dt).values[-1]
        sdf = sdf_table[sdf_table.bird == bird].sdf.values[sdf_early]
        species = sdf.split("/")[-4]
        seq_df = pd.read_pickle(sdf)
        seq_df["bird_name"] = birdname
        seq_df["species"] = species
        seq_dfs.append(seq_df)
        print(birdname, species, len(seq_df))
    seq_dfs = pd.concat(seq_dfs)
    seq_dfs = seq_dfs.sort_values(by=["bird_name", "syllable_time"])
    syllable_duration_s = seq_df.syll_length_s.values

    # produce song df
    song_df = pd.DataFrame(columns=["bird", "species", "syllables", "rec_num", "day"])
    sequences = []
    seq_lens = []
    ISIs = []
    last_syll_len = 0
    seq_end_time = None
    for bird in tqdm(np.unique(seq_dfs.bird_name)):
        bird_df = seq_dfs[seq_dfs.bird_name == bird]
        seq = [bird_df.labels.values[0]]
        seq_start_time = bird_df.syllable_time.values[0]
        seq_date = str(seq_start_time).split("T")[0]
        last_syll_time = bird_df.syllable_time.values[0]
        song_idx = 0

        for idx, row in tqdm(
            bird_df[1:].iterrows(), total=len(bird_df) - 1, leave=False
        ):
            if (row.syllable_time - last_syll_time).total_seconds() > isi_thresh:
                sequences.append(seq)
                seq = [row.labels]
                song_df.loc[len(song_df)] = [bird, "Starling", seq, song_idx, seq_date]
                song_idx += 1
                seq_lens.append((seq_end_time - seq_start_time).total_seconds())
                seq_start_time = row.syllable_time
                seq_date = str(seq_start_time).split(" ")[0]
            else:
                ISIs.append(
                    (row.syllable_time - last_syll_time).total_seconds() - last_syll_len
                )
                seq.append(row.labels)
                seq_end_time = row.syllable_time
            last_syll_time = row.syllable_time
            last_syll_len = row.syll_length_s

    return song_df, seq_lens, syllable_duration_s, ISIs


def prep_BF(label_locs):

    # get dataa from mat file
    bf_isi = []
    song_df = pd.DataFrame(
        columns=[
            "bird",
            "species",
            "stime",
            "syllables",
            "bout_duration",
            "syll_lens",
            "day",
        ]
    )
    for label_loc in tqdm(label_locs):
        mat = loadmat(label_loc)
        loc_time = datetime.strptime(
            "_".join(label_loc.split("/")[-1].split(".")[0].split("_")[-2:]),
            "%d%m%y_%H%M",
        )
        indv = label_loc.split("/")[-1].split("_")[0]
        syll_lens = np.squeeze(mat["offsets"] - mat["onsets"]) / 1000
        # np.array(mat['durations']).flatten()/1000
        labels = list(np.array(mat["labels"]).flatten()[0])
        # if len(labels) > 300: break
        bout_duration = (mat["offsets"][-1][0] - mat["onsets"][0][0]) / 1000
        # mat['bout_duration']/1000
        song_df.loc[len(song_df)] = [
            indv,
            "BF",
            loc_time,
            labels,
            bout_duration,
            syll_lens,
            loc_time.strftime("%d/%m/%y"),
        ]
        if "pauses" in mat.keys():
            bf_isi.append(np.concatenate(mat["pauses"] / 1000))
    bf_isi = np.array([item for sublist in bf_isi for item in sublist])

    song_df["NumNote"] = [len(i) for i in song_df.syllables.values]

    # relabel recording numbers
    song_df["rec_num"] = None
    song_df = song_df.reset_index()
    for bird in np.unique(song_df.bird):
        song_idxs = song_df[song_df.bird.values == bird].sort_values(by="stime").index
        for idxi, idx in enumerate(song_idxs):
            song_df.set_value(idx, "rec_num", idxi)

    # label day
    song_df["day"] = [
        pd.to_datetime(str(i)).strftime("%Y-%m-%d") for i in song_df.stime.values
    ]

    # split by day
    all_days = []
    rn = 0
    for bird in tqdm(np.unique(song_df.bird.values)):
        bird_df = song_df[song_df.bird == bird]
        for day in np.unique(bird_df.day):
            day_df = bird_df[bird_df.day == day].sort_values(by="stime")
            duration = np.sum(day_df.bout_duration.values)
            sylls = list(np.concatenate(day_df.syllables.values))
            day_full = pd.DataFrame(
                [
                    [
                        day_df.bird.values[0],
                        day_df.species.values[0],
                        sylls,
                        duration,
                        day,
                        rn,
                    ]
                ],
                columns=["bird", "species", "syllables", "duration", "day", "rec_num"],
            )
            all_days.append(day_full)
            rn += 1

    day_df = pd.concat(all_days)

    return song_df, bf_isi


def compress_seq(seq):
    cs = [seq[0]]
    for i in (seq[1:]):
        if cs[-1] != i:
            cs.append(i)
    return cs

def MI_seqs(seqs, distances, species, type_="day", n_jobs=20, verbosity=0, nrep=1, indv=None, verbose=True):
    (MI, var_MI), (MI_shuff, MI_shuff_var) = it.sequential_mutual_information(
        seqs,
        distances,
        n_jobs=n_jobs,
        verbosity=verbosity,
        n_shuff_repeats=nrep,
        estimate=True,
    )
    sig = MI - MI_shuff
    results_power, results_exp, results_pow_exp, best_fit_model = mf.fit_models(
        distances, sig
    )
    R2_exp, R2_concat, R2_power, AICc_exp, AICc_pow, AICc_concat = mf.fit_results(
        sig, distances, results_exp, results_power, results_pow_exp
    )
    if verbose: 
        plot_model_fits(
            MI, MI_shuff, distances, results_power, results_exp, results_pow_exp
        )
    r_list = [
        species,
        type_,
        1,
        MI,
        MI_shuff,
        distances,
        var_MI,
        MI_shuff_var,
        len(np.concatenate(seqs)),
        results_exp,
        results_power,
        results_pow_exp,
        R2_exp,
        R2_concat,
        R2_power,
        AICc_exp,
        AICc_concat,
        AICc_pow,
        best_fit_model,
    ]
    if indv is not None:
        r_list = [indv] + r_list
    return r_list