from tqdm.autonotebook import tqdm
import xml.etree.ElementTree
import buckeye
from buckeye.utterance import words_to_utterances
import textgrid

LCOL_DICT = {
    "english": [0.333, 0.659, 0.408],
    "german": "#721825",
    "italian": [0.867, 0.518, 0.322],
    "japanese": "#03585D",
}


def flatlist(list_of_lists):
    return [val for sublist in list_of_lists for val in sublist]


def prep_CSJ(xml_locs):
    IPU_lens = []
    phone_lens = []
    word_lens = []
    session_lens = []

    words = []
    pos = []
    mora = []
    phonemes = []
    phones = []
    phone_class = []

    all_IPU_phonemes = []

    for loc in tqdm(xml_locs):
        session_IPU_lens = []
        speaker_xml = xml.etree.ElementTree.parse(loc).getroot()

        speaker_words = []
        speaker_pos = []
        speaker_mora = []
        speaker_phonemes = []
        speaker_phones = []
        speaker_phoneclass = []
        speaker_IPU_phonemes = []

        # inter pausal unit (more than 200ms between)
        for IPU in tqdm(speaker_xml.getchildren(), leave=False):
            IPU_phonemes = []
            if IPU.tag != "IPU":
                if IPU.tag not in ["TalkComment"]:
                    print("IPU TAG: ", IPU.tag)
                continue
            IPU_len = float(IPU.attrib["IPUEndTime"]) - float(
                IPU.attrib["IPUStartTime"]
            )
            IPU_lens.append(IPU_len)
            session_IPU_lens.append(IPU_len)
            for LUW in IPU.getchildren():
                if LUW.tag != "LUW":
                    if LUW.tag not in ["LineComment"]:
                        print("LUW TAG: ", LUW.tag)
                    continue
                for SUW in LUW.getchildren():
                    if SUW.tag not in ["SUW"]:
                        if SUW.tag not in ["Noise", "LineComment"]:
                            print("SUW TAG: ", SUW.tag)
                        continue

                    speaker_words.append(SUW.attrib["PlainOrthographicTranscription"])
                    word_lens.append(
                        len(
                            flatlist(
                                flatlist(
                                    [
                                        [
                                            [
                                                phn
                                                for phn in MORA.getchildren()
                                                if phn.tag == "Phoneme"
                                            ]
                                            for MORA in transSUW.getchildren()
                                        ]
                                        for transSUW in SUW.getchildren()
                                    ]
                                )
                            )
                        )
                    )

                    if "SUWPOS" in SUW.attrib.keys():
                        speaker_pos.append(SUW.attrib["SUWPOS"])

                    # go through words
                    for TransSUW in SUW.getchildren():

                        word_phones = []
                        word_mora = []
                        word_phonemes = []
                        word_phoneclass = []

                        if TransSUW.tag != "TransSUW":
                            print("TransSUW TAG: ", TransSUW.tag)
                            continue
                        for MORA in TransSUW.getchildren():

                            if MORA.tag != "Mora":
                                if MORA.tag not in ["NonLinguisticSound"]:
                                    print("MORA TAG: ", MORA.tag)
                                continue

                            if "MoraEntity" in MORA.attrib.keys():
                                # speaker_mora.append(MORA.attrib['MoraEntity'])

                                word_mora.append(MORA.attrib["MoraEntity"])

                            for Phoneme in MORA.getchildren():
                                if Phoneme.tag != "Phoneme":
                                    print("Phoneme TAG: ", Phoneme.tag)
                                    continue

                                IPU_phonemes.append(Phoneme.attrib["PhonemeEntity"])
                                word_phonemes.append(Phoneme.attrib["PhonemeEntity"])

                                # speaker_phonemes.append(
                                #    Phoneme.attrib['PhonemeEntity'])

                                for Phone in Phoneme.getchildren():
                                    if Phone.tag != "Phone":
                                        print("Phone TAG: ", Phone.tag)
                                        continue

                                    word_phones.append(Phone.attrib["PhoneEntity"])
                                    word_phoneclass.append(Phone.attrib["PhoneClass"])

                                    phone_lens.append(
                                        float(Phone.attrib["PhoneEndTime"])
                                        - float(Phone.attrib["PhoneStartTime"])
                                    )

                                    for Tone in Phone.getchildren():
                                        if Tone.tag not in [
                                            "XJToBILabelWord",
                                            "XJToBILabelBreak",
                                            "XJToBILabelTone",
                                            "XJToBILabelPrm",
                                            "XJToBILabelMisc",
                                            "XJToBILabelTone",
                                        ]:
                                            print("Tone TAG: ", Tone.tag)
                                            continue

                        speaker_phones.append(word_phones)
                        speaker_mora.append(word_mora)
                        speaker_phonemes.append(word_phonemes)
                        speaker_phoneclass.append(word_phoneclass)
            speaker_IPU_phonemes.append(IPU_phonemes)
        words.append(speaker_words)
        pos.append(speaker_pos)
        mora.append(speaker_mora)
        phonemes.append(speaker_phonemes)
        phones.append(speaker_phones)
        phone_class.append(speaker_phoneclass)
        session_lens.append(session_IPU_lens)
        all_IPU_phonemes.append(speaker_IPU_phonemes)

    return (
        words,
        pos,
        mora,
        phonemes,
        phones,
        phone_class,
        session_lens,
        IPU_lens,
        phone_lens,
        word_lens,
        session_lens,
        all_IPU_phonemes,
    )


def prep_BUCKEYE(speaker_list):
    track_durations = []
    word_durations = []
    utterance_durations = []
    phone_durations = []

    all_words = []
    all_phonemic = []
    all_phonetic = []
    all_pos = []
    for speaker_loc in tqdm(speaker_list):
        speaker_words = []
        speaker_phonemic = []
        speaker_phonetic = []
        speaker_pos = []
        speaker = buckeye.Speaker.from_zip(speaker_loc, load_wavs=True)
        for track in tqdm([track for track in speaker], leave=False):
            track_durations.append(track.words[-1].end - track.words[0].beg)
            for utterance in words_to_utterances(track.words):
                utterance_durations.append(utterance.dur)
                words = []
                phonemic = []
                phonetic = []
                pos = []
                for word in utterance:
                    if hasattr(word, "phonemic"):  # if this is not a pause, etc
                        words.append(word.orthography)
                        word_durations.append(word.dur)
                        if word.phonemic is not None:
                            phonemic.append(word.phonemic)
                        if word.phonetic is not None:
                            phonetic.append(word.phonetic)
                        if word.phonemic is not None:
                            for phone in word.phones:
                                phone_durations.append(phone.dur)
                        pos.append(word.pos)
                speaker_words.append(words)
                speaker_phonemic.append(phonemic)
                speaker_phonetic.append(phonetic)
                speaker_pos.append(pos)
        all_words.append(speaker_words)
        all_phonemic.append(speaker_phonemic)
        all_phonetic.append(speaker_phonetic)
        all_pos.append(speaker_pos)

    return (
        track_durations,
        word_durations,
        utterance_durations,
        phone_durations,
        all_words,
        all_phonemic,
        all_phonetic,
        all_pos,
    )


def prep_GECO(text_grids):
    """ 
    """
    track_durations = []
    word_durations = []
    phone_durations = []
    syll_durations = []

    all_words = []
    all_sylls = []
    all_phones = []

    for grid_loc in tqdm(text_grids, leave=False):

        grid_words = []
        grid_phones = []
        grid_sylls = []

        # get grid info
        grid = textgrid.TextGrid.fromFile(grid_loc)
        tiers_dict = {j.name: i for i, j in enumerate(grid.tiers)}
        words = grid.tiers[tiers_dict["words"]]
        phones = grid.tiers[tiers_dict["phones"]]
        sylls = grid.tiers[tiers_dict["syls"]]

        track_durations.append(words[-1].maxTime - words[0].minTime)

        for word in words:
            if word.mark[0] == "<":
                continue

            syll_start_i = sylls.indexContaining(float(word.minTime) + 1e-4)
            syll_end_i = sylls.indexContaining(float(word.maxTime) - 1e-4)
            syll_marks = [i.mark for i in sylls[syll_start_i : syll_end_i + 1]]

            phone_start_i = phones.indexContaining(float(word.minTime) + 1e-4)
            phone_end_i = phones.indexContaining(float(word.maxTime) - 1e-4)
            phone_marks = [i.mark for i in phones[phone_start_i : phone_end_i + 1]]

            phone_durs = [
                float(i.duration()) for i in phones[phone_start_i : phone_end_i + 1]
            ]
            syll_durs = [
                float(i.duration()) for i in sylls[syll_start_i : syll_end_i + 1]
            ]

            syll_durations.append(syll_durs)
            phone_durations.append(phone_durs)
            word_durations.append(float(word.duration()))

            grid_sylls.append(syll_marks)
            grid_phones.append(phone_marks)
            grid_words.append(word.mark)

            word_durations.append(float(word.duration()))

        all_words.append(grid_words)
        all_sylls.append(grid_sylls)
        all_phones.append(grid_phones)

    return (
        track_durations,
        word_durations,
        phone_durations,
        syll_durations,
        all_words,
        all_sylls,
        all_phones,
    )
