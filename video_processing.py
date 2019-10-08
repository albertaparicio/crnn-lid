# Get predictions for every channel in a video
#
# Author: Albert Aparicio <albert.aparicio.-nd@disneyresearch.com>
import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

from tqdm import tqdm

from keras_custom.predict import predict


def extract_tracks(filepath: Path):
    res = subprocess.check_output(
        f"ffprobe -v 0 "
        f"-show_entries stream=index,codec_name,codec_type "
        f"'{filepath}' -print_format json",
        shell=True,
    )

    res = json.loads(res)

    fname = filepath.stem
    streams_dir = Path("/tmp").joinpath(fname)
    streams_dir.mkdir(exist_ok=True)

    num_audio = 0
    for s in tqdm(res["streams"], desc="Extracting audio tracks"):
        if s["codec_type"] == "audio":
            num_audio += 1

            subprocess.call(
                f"ffmpeg -y -v 0 -i '{filepath}' -vn -map 0:{s['index']} '{streams_dir}/audio_{s['index']}.wav'",
                shell=True,
            )

    return fname, num_audio, streams_dir


def main(args):
    fname, num_tracks, streams_dir = extract_tracks(Path(args.input_file))

    results = {}
    for track_num, track in enumerate(sorted(streams_dir.glob("*.wav"))):
        args.input_file = track

        prediction, _ = predict(args)

        results[str(track_num + 1)] = prediction

    print(json.dumps({"channel_languages": results}, indent=4))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", dest="model_dir", required=True)
    parser.add_argument("--input", dest="input_file", required=True)
    cli_args = parser.parse_args()

    if not os.path.isfile(cli_args.input_file):
        sys.exit("Input is not a file.")

    main(cli_args)
